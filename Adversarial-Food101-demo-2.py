import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from timm import create_model

# ===== CONFIG =====
DATA_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\Food-101-split\test"
MODEL_PATH = "efficientnet_food101_best.pth"
IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPSILON = 10 / 255
ALPHA = 10 / 255
PGD_STEPS = 10

# ===== TRANSFORMS =====
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

inv_transform = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.],
                         std=[1/0.229, 1/0.224, 1/0.225]),
    transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                         std=[1., 1., 1.])
])

# ===== LOAD MODEL =====
model = create_model("efficientnet_b0", pretrained=False, num_classes=101).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===== LOAD LABELS =====
class_names = sorted(os.listdir(DATA_DIR))

def load_random_image():
    class_folder = random.choice(class_names)
    class_path = os.path.join(DATA_DIR, class_folder)
    image_file = random.choice(os.listdir(class_path))
    img_path = os.path.join(class_path, image_file)
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    return tensor, class_folder, img_path

# ===== ATTACK FUNCTIONS =====
def fgsm(x, y, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        output = model(x_adv)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
    x_adv = x_adv + eps * x_adv.grad.sign()
    return torch.clamp(x_adv, 0, 1)

def bim(x, y, eps, alpha=0.005, steps=100):
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        output = model(x_adv)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
        x_adv = torch.clamp(x_adv, 0, 1).detach()
    return x_adv

def pgd(x, y, eps, alpha=0.005, steps=100):
    x_adv = x + torch.empty_like(x).uniform_(-eps, eps).to(DEVICE)
    return bim(x_adv, y, eps, alpha, steps)

# ===== VISUALIZATION =====
def imshow(imgs, titles):
    imgs = [inv_transform(img.squeeze().detach().cpu()).permute(1, 2, 0).numpy() for img in imgs]
    imgs = [np.clip(img, 0, 1) for img in imgs]
    plt.figure(figsize=(12, 4))
    for i, img in enumerate(imgs):
        plt.subplot(1, 4, i+1)
        plt.imshow(img)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# ===== MAIN =====
x, true_class, img_path = load_random_image()
y = torch.tensor([class_names.index(true_class)]).to(DEVICE)

# Predict clean
with torch.no_grad():
    pred = model(x).argmax(1).item()

# Attack
x_fgsm = fgsm(x, y, EPSILON)
x_bim = bim(x, y, EPSILON, ALPHA, 10)
x_pgd = pgd(x, y, EPSILON, ALPHA, PGD_STEPS)

# Predictions
with torch.no_grad():
    pred_fgsm = model(x_fgsm).argmax(1).item()
    pred_bim = model(x_bim).argmax(1).item()
    pred_pgd = model(x_pgd).argmax(1).item()

# PRINT RESULT
print(f"\nOriginal Image: {os.path.basename(img_path)}")
print(f"True Label:      {true_class}")
print(f"Pred (Clean):    {class_names[pred]}")
print(f"Pred (FGSM):     {class_names[pred_fgsm]}")
print(f"Pred (BIM):      {class_names[pred_bim]}")
print(f"Pred (PGD):      {class_names[pred_pgd]}\n")

# SHOW
def imshow(imgs, titles):
    imgs = [inv_transform(img.squeeze().detach().cpu()).permute(1, 2, 0).numpy() for img in imgs]
    imgs = [np.clip(img, 0, 1) for img in imgs]
    plt.figure(figsize=(12, 4))
    for i, img in enumerate(imgs):
        plt.subplot(1, 4, i+1)
        plt.imshow(img)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("attack_visualization.png", dpi=300, bbox_inches='tight')  # <<== Lưu ảnh ở đây
    plt.show()

# Visuallize ảnh bị tấn công