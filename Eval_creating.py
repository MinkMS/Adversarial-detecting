import os
import random
import torch
import torch.nn as nn
from torchvision import models, transforms
from timm import create_model
from PIL import Image
from tqdm import tqdm
import torchattacks

# ===== CONFIG =====
SRC_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\Food101_unused_half\clean"
DEST_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\adversarial-eval"
IMG_SIZE = 512
N_SAMPLES_PER_CLASS = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Attack params
EPSILON = 10 / 255
ALPHA = 10 / 255
STEPS = 10

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
inv_transform = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)

# Create output folders
def init_folders():
    for split in ["clean", "defected"]:
        for model_name in ["resnet18", "efficientnet_b0"]:
            os.makedirs(os.path.join(DEST_DIR, split, model_name), exist_ok=True)

# Load classification model
def load_model(model_name):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 101)
        model.load_state_dict(torch.load("food101_resnet18.pth", map_location=DEVICE))
    elif model_name == "efficientnet_b0":
        model = create_model("efficientnet_b0", pretrained=False, num_classes=101)
        model.load_state_dict(torch.load("efficientnet_food101_best.pth", map_location=DEVICE))
    else:
        raise ValueError("Unsupported model name.")
    return model.to(DEVICE).eval()

# Save tensor as image
def save_image(tensor, path):
    img = inv_transform(tensor.squeeze().cpu()).clamp(0, 1)
    Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype('uint8')).save(path)

# Apply adversarial attacks
def attack_image(model, x, y, img_name, model_name):
    x = x.unsqueeze(0).to(DEVICE)
    y = torch.tensor([y]).to(DEVICE)
    attacks = {
        "fgsm": torchattacks.FGSM(model, eps=EPSILON),
        "bim": torchattacks.BIM(model, eps=EPSILON, alpha=ALPHA, steps=STEPS),
        "pgd": torchattacks.PGD(model, eps=EPSILON, alpha=ALPHA, steps=STEPS)
    }
    for atk_name, atk in attacks.items():
        adv_img = atk(x, y)[0]
        adv_path = os.path.join(DEST_DIR, "defected", model_name, f"{img_name}-{atk_name}.png")
        save_image(adv_img, adv_path)

def main():
    torch.manual_seed(42)
    random.seed(42)
    init_folders()

    class_list = sorted(os.listdir(SRC_DIR))
    label_map = {cls_name: idx for idx, cls_name in enumerate(class_list)}

    for cls_name in tqdm(class_list, desc="Processing classes"):
        cls_path = os.path.join(SRC_DIR, cls_name)
        all_imgs = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png'))]
        random.shuffle(all_imgs)
        samples = all_imgs[:N_SAMPLES_PER_CLASS]

        for i, fname in enumerate(samples):
            img_path = os.path.join(cls_path, fname)
            img = Image.open(img_path).convert("RGB")
            x = transform(img)
            label = label_map[cls_name]
            model_type = "resnet18" if i < N_SAMPLES_PER_CLASS // 2 else "efficientnet_b0"
            out_name = f"{cls_name}-{i:04d}.png"

            # Save clean image
            clean_path = os.path.join(DEST_DIR, "clean", model_type, out_name)
            save_image(x, clean_path)

            # Run attack
            model = load_model(model_type)
            attack_image(model, x, label, out_name.replace(".png", ""), model_type)

    print("Adversarial-eval dataset created.")

if __name__ == "__main__":
    main()

# Tạo dataset để đánh giá autoencoder và random forest