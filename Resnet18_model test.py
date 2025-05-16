import os
import random
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt

# ========= CONFIG =========
MODEL_PATH = r'C:\Users\Mink\OneDrive\Documents\GitHub\Adversarial-detecting\resnet18_128x128.pth'
DATA_ROOT = 'Adversarial-Example'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 128

# ========= LOAD MODEL =========
model = resnet18(num_classes=10)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()

# ========= CIFAR-10 CLASS LABELS =========
cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                 'dog', 'frog', 'horse', 'ship', 'truck']

# ========= CHOOSE RANDOM IMAGE =========
# Chọn split & attack
split = random.choice(['clean', 'defected/fgsm', 'defected/pgd'])

# Lấy class folder
split_path = os.path.join(DATA_ROOT, split)
cls = random.choice(os.listdir(split_path))

# Lấy ảnh
cls_path = os.path.join(split_path, cls)
img_name = random.choice(os.listdir(cls_path))
img_path = os.path.join(cls_path, img_name)

# ========= LOAD & PREPROCESS IMAGE =========
img_pil = Image.open(img_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

# ========= PREDICT =========
with torch.no_grad():
    output = model(img_tensor)
    pred_class_idx = output.argmax(dim=1).item()
    pred_class = cifar_classes[pred_class_idx]

# ========= DISPLAY INFO & IMAGE =========
print("Image path:", img_path)
print("True class (folder):", cls)
print("Type:", 'clean' if split == 'clean' else split.split('/')[-1])
print("Model prediction:", pred_class)

plt.imshow(img_pil)
plt.title(f"Predicted: {pred_class.upper()} | True: {cls.upper()}")
plt.axis('off')
plt.show()