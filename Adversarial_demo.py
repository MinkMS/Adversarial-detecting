import os
import random
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import torchattacks

# ========== CONFIG ==========
clean_dir = r'C:\Users\Mink\OneDrive\Documents\GitHub\data\clean'
model_path = r'C:\Users\Mink\OneDrive\Documents\GitHub\Adversarial-detecting\resnet18_128x128.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== LOAD RANDOM CLEAN IMAGE ==========
image_files = os.listdir(clean_dir)
random_image = random.choice(image_files)
img_path = os.path.join(clean_dir, random_image)

img_pil = Image.open(img_path).convert('RGB')
transform = transforms.Compose([
    transforms.ToTensor(),  # [0,1]
])
img_tensor = transform(img_pil).unsqueeze(0).to(device)  # [1, 3, 128, 128]

# ========== LOAD MODEL ==========
model = models.resnet18(num_classes=10)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device).eval()

# Dummy label nếu không có ground-truth
dummy_label = torch.tensor([0]).to(device)

# ========== FGSM ATTACK ==========
fgsm_attack = torchattacks.FGSM(model, eps=8/255)
img_fgsm = fgsm_attack(img_tensor, dummy_label)

# ========== PGD ATTACK ==========
pgd_attack = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10)
img_pgd = pgd_attack(img_tensor, dummy_label)

# ========== VISUALIZE ==========
def tensor_to_numpy(img):
    img = img.squeeze().detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))  # [H, W, C]
    return np.clip(img, 0, 1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(tensor_to_numpy(img_tensor))
plt.title('Clean Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(tensor_to_numpy(img_fgsm))
plt.title('FGSM Attacked')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(tensor_to_numpy(img_pgd))
plt.title('PGD Attacked')
plt.axis('off')

plt.tight_layout()
plt.suptitle(f'Compare Attacks - {random_image}', fontsize=14, y=1.05)
plt.show()
