import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from PIL import Image
import random

# ====== Set transform for resizing ======
resize_transform = transforms.Compose([
    transforms.Resize((512, 512), interpolation=InterpolationMode.BICUBIC),
])

# ====== Load datasets ======
emnist = datasets.EMNIST(root='data', split='byclass', train=False, download=True)
cifar10 = datasets.CIFAR10(root='data', train=False, download=True)
food101 = datasets.Food101(root='data', split='test', download=True)

# ====== Pick random sample from each dataset ======
idx_emnist = random.randint(0, len(emnist)-1)
idx_cifar = random.randint(0, len(cifar10)-1)
idx_food = random.randint(0, len(food101)-1)

img_emnist, label_emnist = emnist[idx_emnist]
img_cifar, label_cifar = cifar10[idx_cifar]
img_food, label_food = food101[idx_food]

# ====== Fix EMNIST orientation ======
img_emnist = img_emnist.transpose(Image.ROTATE_270)  # 90 deg clockwise
img_emnist = img_emnist.transpose(Image.FLIP_LEFT_RIGHT)  # Flip horizontally
img_emnist = img_emnist.convert("RGB")  # Convert to RGB

# Resize all images
img_emnist = resize_transform(img_emnist)
img_cifar = resize_transform(img_cifar)
img_food = resize_transform(img_food)

# ====== Prepare labels ======
emnist_class = f"Class {label_emnist} (EMNIST)"
cifar_class = f"{cifar10.classes[label_cifar]} (CIFAR-10)"
food_class = f"{food101.classes[label_food]} (Food-101)"

# ====== Plotting and saving ======
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for ax, img, title in zip(axs, [img_emnist, img_cifar, img_food], [emnist_class, cifar_class, food_class]):
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

plt.tight_layout(pad=3)
plt.savefig("comparison_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# Minh họa EMNIST, CIFAR-10 và Food-101 với kích thước 512x512