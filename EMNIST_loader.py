import os
import torch
from torchvision.datasets import EMNIST
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

out_root = 'datasets/emnist_byclass'
os.makedirs(out_root, exist_ok=True)

dataset = EMNIST(root='./data', split='byclass', train=True, download=True)
print(f"Loaded EMNIST with {len(dataset)} samples and {len(dataset.classes)} classes.")

to_pil = transforms.ToPILImage()

for idx, (img, label) in tqdm(enumerate(dataset), total=len(dataset), desc='Saving EMNIST-byclass'):
    char = dataset.classes[label]
    save_dir = os.path.join(out_root, char)
    os.makedirs(save_dir, exist_ok=True)

    # Xử lý xoay ảnh đúng hướng
    if isinstance(img, torch.Tensor):
        img_pil = to_pil(img.transpose(1, 2)).rotate(270)
    else:
        img_pil = img.transpose(method=Image.ROTATE_270)

    img_pil.save(os.path.join(save_dir, f'{idx:05}.png'))
# Chuyển đổi ảnh EMNIST thành định dạng folder