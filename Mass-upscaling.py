import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# 1. Config
output_dir = "data/clean/"
resize_size = 128

os.makedirs(output_dir, exist_ok=True)

# 2. Load CIFAR-10
transform = transforms.ToTensor()
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# 3. Upscale và save từng ảnh
for idx in tqdm(range(len(dataset)), desc="Upscaling CIFAR-10 images"):
    img_tensor, label = dataset[idx]
    img_np = img_tensor.permute(1,2,0).numpy()  # CHW -> HWC
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    
    # Resize bằng bicubic
    img_upscaled = img_pil.resize((resize_size, resize_size), resample=Image.BICUBIC)

    # Save ảnh dạng PNG
    save_path = os.path.join(output_dir, f"img_{idx:05}.png")
    img_upscaled.save(save_path, format='PNG', optimize=True)
