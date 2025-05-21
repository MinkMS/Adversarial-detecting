import os
from torchvision.datasets import CIFAR10
from PIL import Image
from tqdm import tqdm

out_root = 'datasets/cifar10'
os.makedirs(out_root, exist_ok=True)

dataset = CIFAR10(root='./data', train=True, download=True)
classes = dataset.classes

for idx, (img, label) in tqdm(enumerate(dataset), total=len(dataset), desc='Saving CIFAR-10'):
    cls_name = classes[label]
    save_dir = os.path.join(out_root, cls_name)
    os.makedirs(save_dir, exist_ok=True)

    img.save(os.path.join(save_dir, f'{idx:05}.png'))

# Scrpit chuyển đổi ảnh CIFAR-10 thành định dạng folder