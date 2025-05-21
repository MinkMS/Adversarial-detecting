import os
from torchvision.datasets import STL10
from PIL import Image
from tqdm import tqdm

out_root = 'datasets/stl10'
os.makedirs(out_root, exist_ok=True)

dataset = STL10(root='./data', split='train', download=True)
classes = dataset.classes

for idx, (img, label) in tqdm(enumerate(dataset), total=len(dataset), desc='Saving STL-10'):
    cls_name = classes[label]
    save_dir = os.path.join(out_root, cls_name)
    os.makedirs(save_dir, exist_ok=True)

    img_pil = img  # đã là PIL.Image
    img_pil.save(os.path.join(save_dir, f'{idx:05}.png'))
# Chuyển đổi ảnh STL-10 thành định dạng folder