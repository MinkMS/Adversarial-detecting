import os
import shutil

# ====== CONFIG ======
ROOT_DIR = r'C:\Users\Mink\OneDrive\Documents\GitHub\data\Food-101'
SRC_IMG_DIR = os.path.join(ROOT_DIR, 'images')
META_DIR = os.path.join(ROOT_DIR, 'meta')
DEST_DIR = ROOT_DIR + '-split'

# ====== Tạo folder đích ======
for split in ['train', 'test']:
    split_file = os.path.join(META_DIR, f'{split}.txt')
    split_path = os.path.join(DEST_DIR, split)
    os.makedirs(split_path, exist_ok=True)

    with open(split_file, 'r') as f:
        lines = [line.strip() for line in f]

    for rel_path in lines:
        cls, img = rel_path.split('/')
        src = os.path.join(SRC_IMG_DIR, cls, f'{img}.jpg')
        dst_cls_folder = os.path.join(split_path, cls)
        os.makedirs(dst_cls_folder, exist_ok=True)
        dst = os.path.join(dst_cls_folder, f'{img}.jpg')
        shutil.copyfile(src, dst)

    print(f"Done copying {split} set to: {split_path}")
