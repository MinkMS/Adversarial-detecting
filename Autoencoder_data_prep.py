import os
import random
import shutil
from tqdm import tqdm

# ===== CONFIG =====
SRC_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\Food-101-rf\train"
DEST_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\Food-101-rf\ae_data"
TRAIN_RATIO = 0.3
VAL_RATIO = 0.1
TOTAL_RATIO = TRAIN_RATIO + VAL_RATIO

random.seed(42)

# ===== INIT =====
os.makedirs(os.path.join(DEST_DIR, 'train'), exist_ok=True)
os.makedirs(os.path.join(DEST_DIR, 'val'), exist_ok=True)

# ===== COLLECT IMAGES =====
all_images = []
for subfolder in ['clean', 'defected']:
    subdir = os.path.join(SRC_DIR, subfolder)
    for img_name in os.listdir(subdir):
        if img_name.lower().endswith((".jpg", ".png")):
            all_images.append(os.path.join(subdir, img_name))

# ===== SELECT & SPLIT =====
total_used = int(len(all_images) * TOTAL_RATIO)
split_train = int(total_used * (TRAIN_RATIO / TOTAL_RATIO))

selected = random.sample(all_images, total_used)
train_imgs = selected[:split_train]
val_imgs = selected[split_train:]

# ===== COPY =====
for img_path in tqdm(train_imgs, desc="Copying train"):
    fname = os.path.basename(img_path)
    shutil.copy2(img_path, os.path.join(DEST_DIR, 'train', fname))

for img_path in tqdm(val_imgs, desc="Copying val"):
    fname = os.path.basename(img_path)
    shutil.copy2(img_path, os.path.join(DEST_DIR, 'val', fname))

print("Done. AE dataset created at:", DEST_DIR)
