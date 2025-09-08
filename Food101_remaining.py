import os
import random
import shutil
from tqdm import tqdm

# ===== CONFIG =====
RAW_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\Food-101\images"
USED_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\squeeze_dataset"
OUTPUT_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\Food101_unused_half"

IMG_EXT = ('.jpg', '.png')

def get_all_used_image_names(used_dir):
    used_names = set()
    for model_name in os.listdir(used_dir):
        model_path = os.path.join(used_dir, model_name)
        for typ in ['clean', 'defected']:
            type_path = os.path.join(model_path, typ)
            if not os.path.exists(type_path):
                continue
            for fname in os.listdir(type_path):
                base = fname.split('-')[0]
                used_names.add(base)
    return used_names

def copy_remaining_images(used_set, output_dir, portion=0.5):
    all_images = []
    for cls in os.listdir(RAW_DIR):
        class_dir = os.path.join(RAW_DIR, cls)
        for fname in os.listdir(class_dir):
            if fname.endswith(IMG_EXT):
                base = fname.split('.')[0]
                if base not in used_set:
                    all_images.append((cls, fname))

    # Shuffle and take portion
    random.shuffle(all_images)
    selected = all_images[:int(len(all_images) * portion)]
    
    for cls, fname in tqdm(selected, desc="Copying new eval set"):
        src = os.path.join(RAW_DIR, cls, fname)
        tgt_dir = os.path.join(output_dir, 'clean', cls)
        os.makedirs(tgt_dir, exist_ok=True)
        shutil.copy(src, os.path.join(tgt_dir, fname))

    print(f"Copied {len(selected)} images to clean folder.")

if __name__ == "__main__":
    used_img_names = get_all_used_image_names(USED_DIR)
    print(f"Used images: {len(used_img_names)}")
    copy_remaining_images(used_img_names, OUTPUT_DIR)
