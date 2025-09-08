import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

SRC_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\adversarial-eval"
DEST_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\adversarial-eval-split"
SPLIT_RATIO = 0.75  # 70% train, 30% val
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def prepare_split():
    for label in ["clean", "defected"]:
        for model_name in ["resnet18", "efficientnet_b0"]:
            src_folder = Path(SRC_DIR) / label / model_name
            files = sorted([f for f in src_folder.glob("*.png")])

            random.shuffle(files)
            split_idx = int(len(files) * SPLIT_RATIO)
            train_files = files[:split_idx]
            val_files = files[split_idx:]

            for split_name, split_files in [("train", train_files), ("val", val_files)]:
                dest_folder = Path(DEST_DIR) / split_name / label
                dest_folder.mkdir(parents=True, exist_ok=True)
                for file in split_files:
                    shutil.copy(file, dest_folder / file.name)

    print("Split done. New structure ready at:", DEST_DIR)

if __name__ == "__main__":
    prepare_split()
