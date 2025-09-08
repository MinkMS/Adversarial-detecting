import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms
from timm import create_model

from Squeeze_func_me import (
    extract_features,
    reduce_bit_depth,
    median_filter,
    rgb_channel_squeeze,
    jpeg_compression,
    resize_squeeze,
    hist_equalize
)

# ===== CONFIG =====
DATASET_ROOT = r"C:\Users\Mink\OneDrive\Documents\GitHub\squeeze_dataset"
OUTPUT_CSV = "features_combined_extended.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== SQUEEZERS =====
squeezers = [
    reduce_bit_depth,
    median_filter,
    rgb_channel_squeeze,
    jpeg_compression,
    resize_squeeze,
    hist_equalize
]
squeezer_names = ['bit', 'median', 'rgb', 'jpeg', 'resize', 'hist']

# ===== TRANSFORM =====
transform = transforms.Compose([
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model(model_name, model_path):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 101)
    else:
        model = create_model(model_name, pretrained=False, num_classes=101)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model

def main():
    records = []

    for model_name in os.listdir(DATASET_ROOT):
        model_path = os.path.join(DATASET_ROOT, model_name, "model.pth")
        if not os.path.exists(model_path):
            continue

        model = load_model(model_name, model_path)
        print(f"Processing model: {model_name}")

        for label in ['clean', 'defected']:
            folder = os.path.join(DATASET_ROOT, model_name, label)
            images = os.listdir(folder)

            for img_name in tqdm(images, desc=f"{model_name} ({label})"):
                img_path = os.path.join(folder, img_name)
                try:
                    img = Image.open(img_path).convert("RGB")
                    x = transform(img).to(DEVICE)
                    feats = extract_features(model, x, squeezers, squeezer_names)
                    feats['image'] = img_name
                    feats['label'] = 0 if label == "clean" else 1
                    feats['source_model'] = model_name
                    records.append(feats)
                except Exception as e:
                    print(f"Error on {img_path}: {e}")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFeature extraction complete → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

# Script để extract feature theo feature tự chọn