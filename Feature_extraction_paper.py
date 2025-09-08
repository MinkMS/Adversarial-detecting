import os
import torch
from torchvision import models, transforms
from torchvision.datasets.folder import default_loader
from timm import create_model
import pandas as pd
from tqdm import tqdm
import numpy as np
from Squeeze_func_paper import reduce_bit_depth, median_filter, rgb_channel_squeeze, extract_features

# ===== CONFIG =====
DATASET_ROOT = r"C:\Users\Mink\OneDrive\Documents\GitHub\squeeze_dataset"
OUTPUT_CSV = "features_combined.csv"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===== TRANSFORM =====
TRANSFORM = transforms.Compose([
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== FEATURE HEADERS =====
squeezer_names = ['bit', 'median', 'rgb']
metric_names = ['conf_drop', 'kl', 'changed', 'entropy']
squeezers = [reduce_bit_depth, median_filter, rgb_channel_squeeze]

def process_model_folder(folder_path, model_name):
    print(f"\n🔍 Processing: {model_name}")
    model_path = os.path.join(folder_path, "model.pth")
    clean_dir = os.path.join(folder_path, "clean")
    defected_dir = os.path.join(folder_path, "defected")

    # Load model
    if model_name == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 101)
    else:
        model = create_model(model_name, pretrained=False, num_classes=101)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval().to(DEVICE)

    rows = []
    for label, dir_path in [(0, clean_dir), (1, defected_dir)]:
        image_files = [f for f in os.listdir(dir_path) if f.endswith('.png')]
        for fname in tqdm(image_files, desc=f"{model_name} ({'clean' if label==0 else 'defected'})"):
            img_path = os.path.join(dir_path, fname)
            try:
                img = default_loader(img_path)
                img_tensor = TRANSFORM(img).to(DEVICE)

                feats = extract_features(model, img_tensor, squeezers, squeezer_names)

                row = {
                    "image": fname,
                    "label": label,
                    "source_model": model_name
                }
                row.update(feats)
                rows.append(row)
            except Exception as e:
                print(f"Skipping {fname}: {e}")
    return rows

def main():
    print("Starting feature extraction...")
    if DEVICE == 'cuda':
        print("Using GPU for processing.") 
    else:
        print("Using CPU for processing. This may be slow.")

    all_rows = []
    subfolders = [f for f in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, f))]

    for sub in subfolders:
        folder = os.path.join(DATASET_ROOT, sub)
        rows = process_model_folder(folder, sub)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # ===== CLEAN NaN / Inf =====
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFeature extraction completed. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

#Script extract cái feature của ảnh theo paper và lưu vào file csv để train mô hình Random Forest