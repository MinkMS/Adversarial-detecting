import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms
from timm import create_model

from Squeeze_func_me import (
    extract_features as extract_me,
    reduce_bit_depth,
    median_filter,
    rgb_channel_squeeze,
    jpeg_compression,
    resize_squeeze,
    hist_equalize
)

from Squeeze_func_paper import extract_features as extract_paper

# ========== CONFIG ==========
DATASET_ROOT = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\adversarial-eval"
CSV_PAPER = "features_eval_paper.csv"
CSV_ME = "features_eval_me.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ========== TRANSFORM ==========
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========== SQUEEZERS ==========
squeezers = [
    reduce_bit_depth,
    median_filter,
    rgb_channel_squeeze,
    jpeg_compression,
    resize_squeeze,
    hist_equalize
]
squeezer_names = ['bit', 'median', 'rgb', 'jpeg', 'resize', 'hist']

# ========== MODEL LOADER ==========
def load_model(model_name):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 101)
        ckpt = "food101_resnet18.pth"
    else:
        model = create_model("efficientnet_b0", pretrained=False, num_classes=101)
        ckpt = "efficientnet_food101_best.pth"
    
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return model.to(DEVICE).eval()

# ========== PROCESS FUNCTION ==========
def process_folder(label_folder, label):
    rows_paper = []
    rows_me = []

    for model_name in ['resnet18', 'efficientnet_b0']:
        sub_dir = os.path.join(label_folder, model_name)
        if not os.path.exists(sub_dir): continue

        model = load_model(model_name)
        img_files = sorted([
            f for f in os.listdir(sub_dir) if f.lower().endswith(('.png', '.jpg'))
        ])

        print(f"[{label} / {model_name}] Found {len(img_files)} images.")
        for fname in tqdm(img_files, desc=f"{label}/{model_name}"):
            try:
                path = os.path.join(sub_dir, fname)
                img = Image.open(path).convert("RGB")
                x = transform(img).to(DEVICE)

                feat_paper = extract_paper(model, x, squeezers[:3], squeezer_names[:3])
                feat_me = extract_me(model, x, squeezers, squeezer_names)

                common_info = {
                    'image': fname,
                    'label': 0 if label == 'clean' else 1,
                    'source_model': model_name
                }

                feat_paper.update(common_info)
                feat_me.update(common_info)

                rows_paper.append(feat_paper)
                rows_me.append(feat_me)

            except Exception as e:
                print(f"Error processing {fname}: {e}")

    return rows_paper, rows_me

# ========== MAIN ==========
def main():
    torch.manual_seed(42)
    all_paper, all_me = [], []

    for label in ['clean', 'defected']:
        folder = os.path.join(DATASET_ROOT, label)
        paper_rows, me_rows = process_folder(folder, label)
        all_paper.extend(paper_rows)
        all_me.extend(me_rows)

    pd.DataFrame(all_paper).to_csv(CSV_PAPER, index=False)
    pd.DataFrame(all_me).to_csv(CSV_ME, index=False)
    print(f"\nSaved {CSV_PAPER} and {CSV_ME}")

if __name__ == "__main__":
    main()

# Tạo file csv đánh giá random forest