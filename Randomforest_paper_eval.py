import os
import joblib
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from Squeeze_func_paper import extract_features

# ========== CONFIG ==========
MODEL_PATH = "rf_detector_paper.pkl"
DATA_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\adversarial-eval"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_images_from_folder(folder_path, label):
    data = []
    for file in os.listdir(folder_path):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(folder_path, file)
        img = Image.open(img_path).convert("RGB")
        tensor_img = transform(img).unsqueeze(0)
        feat = extract_features(tensor_img)
        data.append((feat, label, file))
    return data

def main():
    print(f"Loading model: {MODEL_PATH}")
    rf = joblib.load(MODEL_PATH)

    all_data = []
    for model_name in ['resnet18', 'efficientnet_b0']:
        print(f"\nProcessing model: {model_name}")
        clean_dir = os.path.join(DATA_DIR, "clean", model_name)
        defected_dir = os.path.join(DATA_DIR, "defected", model_name)

        all_data += load_images_from_folder(clean_dir, 0)
        all_data += load_images_from_folder(defected_dir, 1)

    X = np.array([d[0] for d in all_data])
    y_true = np.array([d[1] for d in all_data])
    file_names = [d[2] for d in all_data]

    print("Running inference...")
    y_pred = rf.predict(X)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

    df = pd.DataFrame({
        "image": file_names,
        "true": y_true,
        "pred": y_pred
    })
    df.to_csv("rf_eval_results.csv", index=False)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()
