import os
import random
import joblib
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import warnings
from torchvision import transforms, models
from PIL import Image
from Autoencoder_train import Autoencoder
from Squeeze_func_me import extract_features as extract_me
from Squeeze_func_paper import extract_features as extract_paper

# ===== CONFIG =====
warnings.filterwarnings("ignore")
IMAGE_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\Food-101_final_eval"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512

# ===== TRANSFORMS =====
transform_tensor = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

transform_tensor_norm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== CNN MODEL =====
def load_cnn():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("cnn_classifier.pth", map_location=DEVICE))
    return model.to(DEVICE).eval()

# ===== AUTOENCODER =====
def load_autoencoder():
    model = Autoencoder()
    model.load_state_dict(torch.load("autoencoder_model.pth", map_location=DEVICE))
    return model.to(DEVICE).eval()

# ===== RF MODEL =====
def load_rf(model_name):
    if model_name == "rf_me":
        return joblib.load("rf_detector_me.pkl"), "me"
    elif model_name == "rf_paper":
        return joblib.load("rf_detector_paper.pkl"), "paper"

# ===== FEATURE EXTRACTOR MODEL =====
def load_feature_cnn():
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("cnn_classifier.pth", map_location=DEVICE))
    return model.to(DEVICE).eval()

# ===== SELECTED IMAGE =====
def load_random_image_from(label):
    folder = os.path.join(IMAGE_DIR, label)
    img_file = random.choice(os.listdir(folder))
    img_path = os.path.join(folder, img_file)
    return Image.open(img_path).convert("RGB"), label, img_file

# ===== SHOW IMAGE =====
def show_prediction(img_pil, pred_label, true_label, file_name, model_name):
    plt.figure(figsize=(5, 5))
    plt.imshow(img_pil)
    plt.title(f"{model_name.upper()} Prediction: {pred_label.upper()}\nGround Truth: {true_label.upper()}\nFile: {file_name}", fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ===== MAIN =====
def main():
    choice = input("Chọn model (cnn / autoencoder / rf_me / rf_paper): ").strip().lower()
    label_choice = input("Chọn folder ảnh (clean / defected): ").strip().lower()

    if choice not in ["cnn", "autoencoder", "rf_me", "rf_paper"]:
        print("Model sai.")
        return
    if label_choice not in ["clean", "defected"]:
        print("Folder sai (chọn clean hoặc defected).")
        return

    img_pil, true_label, filename = load_random_image_from(label_choice)
    print(f"Selected: {filename} | Ground Truth: {true_label}")

    pred_label = "unknown"

    if choice == "cnn":
        model = load_cnn()
        x = transform_tensor_norm(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(x).argmax(1).item()
        pred_label = "clean" if pred == 0 else "defected"

    elif choice == "autoencoder":
        model = load_autoencoder()
        x = transform_tensor(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            recon = model(x)
            mse = F.mse_loss(recon, x).item()
        pred_label = "defected" if mse > 0.01 else "clean"
        print(f"MSE: {mse:.6f}")

    elif choice in ["rf_me", "rf_paper"]:
        rf_model, version = load_rf(choice)
        cnn = load_feature_cnn()
        x = transform_tensor_norm(img_pil).to(DEVICE)

        if version == "me":
            from Squeeze_func_me import reduce_bit_depth, median_filter, jpeg_compression, resize_squeeze, hist_equalize, rgb_channel_squeeze
            squeezers = [lambda x: x, reduce_bit_depth, median_filter, jpeg_compression, resize_squeeze, hist_equalize]
            names = ["id", "bit", "median", "jpeg", "resize", "histeq"]
            feats = extract_me(cnn, x, squeezers=squeezers, squeezer_names=names)
        else:
            from Squeeze_func_paper import reduce_bit_depth, median_filter, rgb_channel_squeeze
            squeezers = [lambda x: x, reduce_bit_depth, median_filter, rgb_channel_squeeze]
            names = ["id", "bit", "median", "rgb"]
            feats = extract_paper(cnn, x, squeezers=squeezers, squeezer_names=names)

        X = np.array([list(feats.values())])
        pred = rf_model.predict(X)[0]
        pred_label = "clean" if pred == 0 else "defected"

    print(f"{choice.upper()} Prediction: {pred_label.upper()}")
    show_prediction(img_pil, pred_label, true_label, filename, choice)

if __name__ == "__main__":
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Curent device: {DEVICE.upper()}")
    main()
