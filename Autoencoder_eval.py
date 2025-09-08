import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
from PIL import Image

# ===== CONFIG =====
MODEL_PATH = "autoencoder_model.pth"
DATASET_PATH = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\adversarial-eval"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMG_SIZE = 512
OUTPUT_TXT = "autoencoder_eval_result.txt"

# ===== AUTOENCODER CLASS (Same as training) =====
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ===== TRANSFORMS =====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ===== LOAD MODEL =====
def load_model():
    model = Autoencoder().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# ===== LOAD DATA =====
def load_dataset():
    data = ImageFolder(root=DATASET_PATH, transform=transform)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)
    return loader, data.classes

# ===== EVALUATION =====
def evaluate_autoencoder():
    model = load_model()
    loader, class_names = load_dataset()
    
    all_labels, all_losses = [], []

    print(f"Using device: {DEVICE}")
    print("Calculating reconstruction losses...")

    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            loss = F.mse_loss(outputs, imgs, reduction='none')
            loss = loss.view(loss.size(0), -1).mean(dim=1).cpu().numpy()
            all_losses.extend(loss)
            all_labels.extend(labels.numpy())

    all_losses = np.array(all_losses)
    all_labels = np.array(all_labels)

    # ===== FIND BEST THRESHOLD =====
    best_acc = 0
    best_thresh = 0.0
    for thresh in np.linspace(all_losses.min(), all_losses.max(), 200):
        preds = (all_losses > thresh).astype(int)
        acc = accuracy_score(all_labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    final_preds = (all_losses > best_thresh).astype(int)

    report = classification_report(all_labels, final_preds, target_names=class_names, digits=4)
    acc = accuracy_score(all_labels, final_preds)
    f1 = f1_score(all_labels, final_preds)

    print(f"\n===== Autoencoder Evaluation =====")
    print(f"Best threshold: {best_thresh:.6f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(report)

    with open(OUTPUT_TXT, "w") as f:
        f.write("===== Autoencoder Evaluation =====\n")
        f.write(f"Best threshold: {best_thresh:.6f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write(report)

    print(f"Results saved to {OUTPUT_TXT}")

# ===== RUN =====
if __name__ == "__main__":
    torch.manual_seed(42)
    evaluate_autoencoder()
