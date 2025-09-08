import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from skimage.metrics import structural_similarity as ssim_func
from sklearn.metrics import mean_squared_error
from PIL import Image

# ========== CONFIG ==========
DATA_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\Food-101-rf\ae_data"
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3
IMG_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "autoencoder_model.pth"
LOG_CSV = "autoencoder_training_log.csv"

# ========== TRANSFORM ==========
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ========== AUTOENCODER ==========
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

# ========== METRIC ==========
def calc_metrics(orig, recon):
    orig_np = orig.permute(1, 2, 0).cpu().numpy()
    recon_np = recon.permute(1, 2, 0).cpu().numpy()
    mse = mean_squared_error(orig_np.flatten(), recon_np.flatten())
    psnr = -10 * np.log10(mse + 1e-8)
    ssim_val = ssim_func(orig_np, recon_np, channel_axis=-1, data_range=1.0)
    return mse, psnr, ssim_val

# ========== MAIN ==========
def main():
    print(f"Using device: {DEVICE}")
    print(f"Data directory: {DATA_DIR}")

    train_dataset = ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
    val_dataset = ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = Autoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    logs = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)

        # ========== VALIDATION ==========
        model.eval()
        val_loss, psnr_total, ssim_total = 0, 0, 0

        with torch.no_grad():
            for imgs, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, imgs)

                val_loss += loss.item() * imgs.size(0)

                for i in range(imgs.size(0)):
                    mse, psnr, ssim_v = calc_metrics(imgs[i], outputs[i])
                    psnr_total += psnr
                    ssim_total += ssim_v

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_psnr = psnr_total / len(val_loader.dataset)
        avg_ssim = ssim_total / len(val_loader.dataset)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | PSNR={avg_psnr:.2f} | SSIM={avg_ssim:.4f}")

        logs.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "psnr": avg_psnr,
            "ssim": avg_ssim
        })

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    pd.DataFrame(logs).to_csv(LOG_CSV, index=False)
    print(f"Logs saved to {LOG_CSV}")

# ========== SAFE ENTRY POINT ==========
if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True

    main()
