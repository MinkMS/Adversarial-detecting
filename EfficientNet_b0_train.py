import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from timm import create_model
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import f1_score

# ====== FIX SEED ======
def seed_all(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ====== CONFIG ======
DATA_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\Food-101-split"
IMG_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-3
NUM_CLASSES = 101
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== TRANSFORMS ======
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== MAIN BLOCK FOR WINDOWS ======
if __name__ == '__main__':
    seed_all()
    torch.backends.cudnn.benchmark = True

    # LOAD DATA
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # BUILD MODEL
    model = create_model('efficientnet_b0', pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0
    log_rows = []

    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct, total = 0, 0, 0
        preds_train, labels_train = [], []

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += inputs.size(0)
            preds_train.extend(preds.cpu().tolist())
            labels_train.extend(targets.cpu().tolist())

        train_acc = correct / total
        train_f1 = f1_score(labels_train, preds_train, average='macro')
        train_loss /= total

        # ====== VALIDATION ======
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        preds_val, labels_val, file_names = [], [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(targets).sum().item()
                val_total += inputs.size(0)
                preds_val.extend(preds.cpu().tolist())
                labels_val.extend(targets.cpu().tolist())
                file_names.extend([""] * inputs.size(0))  # Optional: link to filename

        val_acc = val_correct / val_total
        val_f1 = f1_score(labels_val, preds_val, average='macro')
        val_loss /= val_total

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f} | F1={val_f1:.4f}")

        log_rows.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "efficientnet_food101.pth")
            print("Saved best model!")

    # SAVE LOG
    df_log = pd.DataFrame(log_rows)
    df_log.to_csv("train_log.csv", index=False)

    # SAVE VAL PREDS
    df_preds = pd.DataFrame({
        "file": file_names,
        "true_label": labels_val,
        "pred_label": preds_val
    })
    df_preds.to_csv("val_preds.csv", index=False)

    print("Training complete. All outputs saved.")
