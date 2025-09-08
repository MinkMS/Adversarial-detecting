import os
import time
import torch
import shutil
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import csv

# ====== CONFIG ======
DATA_DIR = r"C:\Users\Mink\OneDrive\Documents\GitHub\data\adversarial-eval-split"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-4
IMG_SIZE = 512

MODEL_PATH = "cnn_classifier.pth"
LOG_PATH = "cnn_training_log.txt"
CSV_PATH = "cnn_metrics_log.csv"

# ====== TRANSFORM ======
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== DATASET & DATALOADER ======
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ====== MODEL ======
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

# ====== LOSS & OPTIMIZER ======
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ====== LOGGING ======
def log(msg):
    print(msg)
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")

def init_csv():
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Epoch", "Time(s)", "Train Acc", "Val Acc",
            "Train F1", "Val F1",
            "Train Precision", "Val Precision",
            "Train Recall", "Val Recall"
        ])

def append_csv(epoch, t, train_acc, val_acc, train_f1, val_f1, train_prec, val_prec, train_rec, val_rec):
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, round(t, 2), round(train_acc, 4), round(val_acc, 4),
            round(train_f1, 4), round(val_f1, 4),
            round(train_prec, 4), round(val_prec, 4),
            round(train_rec, 4), round(val_rec, 4)
        ])

# ====== TRAINING ======
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    report = classification_report(y_true, y_pred, target_names=["clean", "defected"], output_dict=True)
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    return acc, report

def train_one_epoch():
    model.train()
    running_loss = 0
    for x, y in tqdm(train_loader, desc="Training", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

# ====== RUN TRAINING LOOP ======
log(f"Training on device: {DEVICE}")
init_csv()

for epoch in range(1, EPOCHS + 1):
    start = time.time()
    train_loss = train_one_epoch()
    train_acc, train_report = evaluate(model, train_loader)
    val_acc, val_report = evaluate(model, val_loader)
    end = time.time()

    train_f1 = train_report['weighted avg']['f1-score']
    val_f1 = val_report['weighted avg']['f1-score']
    train_prec = train_report['weighted avg']['precision']
    val_prec = val_report['weighted avg']['precision']
    train_rec = train_report['weighted avg']['recall']
    val_rec = val_report['weighted avg']['recall']

    log_msg = (
        f"\nEpoch {epoch:02d}/{EPOCHS} - Time: {end - start:.2f}s\n"
        f"Train Loss: {train_loss:.4f}\n"
        f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}\n"
        f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}\n"
        f"Train Precision: {train_prec:.4f} | Val Precision: {val_prec:.4f}\n"
        f"Train Recall: {train_rec:.4f} | Val Recall: {val_rec:.4f}"
    )

    log(log_msg)
    append_csv(epoch, end - start, train_acc, val_acc, train_f1, val_f1, train_prec, val_prec, train_rec, val_rec)

# ====== SAVE MODEL ======
torch.save(model.state_dict(), MODEL_PATH)
log(f"\n✅ Model saved to: {MODEL_PATH}")
