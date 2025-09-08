import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

# ===== CONFIG =====
DATA_DIR = r'C:\Users\Mink\OneDrive\Documents\GitHub\data\Food-101-split'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3
MODEL_PATH = 'food101_resnet18.pth'
CSV_LOG = 'train_log_resnet18.csv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===== TRANSFORMS =====
transform = transforms.Compose([
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ===== DATASET =====
train_set = datasets.ImageFolder(TRAIN_DIR, transform=transform)
test_set = datasets.ImageFolder(TEST_DIR, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(train_set.classes)

# ===== MODEL =====
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# ===== OPTIMIZER & SCHEDULER =====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LEARNING_RATE,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS
)

# ===== TRAIN LOOP =====

if('cuda' if torch.cuda.is_available() else 'cpu') == 'cuda':
    print("Using GPU for training") 

log = []

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # ===== VALIDATION =====
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    log.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    })

# ===== SAVE MODEL & CSV =====
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")

df = pd.DataFrame(log)
df.to_csv(CSV_LOG, index=False)
print(f"Training log saved to: {CSV_LOG}")
