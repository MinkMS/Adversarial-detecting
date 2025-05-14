import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm

# =========================
# 🔧 HYPERPARAMETERS
# =========================
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
SAVE_PATH = 'resnet18_128x128.pth'
LOG_CSV = 'training_log.csv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 2  # Nếu vẫn lỗi, thử đổi thành 0

# =========================
# MAIN FUNCTION
# =========================
def main():
    # === DATA TRANSFORMS ===
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # === LOAD DATASETS ===
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=NUM_WORKERS)

    # === INIT MODEL ===
    model = resnet18(num_classes=10).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # === CSV LOG INIT ===
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc'])

    # === TRAINING LOOP ===
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = total = 0

        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total

        # === EVALUATE ON TEST SET ===
        model.eval()
        test_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss /= len(testloader)
        test_acc = 100. * correct / total

        print(f"[Epoch {epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # === LOG CSV ===
        with open(LOG_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, test_loss, train_acc, test_acc])

    # === SAVE MODEL ===
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")


if __name__ == '__main__':
    main()
