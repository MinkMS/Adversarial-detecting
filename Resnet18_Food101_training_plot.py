import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = 'train_log_resnet18.csv'

df = pd.read_csv(CSV_FILE)

plt.figure(figsize=(12, 5))

# ----- Accuracy -----
plt.subplot(1, 2, 1)
plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy')
plt.plot(df['epoch'], df['val_acc'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid(True)

# ----- Loss -----
plt.subplot(1, 2, 2)
plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_plot.png')
plt.show()
#Scipt vẽ biểu đồ loss và accuracy trong quá trình huấn luyện mô hình ResNet18 trên tập dữ liệu Food101.