import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# === Load CSV ===
log_path = r"C:\Users\Mink\OneDrive\Documents\GitHub\Adversarial-detecting\training_log.csv"    # Đường dẫn đến file CSV
df = pd.read_csv(log_path)

epochs = df['epoch']

# === Plot Loss ===
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.plot(epochs, df['train_loss'], label='Train Loss', color='blue')
plt.plot(epochs, df['test_loss'], label='Test Loss', color='orange')
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.legend()
plt.grid(True)

# === Plot Accuracy ===
plt.subplot(1, 2, 2)
plt.plot(epochs, df['train_acc'], label='Train Accuracy', color='green')
plt.plot(epochs, df['test_acc'], label='Test Accuracy', color='red')
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(100))
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_plot.png')  # Save hình
plt.show()