import pandas as pd
import matplotlib.pyplot as plt

# ===== LOAD CSV =====
CSV_FILE = "cnn_metrics_log.csv"
df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip()
epochs = df["Epoch"]

# ===== DEFINE PLOTS =====
metrics = [
    ("Train Acc", "Val Acc", "Accuracy", "accuracy_plot.png"),
    ("Train F1", "Val F1", "F1 Score", "f1_plot.png"),
    ("Train Precision", "Val Precision", "Precision", "precision_plot.png"),
    ("Train Recall", "Val Recall", "Recall", "recall_plot.png")
]

# ===== DRAW EACH PLOT =====
for train_col, val_col, title, filename in metrics:
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, df[train_col], label=train_col, marker="o")
    plt.plot(epochs, df[val_col], label=val_col, marker="o")
    plt.title(f"{title} over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
    print(f"Saved: {filename}")
