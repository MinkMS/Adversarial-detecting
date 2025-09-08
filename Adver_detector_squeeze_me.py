import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ===== CONFIG =====
CSV_FEATURE_FILE = "features_combined_extended.csv"
MODEL_PATH = "rf_detector_me.pkl"
CONFUSION_OUTPUT = "rf_confusion_eval_me.png"
REPORT_TXT = "rf_eval_report_me.txt"

def visualize_rf_model():
    print(f"Loading features from {CSV_FEATURE_FILE}")
    df = pd.read_csv(CSV_FEATURE_FILE)

    X = df.drop(columns=["image", "label", "source_model"], errors='ignore')
    y = df["label"]

    print("Cleaning NaN / Inf...")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"Loading model from {MODEL_PATH}")
    rf = joblib.load(MODEL_PATH)

    print("Running prediction...")
    preds = rf.predict(X)

    print("\n===== Classification Report =====")
    report = classification_report(y, preds, target_names=["Clean", "Defected"])
    print(report)

    with open(REPORT_TXT, "w") as f:
        f.write(report)
        print(f"Report saved to: {REPORT_TXT}")

    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=["Clean", "Defected"],
                yticklabels=["Clean", "Defected"])
    plt.title("Confusion Matrix (Your Squeeze)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(CONFUSION_OUTPUT)
    print(f"Confusion matrix saved to: {CONFUSION_OUTPUT}")

if __name__ == "__main__":
    visualize_rf_model()
