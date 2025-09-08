import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# ===== CONFIG =====
CSV_FILE = "features_combined.csv"
MODEL_PATH = "rf_detector_paper.pkl"
LOG_TXT = "rf_log_paper.txt"
LOG_CSV = "rf_log_paper.csv"
CONF_PNG = "rf_confusion_paper.png"

def train_rf():
    start_time = time.time()
    print(f"Loading: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)

    feature_cols = [c for c in df.columns if c not in ['image', 'label', 'source_model']]
    X = df[feature_cols]
    y = df['label']

    print("Checking NaN/Inf...")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    print("Training Random Forest (paper)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    preds = rf.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    precision = precision_score(y, preds)
    recall = recall_score(y, preds)

    elapsed = round(time.time() - start_time, 2)
    samples_total = len(y)
    samples_clean = int((y == 0).sum())
    samples_defected = int((y == 1).sum())
    feature_count = X.shape[1]

    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | Prec: {precision:.4f} | Recall: {recall:.4f}")
    joblib.dump(rf, MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")

    top_feats = sorted(zip(rf.feature_importances_, feature_cols), reverse=True)[:5]

    # Save log text
    with open(LOG_TXT, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nF1 Score: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\nRecall: {recall:.4f}\n")
        f.write(f"Samples: {samples_total} (Clean: {samples_clean}, Defected: {samples_defected})\n")
        f.write(f"Feature count: {feature_count}\nTrain time: {elapsed}s\n")
        f.write("Top 5 Features:\n")
        for score, name in top_feats:
            f.write(f"  - {name}: {score:.4f}\n")

    # Save log CSV
    pd.DataFrame([{
        "accuracy": acc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "samples_total": samples_total,
        "samples_clean": samples_clean,
        "samples_defected": samples_defected,
        "feature_count": feature_count,
        "train_time_sec": elapsed
    }]).to_csv(LOG_CSV, index=False)

    # Conf matrix
    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Clean', 'Defected'], yticklabels=['Clean', 'Defected'])
    plt.title("Confusion Matrix (Paper Squeeze)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(CONF_PNG)
    print(f"Confusion matrix saved: {CONF_PNG}")

if __name__ == "__main__":
    train_rf()

#Script để train mô hình Random Forest từ file feature đã extract