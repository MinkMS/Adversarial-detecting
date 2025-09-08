import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ===== CONFIG =====
MODEL_PATH = "rf_detector_me.pkl"
CSV_PATH = "features_eval_me.csv"
OUTPUT_TXT = "eval_result_me.txt"

def main():
    print("Evaluating Random Forest (paper)...")

    # Load model
    model = joblib.load(MODEL_PATH)

    # Load CSV
    df = pd.read_csv(CSV_PATH)

    # Clean invalid values
    orig_len = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"Dropped {orig_len - len(df)} rows with NaN or Inf.")

    # Extract features and labels
    X = df[model.feature_names_in_]
    y = df['label']

    # Predict
    y_pred = model.predict(X)

    # Evaluate
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    report = classification_report(y, y_pred, digits=4)

    # Print and save
    print("\n===== Evaluation Result (Paper Model) =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(report)

    with open(OUTPUT_TXT, "w") as f:
        f.write("===== Evaluation Result (Paper Model) =====\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(report)

    print(f"\nResults saved to {OUTPUT_TXT}")

if __name__ == "__main__":
    main()

# Đánh giá random forest me