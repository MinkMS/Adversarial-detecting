import re
import csv
from prettytable import PrettyTable
from datetime import datetime

try:
    from matplotlib import pyplot as plt
    import pandas as pd
    SAVE_PNG = True
except ImportError:
    SAVE_PNG = False

# === CONFIG ===
FILES = {
    "ME": "eval_result_me.txt",
    "PAPER": "eval_result_paper.txt",
    "AUTOENCODER": "autoencoder_eval_result.txt",
    "CNN": "cnn_metrics_log.csv"
}
CUSTOM_LOG = "custom_log.txt"  # log có dạng Epoch XX/XX
OUTPUT_TXT = "Model_comp.txt"
OUTPUT_PNG = "Model_comp.png"

def extract_summary_metrics_txt(path):
    with open(path, 'r') as f:
        content = f.read()

    return {
        'Accuracy': get_float(r"Accuracy:\s*([\d.]+)", content),
        'F1 Score': get_float(r"F1 Score:\s*([\d.]+)", content),
        'Precision': get_float(r"Precision:\s*([\d.]+)", content),
        'Recall': get_float(r"Recall:\s*([\d.]+)", content),
    }

def extract_summary_metrics_csv(path):
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        last_row = list(reader)[-1]
        return {
            'Accuracy': float(last_row.get('Accuracy', 0)),
            'F1 Score': float(last_row.get('F1 Score', 0)),
            'Precision': float(last_row.get('Precision', 0)),
            'Recall': float(last_row.get('Recall', 0)),
        }

def extract_custom_from_log(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    # Tìm dòng cuối cùng có Epoch
    for line in reversed(lines):
        if "Epoch" in line and "Val Acc" in line:
            acc = get_float(r"Val Acc:\s*([\d.]+)", line)
            f1 = get_float(r"Val F1:\s*([\d.]+)", line)
            precision = get_float(r"Val Precision:\s*([\d.]+)", line)
            recall = get_float(r"Val Recall:\s*([\d.]+)", line)
            return {
                'Accuracy': acc,
                'F1 Score': f1,
                'Precision': precision,
                'Recall': recall
            }
    return {}

def get_float(pattern, text):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else 0.0

def collect_all_metrics():
    all_metrics = {}
    for name, file in FILES.items():
        if file.endswith(".csv"):
            all_metrics[name] = extract_summary_metrics_csv(file)
        else:
            all_metrics[name] = extract_summary_metrics_txt(file)

    # Thêm mô hình CUSTOM
    all_metrics["CUSTOM"] = extract_custom_from_log(CUSTOM_LOG)
    return all_metrics

def save_table_to_txt(table):
    with open(OUTPUT_TXT, 'w') as f:
        f.write(f"Model Comparison - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(str(table))
    print(f"=> Đã lưu kết quả vào '{OUTPUT_TXT}'")

def save_table_to_png(all_metrics):
    if not SAVE_PNG:
        print("=> Thiếu matplotlib/pandas để lưu ảnh PNG.")
        return

    df = pd.DataFrame(all_metrics).T
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values.round(4), colLabels=df.columns, rowLabels=df.index, loc='center')
    plt.title("Model Evaluation Comparison", fontsize=14)
    plt.savefig(OUTPUT_PNG, bbox_inches='tight')
    print(f"=> Đã lưu ảnh bảng vào '{OUTPUT_PNG}'")

def display_comparison(all_metrics):
    table = PrettyTable()
    table.field_names = ["Metric"] + list(all_metrics.keys())

    for key in ['Accuracy', 'F1 Score', 'Precision', 'Recall']:
        row = [key]
        for model in all_metrics:
            val = all_metrics[model].get(key)
            row.append(f"{val:.4f}")
        table.add_row(row)

    print(table)
    save_table_to_txt(table)
    save_table_to_png(all_metrics)

if __name__ == "__main__":
    print("=== So sánh các mô hình (4 chỉ số tổng quát) ===")
    metrics = collect_all_metrics()
    display_comparison(metrics)
