import json
import pandas as pd
import matplotlib.pyplot as plt

CONFIDENCE_THRESHOLD = 0.4
RESULTS_FILE = "evaluation_results.json"

def load_results(path):
    with open(path, "r") as f:
        return json.load(f)

def analyze_results(data):
    summary = []

    for model_name, metrics in data.items():
        predictions = metrics.get("predictions", [])
        mAP50 = metrics.get("mAP@50", 0)
        avg_time = metrics.get("avg_inference_time", 0)

        # Filter predictions with confidence >= threshold
        confident_preds = [pred["score"] for pred in predictions if pred["score"] >= CONFIDENCE_THRESHOLD]
        avg_conf = sum(confident_preds) / len(confident_preds) if confident_preds else 0

        summary.append({
            "Model": model_name,
            "Avg Confidence (≥ 0.4)": round(avg_conf, 4),
            "Avg Inference Time (s)": round(avg_time, 4),
            "mAP@50": round(mAP50, 4)
        })

    return pd.DataFrame(summary)

def plot_metrics(df):
    plt.figure(figsize=(12, 10))

    # Subplot 1: Average Confidence
    plt.subplot(3, 1, 1)
    plt.bar(df["Model"], df["Avg Confidence (≥ 0.4)"], color="skyblue")
    plt.title("Average Confidence per Model (≥ 0.4)")
    plt.ylabel("Confidence")
    plt.ylim(0, 1)
    plt.grid(True, axis='y')

    # Subplot 2: Inference Time
    plt.subplot(3, 1, 2)
    plt.bar(df["Model"], df["Avg Inference Time (s)"], color="orange")
    plt.title("Average Inference Time per Model")
    plt.ylabel("Time (s)")
    plt.grid(True, axis='y')

    # Subplot 3: mAP@50
    plt.subplot(3, 1, 3)
    plt.bar(df["Model"], df["mAP@50"], color="green")
    plt.title("mAP@50 per Model")
    plt.ylabel("mAP@50")
    plt.ylim(0, .1)
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.show()

def main():
    data = load_results(RESULTS_FILE)
    df = analyze_results(data)

    print("\n==== Evaluation Summary ====")
    print(df.to_string(index=False))

    plot_metrics(df)

if __name__ == "__main__":
    main()
