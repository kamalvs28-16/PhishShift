import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "reports" / "model_results.json"


def main():
    print("=" * 60)
    print("PHISHSHIFT - MODEL EVALUATION")
    print("=" * 60)

    if not RESULTS_FILE.exists():
        print("Evaluation results not found.")
        print("Run the training script first:")
        print("python src/train_models.py")
        return

    with open(RESULTS_FILE, "r", encoding="utf-8") as file:
        results = json.load(file)

    rows = []

    for model_name, metrics in results.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": metrics.get("accuracy", 0),
                "Precision": metrics.get("precision", 0),
                "Recall": metrics.get("recall", 0),
                "F1 Score": metrics.get("f1_score", 0),
                "False Positive Rate": metrics.get(
                    "false_positive_rate", 0
                ),
            }
        )

    if not rows:
        print("No model results are available.")
        return

    results_df = pd.DataFrame(rows)

    print("\nMODEL COMPARISON")
    print("-" * 60)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    output_file = ROOT / "reports" / "model_comparison.csv"
    results_df.to_csv(output_file, index=False)

    print("\nComparison saved to:")
    print(output_file)

    print(
        "\nNote: These results reflect the available dataset and test split. "
        "They do not guarantee real-world phishing detection performance."
    )


if __name__ == "__main__":
    main()