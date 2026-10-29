from pathlib import Path
import json
import sys

import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report


# ==================================================
# 1. Configure project paths
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = ROOT / "src"
DATA_FILE = ROOT / "data" / "messages_cleaned.csv"
MODEL_PATH = ROOT / "models" / "linear_svm.joblib"
REPORTS_DIR = ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ==================================================
# 2. Load data and trained model
# ==================================================

def main():
    print("=" * 65)
    print("PHISHSHIFT - FALSE-POSITIVE ANALYSIS")
    print("=" * 65)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Run python src/train_models.py first."
        )

    data = pd.read_csv(DATA_FILE)

    required_columns = [
        "body_text",
        "label",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    data = data.dropna(
        subset=["body_text", "label"]
    ).copy()

    data["body_text"] = data["body_text"].astype(str)

    data["label"] = (
        data["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data = data[
        data["label"].isin(
            ["legitimate", "phishing"]
        )
    ].copy()

    if data.empty:
        raise ValueError("No valid labeled messages found.")

    model = joblib.load(MODEL_PATH)

    # ==================================================
    # 3. Predict messages
    # ==================================================

    data["predicted_label"] = model.predict(
        data["body_text"]
    )

    # A false positive is a legitimate message
    # incorrectly predicted as phishing.
    false_positives = data[
        (data["label"] == "legitimate")
        & (data["predicted_label"] == "phishing")
    ].copy()

    # A false negative is a phishing message
    # incorrectly predicted as legitimate.
    false_negatives = data[
        (data["label"] == "phishing")
        & (data["predicted_label"] == "legitimate")
    ].copy()

    # ==================================================
    # 4. Print the analysis
    # ==================================================

    print(f"\nMessages analyzed: {len(data)}")
    print(f"False positives: {len(false_positives)}")
    print(f"False negatives: {len(false_negatives)}")

    print("\nClassification report:")
    print(
        classification_report(
            data["label"],
            data["predicted_label"],
            labels=["legitimate", "phishing"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            data["label"],
            data["predicted_label"],
            labels=["legitimate", "phishing"],
        )
    )

    # ==================================================
    # 5. Save error examples
    # ==================================================

    false_positive_file = (
        REPORTS_DIR / "false_positives.csv"
    )

    false_negative_file = (
        REPORTS_DIR / "false_negatives.csv"
    )

    false_positives.to_csv(
        false_positive_file,
        index=False,
    )

    false_negatives.to_csv(
        false_negative_file,
        index=False,
    )

    summary = {
        "messages_analyzed": int(len(data)),
        "false_positives": int(len(false_positives)),
        "false_negatives": int(len(false_negatives)),
        "evaluation_note": (
            "This analysis uses the complete cleaned dataset, "
            "including training examples. These results are "
            "diagnostic only and are not an unbiased estimate "
            "of model performance."
        ),
    }

    summary_file = REPORTS_DIR / "error_analysis.json"

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\nSaved reports:")
    print(false_positive_file)
    print(false_negative_file)
    print(summary_file)

    print(
        "\nAnalysis complete. Review the false-positive "
        "examples before changing the training data."
    )


if __name__ == "__main__":
    main()