import json
import re
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "messages_cleaned.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Rule-based prediction
# --------------------------------------------------

def rule_based_predict(text):
    """Predict whether a message is phishing using simple rules."""

    text = str(text).lower()

    patterns = [
        r"\bverify\b",
        r"\bimmediately\b",
        r"\burgent\b",
        r"\bcredentials\b",
        r"\brestricted\b",
        r"\bdisabled\b",
        r"\bclaim\b",
        r"\bunpaid\b",
        r"confirm your bank",
        r"hxxps?://",
    ]

    score = sum(
        1 for pattern in patterns
        if re.search(pattern, text)
    )

    if score >= 2:
        return "phishing"

    return "legitimate"


# --------------------------------------------------
# 3. Calculate evaluation metrics
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics and confusion matrix."""

    labels = ["legitimate", "phishing"]

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    true_negative, false_positive, false_negative, true_positive = (
        matrix.ravel()
    )

    total = len(y_true)

    false_positive_rate = (
        false_positive / (false_positive + true_negative)
        if (false_positive + true_negative) > 0
        else 0.0
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label="phishing",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label="phishing",
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                pos_label="phishing",
                zero_division=0,
            )
        ),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "total_samples": int(total),
    }


# --------------------------------------------------
# 4. Main training function
# --------------------------------------------------

def main():
    print("=" * 60)
    print("PHISHSHIFT - MODEL TRAINING")
    print("=" * 60)

    # Check whether the cleaned dataset exists.
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_FILE}\n"
            "Run the data preprocessing script first."
        )

    # Load the cleaned dataset.
    data = pd.read_csv(DATA_FILE)

    required_columns = [
        "message_id",
        "body_text",
        "label",
        "campaign_group",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Remove rows with missing required values.
    data = data.dropna(
        subset=[
            "body_text",
            "label",
            "campaign_group",
        ]
    ).copy()

    # Standardize labels and message text.
    data["label"] = (
        data["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data["body_text"] = (
        data["body_text"]
        .astype(str)
        .str.strip()
    )

    data["campaign_group"] = (
        data["campaign_group"]
        .astype(str)
        .str.strip()
    )

    # Keep only supported labels.
    data = data[
        data["label"].isin(
            ["legitimate", "phishing"]
        )
    ].copy()

    # Remove empty messages and missing campaign groups.
    data = data[
        (data["body_text"] != "")
        & (data["campaign_group"] != "")
    ].copy()

    if data.empty:
        raise ValueError(
            "The cleaned dataset contains no usable messages."
        )

    if data["label"].nunique() != 2:
        raise ValueError(
            "The dataset must contain both "
            "'legitimate' and 'phishing' labels."
        )

    if data["campaign_group"].nunique() < 2:
        raise ValueError(
            "At least two campaign groups are required "
            "for campaign-separated testing."
        )

    print(f"Total usable messages: {len(data)}")
    print("\nLabel distribution:")
    print(data["label"].value_counts())

    # --------------------------------------------------
    # 5. Split data by campaign group
    # --------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=100,
        test_size=0.30,
        random_state=42,
    )

    train_indices = None
    test_indices = None

    for candidate_train, candidate_test in splitter.split(
        data,
        data["label"],
        groups=data["campaign_group"],
    ):
        candidate_train_labels = set(
            data.iloc[candidate_train]["label"]
        )

        candidate_test_labels = set(
            data.iloc[candidate_test]["label"]
        )

        if (
            len(candidate_train_labels) == 2
            and len(candidate_test_labels) == 2
        ):
            train_indices = candidate_train
            test_indices = candidate_test
            break

    if train_indices is None or test_indices is None:
        raise ValueError(
            "Could not create a campaign-separated train/test split "
            "containing both labels in each set. Check the number "
            "of campaign groups and their label distribution."
        )

    train_data = data.iloc[train_indices].copy()
    test_data = data.iloc[test_indices].copy()

    # Confirm that no campaign appears in both sets.
    train_campaigns = set(
        train_data["campaign_group"]
    )

    test_campaigns = set(
        test_data["campaign_group"]
    )

    overlapping_campaigns = (
        train_campaigns & test_campaigns
    )

    if overlapping_campaigns:
        raise ValueError(
            "Campaign leakage detected between training and testing."
        )

    X_train = train_data["body_text"]
    y_train = train_data["label"]

    X_test = test_data["body_text"]
    y_test = test_data["label"]

    print(f"\nTraining messages: {len(train_data)}")
    print(f"Testing messages: {len(test_data)}")
    print(f"Training campaigns: {len(train_campaigns)}")
    print(f"Testing campaigns: {len(test_campaigns)}")
    print("Campaign overlap: 0")

    # --------------------------------------------------
    # 6. Evaluate the rule-based baseline
    # --------------------------------------------------

    print("\nTraining and evaluating rule-based baseline...")

    rule_predictions = [
        rule_based_predict(message)
        for message in X_test
    ]

    results = {
        "rule_based": calculate_metrics(
            y_test,
            rule_predictions,
        )
    }

    # --------------------------------------------------
    # 7. Train TF-IDF + Logistic Regression
    # --------------------------------------------------

    print("Training TF-IDF + Logistic Regression...")

    logistic_model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    max_features=20000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    logistic_model.fit(X_train, y_train)

    logistic_predictions = logistic_model.predict(X_test)

    results["tfidf_logistic_regression"] = calculate_metrics(
        y_test,
        logistic_predictions,
    )

    joblib.dump(
        logistic_model,
        MODELS_DIR / "logistic_regression.joblib",
    )

    # --------------------------------------------------
    # 8. Train TF-IDF + Linear SVM
    # --------------------------------------------------

    print("Training TF-IDF + Linear SVM...")

    svm_model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(3, 5),
                    max_features=20000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    svm_model.fit(X_train, y_train)

    svm_predictions = svm_model.predict(X_test)

    results["tfidf_linear_svm"] = calculate_metrics(
        y_test,
        svm_predictions,
    )

    joblib.dump(
        svm_model,
        MODELS_DIR / "linear_svm.joblib",
    )

    # --------------------------------------------------
    # 9. Save model evaluation results
    # --------------------------------------------------

    results_file = REPORTS_DIR / "model_results.json"

    with open(
        results_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    split_report = {
        "total_messages": int(len(data)),
        "training_messages": int(len(train_data)),
        "testing_messages": int(len(test_data)),
        "training_campaigns": sorted(train_campaigns),
        "testing_campaigns": sorted(test_campaigns),
        "campaign_overlap": sorted(overlapping_campaigns),
        "training_label_distribution": {
            str(label): int(count)
            for label, count in (
                train_data["label"].value_counts().items()
            )
        },
        "testing_label_distribution": {
            str(label): int(count)
            for label, count in (
                test_data["label"].value_counts().items()
            )
        },
        "random_state": 42,
    }

    split_file = REPORTS_DIR / "data_split.json"

    with open(
        split_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            split_report,
            file,
            indent=4,
        )

    # --------------------------------------------------
    # 10. Print results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")

        print(
            f"Accuracy: "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1 Score: "
            f"{metrics['f1_score']:.4f}"
        )

        print(
            f"False Positive Rate: "
            f"{metrics['false_positive_rate']:.4f}"
        )

        print(
            f"True Negatives: "
            f"{metrics['true_negative']}"
        )

        print(
            f"False Positives: "
            f"{metrics['false_positive']}"
        )

        print(
            f"False Negatives: "
            f"{metrics['false_negative']}"
        )

        print(
            f"True Positives: "
            f"{metrics['true_positive']}"
        )

    print("\nTraining completed successfully.")

    print(f"Saved model files in: {MODELS_DIR}")
    print(f"Saved evaluation results in: {results_file}")
    print(f"Saved data split report in: {split_file}")


# --------------------------------------------------
# 11. Run the program
# --------------------------------------------------

if __name__ == "__main__":
    main()