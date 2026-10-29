
from pathlib import Path
import re

import joblib


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "linear_svm.joblib"


# ==========================================================
# 2. LOAD THE TRAINED MODEL
# ==========================================================

def load_model():
    """Load the trained Linear SVM pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained SVM model not found. "
            "Run: python src/train_models.py"
        )

    return joblib.load(MODEL_PATH)


# ==========================================================
# 3. DETECT DEFANGED URL INDICATORS
# ==========================================================

def contains_defanged_url(message):
    """
    Identify common defanged URL patterns.

    A defanged URL is an intentionally modified URL, often
    used in security reports to avoid accidental navigation.

    A defanged URL alone does not prove a message is phishing.
    """

    text = str(message).lower()

    patterns = [
        r"hxxps?://",
        r"\[\.\]",
        r"\(\.\)",
        r"\bdot\b",
    ]

    return any(
        re.search(pattern, text)
        for pattern in patterns
    )


# ==========================================================
# 4. PREDICT MESSAGE RISK
# ==========================================================

def predict_message(message, review_threshold=0.25):
    """
    Classify a message using the trained Linear SVM.

    Supported predictions:
        - phishing
        - legitimate
        - uncertain

    Decision scores are not probabilities.

    Messages containing defanged URL indicators are flagged
    for manual review instead of being automatically declared
    phishing or legitimate.
    """

    message = str(message).strip()

    if not message:
        raise ValueError("Please enter a message.")

    if review_threshold < 0:
        raise ValueError(
            "review_threshold must be zero or greater."
        )

    model = load_model()

    classes = list(model.classes_)

    if (
        "phishing" not in classes
        or "legitimate" not in classes
    ):
        raise ValueError(
            "The trained model must contain both "
            "'phishing' and 'legitimate' classes."
        )

    # ------------------------------------------------------
    # Check for defanged URL indicators
    # ------------------------------------------------------

    has_defanged_url = contains_defanged_url(message)

    # ------------------------------------------------------
    # Calculate the SVM decision score
    # ------------------------------------------------------

    score = float(
        model.decision_function([message])[0]
    )

    # Convert the score so positive values point toward
    # the phishing class, regardless of class ordering.
    if classes[1] == "phishing":
        phishing_score = score
    else:
        phishing_score = -score

    # ------------------------------------------------------
    # Apply the review policy
    # ------------------------------------------------------

    if has_defanged_url:
        prediction = "uncertain"
        risk_level = "Needs manual review"

        recommendation = (
            "This message contains a defanged URL indicator. "
            "Its presence does not prove phishing, but the "
            "message should be reviewed carefully. Do not "
            "visit the URL. Verify the sender and any request "
            "through an independently obtained official channel."
        )

    elif abs(phishing_score) < review_threshold:
        prediction = "uncertain"
        risk_level = "Needs manual review"

        recommendation = (
            "The model is not sufficiently decisive. "
            "Do not rely on this prediction alone. "
            "Verify the sender and message through an official channel."
        )

    elif phishing_score > 0:
        prediction = "phishing"
        risk_level = "High"

        recommendation = (
            "The model flagged this message as potentially suspicious. "
            "Do not click links or share sensitive information. "
            "Verify the message through an independently obtained "
            "official channel."
        )

    else:
        prediction = "legitimate"
        risk_level = "Not flagged"

        recommendation = (
            "The model did not flag this message as phishing. "
            "This does not guarantee that it is safe. "
            "Check the sender and verify unexpected requests."
        )

    return {
        "prediction": prediction,
        "risk_level": risk_level,
        "decision_score": round(phishing_score, 4),
        "recommendation": recommendation,
    }


# ==========================================================
# 5. INTERACTIVE TERMINAL SCANNER
# ==========================================================

def main():
    print("=" * 55)
    print("PHISHSHIFT - MESSAGE RISK SCANNER")
    print("=" * 55)

    print("\nEnter a message to analyze.")
    print("Type 'exit' to stop.")

    while True:
        try:
            message = input("\nMessage: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nScanner closed.")
            break

        if message.lower() == "exit":
            print("Scanner closed.")
            break

        if not message:
            print("Please enter a message.")
            continue

        try:
            result = predict_message(message)

            print("\n--- ANALYSIS RESULT ---")
            print("Prediction:", result["prediction"])
            print("Risk level:", result["risk_level"])
            print("Decision score:", result["decision_score"])
            print("Recommendation:", result["recommendation"])

        except (ValueError, FileNotFoundError) as error:
            print("Error:", error)

        except Exception as error:
            print("Unexpected error:", error)


if __name__ == "__main__":
    main()