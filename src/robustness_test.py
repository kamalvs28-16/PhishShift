from pathlib import Path
import json

import joblib


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "linear_svm.joblib"
REPORTS_DIR = ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load the trained model
# --------------------------------------------------

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "SVM model not found. Run "
            "'python src/train_models.py' first."
        )

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# 3. Predict a message
# --------------------------------------------------

def predict(model, message):
    return str(model.predict([message])[0])


# --------------------------------------------------
# 4. Define controlled test cases
# --------------------------------------------------

def create_test_cases():
    return [
        {
            "test_id": "ROB-01",
            "category": "baseline",
            "message": (
                "Urgent: Your account is restricted. "
                "Verify your credentials immediately."
            ),
        },
        {
            "test_id": "ROB-02",
            "category": "capitalization",
            "message": (
                "URGENT: YOUR ACCOUNT IS RESTRICTED. "
                "VERIFY YOUR CREDENTIALS IMMEDIATELY."
            ),
        },
        {
            "test_id": "ROB-03",
            "category": "punctuation",
            "message": (
                "Urgent!!! Your account is restricted... "
                "Verify your credentials immediately!!!"
            ),
        },
        {
            "test_id": "ROB-04",
            "category": "wording",
            "message": (
                "Your account access has been limited. "
                "Please check your account status."
            ),
        },
        {
            "test_id": "ROB-05",
            "category": "legitimate",
            "message": (
                "Your weekly class timetable is available "
                "from your college office."
            ),
        },
        {
            "test_id": "ROB-06",
            "category": "legitimate",
            "message": (
                "The department meeting is scheduled for "
                "Friday at 10 AM."
            ),
        },
        {
            "test_id": "ROB-07",
            "category": "password_reset",
            "message": (
                "You requested a password reset. If this "
                "was not you, contact the service using "
                "its official website."
            ),
        },
        {
            "test_id": "ROB-08",
            "category": "defanged_url",
            "message": (
                "Account notice: review the details at "
                "hxxps://account-check[.]example"
            ),
        },
    ]


# --------------------------------------------------
# 5. Run robustness tests
# --------------------------------------------------

def main():
    print("=" * 65)
    print("PHISHSHIFT - ROBUSTNESS TESTING")
    print("=" * 65)

    model = load_model()
    test_cases = create_test_cases()

    results = []

    for case in test_cases:
        prediction = predict(model, case["message"])

        result = {
            "test_id": case["test_id"],
            "category": case["category"],
            "message": case["message"],
            "prediction": prediction,
        }

        results.append(result)

        print(f"\nTest ID: {case['test_id']}")
        print(f"Category: {case['category']}")
        print(f"Message: {case['message']}")
        print(f"Prediction: {prediction}")

    # Compare the baseline with capitalization and punctuation variants.
    baseline = next(
        item["prediction"]
        for item in results
        if item["test_id"] == "ROB-01"
    )

    variation_tests = [
        item
        for item in results
        if item["test_id"] in ["ROB-02", "ROB-03"]
    ]

    consistent_variations = sum(
        item["prediction"] == baseline
        for item in variation_tests
    )

    summary = {
        "total_tests": len(results),
        "baseline_prediction": baseline,
        "capitalization_punctuation_tests": len(
            variation_tests
        ),
        "consistent_variations": consistent_variations,
        "consistency_rate": (
            consistent_variations / len(variation_tests)
            if variation_tests
            else None
        ),
        "results": results,
        "note": (
            "These are controlled synthetic examples. "
            "Consistency does not establish real-world accuracy."
        ),
    }

    report_path = REPORTS_DIR / "robustness_results.json"

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 65)
    print("ROBUSTNESS TEST SUMMARY")
    print("=" * 65)

    print("Total tests:", summary["total_tests"])
    print(
        "Baseline prediction:",
        summary["baseline_prediction"],
    )
    print(
        "Consistent punctuation/capitalization variants:",
        f"{consistent_variations}/"
        f"{len(variation_tests)}",
    )
    print(
        "Consistency rate:",
        f"{summary['consistency_rate']:.2%}",
    )
    print("Report saved to:", report_path)


if __name__ == "__main__":
    main()