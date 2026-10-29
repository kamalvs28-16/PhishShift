from pathlib import Path
import sys

import joblib
import pytest


# --------------------------------------------------
# Configure project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
MODEL_PATH = ROOT / "models" / "linear_svm.joblib"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict_message import predict_message


# --------------------------------------------------
# Test 1: Model file exists
# --------------------------------------------------

def test_trained_model_exists():
    assert MODEL_PATH.exists(), (
        "SVM model not found. Run python src/train_models.py"
    )


# --------------------------------------------------
# Test 2: Model contains both classes
# --------------------------------------------------

def test_model_has_both_classes():
    assert MODEL_PATH.exists(), "SVM model not found."

    model = joblib.load(MODEL_PATH)

    assert set(model.classes_) == {"legitimate", "phishing"}


# --------------------------------------------------
# Test 3: Empty input is rejected
# --------------------------------------------------

def test_empty_message_is_rejected():
    with pytest.raises(ValueError):
        predict_message("")


# --------------------------------------------------
# Test 4: Whitespace input is rejected
# --------------------------------------------------

def test_whitespace_message_is_rejected():
    with pytest.raises(ValueError):
        predict_message("   ")


# --------------------------------------------------
# Test 5: Prediction contains required fields
# --------------------------------------------------

def test_prediction_returns_required_fields():
    result = predict_message(
        "The department meeting is scheduled for Friday."
    )

    required_fields = {
        "prediction",
        "risk_level",
        "decision_score",
        "recommendation",
    }

    assert required_fields.issubset(result.keys())


# --------------------------------------------------
# Test 6: Prediction label is valid
# --------------------------------------------------

def test_prediction_label_is_supported():
    result = predict_message(
        "Your weekly class timetable is available."
    )

    assert result["prediction"] in {
        "phishing",
        "legitimate",
        "uncertain",
    }


# --------------------------------------------------
# Test 7: Suspicious message is processed
# --------------------------------------------------

def test_suspicious_message_returns_valid_result():
    result = predict_message(
        "Urgent: Your account is restricted. "
        "Verify your credentials immediately."
    )

    assert result["prediction"] in {
        "phishing",
        "legitimate",
        "uncertain",
    }

    assert result["recommendation"]


# --------------------------------------------------
# Test 8: Defanged URL message is processed
# --------------------------------------------------

def test_defanged_url_message_returns_valid_result():
    result = predict_message(
        "Account notice: review details at "
        "hxxps://account-check[.]example"
    )

    assert result["prediction"] in {
        "phishing",
        "legitimate",
        "uncertain",
    }

    assert result["recommendation"]


# --------------------------------------------------
# End of acceptance tests
# --------------------------------------------------