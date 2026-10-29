from pathlib import Path
import sys

import streamlit as st

# Allow imports from the src directory.
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict_message import predict_message


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PhishShift - Message Risk Scanner",
    page_icon="🛡️",
    layout="centered",
)


# --------------------------------------------------
# Dashboard header
# --------------------------------------------------

st.title("🛡️ PhishShift")
st.subheader("AI-Powered Message Risk Scanner")

st.write(
    "Analyze a message for potential phishing indicators "
    "using a trained machine-learning model."
)

st.warning(
    "This is a prototype. Predictions may be incorrect. "
    "Do not rely on this tool alone to determine whether "
    "a message is safe."
)


# --------------------------------------------------
# Message input
# --------------------------------------------------

st.markdown("### Enter a message")

message = st.text_area(
    "Message to analyze",
    height=180,
    placeholder=(
        "Paste a message here. Do not include real passwords, "
        "financial details, or other sensitive information."
    ),
)

analyze_button = st.button(
    "Analyze Message",
    type="primary",
    use_container_width=True,
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if analyze_button:
    if not message.strip():
        st.error("Please enter a message before analyzing it.")

    else:
        try:
            with st.spinner("Analyzing message..."):
                result = predict_message(message)

            prediction = result["prediction"]
            risk_level = result["risk_level"]
            score = result["decision_score"]
            recommendation = result["recommendation"]

            st.markdown("---")
            st.markdown("### Analysis Result")

            if prediction == "phishing":
                st.error("🚨 Potential phishing message detected")

            elif prediction == "legitimate":
                st.success("No phishing classification returned")

            else:
                st.warning("⚠️ Uncertain result — manual review recommended")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label="Prediction",
                    value=prediction.upper(),
                )

            with col2:
                st.metric(
                    label="Risk status",
                    value=risk_level,
                )

            st.caption(
                f"SVM decision score: {score:.4f} "
                "(not a probability)"
            )

            st.markdown("### Recommendation")
            st.write(recommendation)

            with st.expander("View analyzed message"):
                st.text(message)

        except FileNotFoundError as error:
            st.error(str(error))

        except Exception as error:
            st.error(
                f"Unable to analyze the message: {error}"
            )


# --------------------------------------------------
# About section
# --------------------------------------------------

st.markdown("---")
st.markdown("### About PhishShift")

st.write(
    """
    PhishShift compares machine-learning approaches for
    identifying potentially suspicious messages.

    **Current features**
    - TF-IDF with Linear SVM
    - Message classification
    - Uncertainty handling
    - Controlled robustness tests
    - Safety recommendations
    """
)

st.caption(
    "Research prototype using synthetic data. "
    "It is not a production-grade security service."
)