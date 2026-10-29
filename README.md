# 🛡️ PhishShift – AI-Powered Phishing Message Detection

### Detect Suspicious Messages. Stay Aware. Stay Safe.

PhishShift is a machine learning-based message risk scanner designed to identify potentially malicious phishing messages and distinguish them from legitimate communications. It uses Natural Language Processing (NLP), TF-IDF text features, and machine learning classification techniques to analyze message content and provide security recommendations.

🌐 **Live Demo:** https://phishshift-s71z.onrender.com

---

## 📌 Project Overview

Phishing messages often attempt to trick users into sharing sensitive information, clicking malicious links, or responding to fraudulent requests.

PhishShift demonstrates how machine learning can help analyze message content and flag potentially suspicious communications. Users can enter a message through an interactive Streamlit dashboard and receive a classification, risk assessment, and recommendation.

This project is an educational prototype evaluated using synthetic data. It is not a production-grade phishing detection service.

## 🎯 Objectives

* Classify messages as phishing or legitimate.
* Apply NLP techniques to analyze message text.
* Compare rule-based and machine learning approaches.
* Identify suspicious language patterns.
* Flag defanged URL indicators for manual review.
* Evaluate model performance using held-out test data.
* Test robustness against changes in capitalization and punctuation.
* Provide an interactive web interface for message analysis.

## ✨ Key Features

* **Message Classification:** Predicts whether a message is phishing or legitimate.
* **NLP-Based Analysis:** Uses TF-IDF text features for machine learning.
* **Multiple Model Comparison:** Evaluates rule-based detection, Logistic Regression, and Linear SVM.
* **Risk Assessment:** Provides a prediction, risk label, decision score, and recommendation.
* **Uncertainty Handling:** Flags messages requiring additional review.
* **Defanged URL Detection:** Flags selected intentionally modified URL patterns for manual review.
* **Interactive Dashboard:** Provides a browser-based interface built with Streamlit.
* **Model Evaluation:** Reports accuracy, precision, recall, F1 score, and false positive rate.
* **Automated Testing:** Uses pytest to validate core functionality.

## 🧰 Technology Stack

| Technology          | Purpose                                 |
| ------------------- | --------------------------------------- |
| Python              | Application development                 |
| Pandas              | Dataset handling                        |
| NumPy               | Numerical operations                    |
| Scikit-learn        | Machine learning and evaluation         |
| TF-IDF              | Text feature extraction                 |
| Logistic Regression | Classification baseline                 |
| Linear SVM          | Message classification                  |
| Streamlit           | Interactive web dashboard               |
| Joblib              | Saving and loading trained models       |
| Pytest              | Automated acceptance testing            |
| Git and GitHub      | Version control and source code hosting |
| Render              | Application deployment                  |

## 🏗️ Project Structure

```text
PhishShift/
├── data/
│   ├── messages.csv
│   └── messages_cleaned.csv
├── models/
│   ├── logistic_regression.joblib
│   └── linear_svm.joblib
├── reports/
├── src/
│   ├── app.py
│   ├── predict_message.py
│   ├── generate_dataset.py
│   ├── data_preprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── robustness_test.py
├── tests/
│   └── test_acceptance.py
├── requirements.txt
└── README.md
```

The files listed above describe the expected project layout; individual generated reports or data files may vary.

## ⚙️ How It Works

1. **Dataset Preparation:** Generate and prepare synthetic phishing and legitimate messages.
2. **Data Preprocessing:** Validate message records, normalize text, remove exact duplicates, and flag near-duplicate pairs.
3. **Feature Extraction:** Convert text into numerical TF-IDF features.
4. **Model Training:** Train and compare Logistic Regression and Linear SVM classifiers alongside a rule-based baseline.
5. **Model Evaluation:** Evaluate predictions on a campaign-separated test set.
6. **Risk Assessment:** Analyze a submitted message and produce a classification and decision score.
7. **Manual Review:** Flag selected defanged URL patterns and uncertain predictions for further inspection.
8. **User Interface:** Display the results and safety recommendations through Streamlit.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/PhishShift.git
cd PhishShift
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
```

### 3. Activate the environment

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Start the dashboard

If the trained model is already included in the repository:

```powershell
python -m streamlit run src/app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

If the trained model is missing, follow the project's model-training instructions before launching the application.

## 🧪 Testing

Run the acceptance tests with:

```powershell
python -m pytest tests/test_acceptance.py -v
```

The current acceptance test suite contains eight tests covering model availability, supported labels, empty input handling, prediction output fields, suspicious-message handling, and defanged URL handling.

The suite passed all eight tests in the development environment at the time of testing. Run the command again to verify the current version.

## 📊 Model Evaluation

PhishShift compares three approaches:

* Rule-based detection
* TF-IDF with Logistic Regression
* TF-IDF with Linear SVM

The evaluation process measures:

* Accuracy
* Precision
* Recall
* F1 score
* False positive rate
* True positives, true negatives, false positives, and false negatives

The project uses a campaign-separated test split to reduce overlap between message-template groups in training and testing.

**Important:** Previously observed perfect scores for the machine learning models were obtained on a small synthetic dataset. Such results do not establish real-world phishing detection accuracy. Broader evaluation on independent, representative datasets is required.

## 🔐 Safety and Limitations

* The dataset is synthetic and may not represent the diversity of real phishing messages.
* Model predictions are advisory and can be incorrect.
* A message classified as legitimate is not guaranteed to be safe.
* A defanged URL indicator alone does not prove that a message is malicious.
* Decision scores are not calibrated probabilities.
* Users should not visit suspicious links or share sensitive information based solely on the model's output.
* Unexpected requests should be verified through an independently obtained official contact method.
* This prototype is intended for educational and demonstration purposes.

## 🔮 Future Improvements

* Evaluate the system on larger, independently sourced datasets.
* Improve generalization to unfamiliar phishing campaigns.
* Add calibrated confidence estimates and more robust uncertainty handling.
* Expand explainability to identify influential text features.
* Improve accessibility and dashboard usability.
* Add structured evaluation reports and more comprehensive regression tests.
* Explore multilingual message classification.

## 👨‍💻 Author

**Kammalesh Raj V. S**

GitHub: https://github.com/kamalvs28-16

## 📄 License

No license has been specified yet. If you want others to reuse or modify this project, consider selecting an appropriate open-source license and adding its license file.

---

**PhishShift – Detect Suspicious Messages. Stay Aware. Stay Safe.**
