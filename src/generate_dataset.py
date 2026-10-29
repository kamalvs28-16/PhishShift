"""
PhishShift - Synthetic Dataset Generator

Generates synthetic legitimate and phishing messages.
The dataset is intended for educational and experimental use.

Output:
    data/messages.csv
"""

import random
from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "messages.csv"

RANDOM_SEED = 42
VARIANTS_PER_TEMPLATE = 32

DATA_DIR.mkdir(parents=True, exist_ok=True)

random.seed(RANDOM_SEED)


# ============================================================
# 2. LEGITIMATE MESSAGE TEMPLATES
# ============================================================

LEGITIMATE_TEMPLATES = {
    "account": [
        "Your account statement is available through the official website.",
        "Your monthly account summary is ready for review.",
    ],

    "billing": [
        "Your monthly bill is available in your account.",
        "Your recent payment has been recorded. The receipt is available online.",
    ],

    "security": [
        "A new sign-in was recorded. Review your security settings through the official website if needed.",
        "Your security settings were updated. Contact official support if you did not make this change.",
    ],

    "support": [
        "Your support request has been received. Check its status through the official support portal.",
        "Our support team has updated your service request.",
    ],

    "delivery": [
        "Your delivery status has been updated on the official delivery service website.",
        "Your package is scheduled for delivery. Check the official tracking service for details.",
    ],

    "password": [
        "You requested a password reset. If this was you, continue through the official website.",
        "Your password reset request has been received. Visit the official website directly if you need to review it.",
    ],

    "meeting": [
        "The team meeting is scheduled for Friday at 10 AM.",
        "The meeting agenda is available in the shared workspace.",
    ],

    "subscription": [
        "Your subscription renewal date is approaching. Review your plan through the official account portal.",
        "Your subscription payment receipt is available in your account.",
    ],

    "college_timetable": [
        "Your weekly class timetable is available from your college office.",
        "The updated class timetable is available on the official college notice board.",
    ],

    "department_meeting": [
        "The department meeting is scheduled for Friday at 10 AM.",
        "The department meeting agenda is available from your department office.",
    ],

    "assignment": [
        "Your assignment submission deadline is Monday. Check your course portal for the requirements.",
        "The assignment instructions are available through your official course portal.",
    ],

    "library": [
        "The library will remain closed on Sunday. Normal opening hours resume on Monday.",
        "Your requested library book is available for collection during normal library hours.",
    ],

    "examination": [
        "The examination timetable has been published. Check the official college notice board.",
        "Your examination hall allocation is available through the official college portal.",
    ],

    "classroom": [
        "Tomorrow's lecture will begin at 9 AM in the usual classroom.",
        "Today's class has been moved to Room 204. Confirm the update with your department.",
    ],

    "attendance": [
        "Your attendance summary is available through the college portal.",
        "The monthly attendance report is available from your department office.",
    ],

    "campus_event": [
        "The college cultural event will take place next Friday. Refer to the official college notice.",
        "Registration details for the campus event are available through the official college website.",
    ],
}


# ============================================================
# 3. PHISHING MESSAGE TEMPLATES
# ============================================================

PHISHING_TEMPLATES = {
    "account_alert": [
        "Your account is restricted. Verify your credentials immediately.",
        "Your account will be disabled unless you verify your login details immediately.",
    ],

    "payment_alert": [
        "Your payment failed. Confirm your banking credentials immediately to avoid suspension.",
        "Submit your account details immediately to prevent payment service interruption.",
    ],

    "password_alert": [
        "Your password has expired. Verify your credentials immediately to restore access.",
        "Confirm your password immediately to avoid account suspension.",
    ],

    "delivery_alert": [
        "Your package is on hold. Submit your personal details to release the delivery.",
        "Delivery failed. Confirm your account information to reschedule your package.",
    ],

    "reward_alert": [
        "You have won a special reward. Submit your personal information to claim it.",
        "Congratulations! Verify your account details to receive your prize.",
    ],

    "security_alert": [
        "Suspicious activity detected. Verify your credentials to prevent account closure.",
        "Your account security is at risk. Confirm your login details immediately.",
    ],

    "document_alert": [
        "A confidential document is waiting. Enter your account password to view it.",
        "Verify your credentials to access this urgent document.",
    ],

    "refund_alert": [
        "Your refund is pending. Submit your banking credentials to receive the payment.",
        "Confirm your payment information immediately to process your refund.",
    ],
}


# ============================================================
# 4. MESSAGE VARIATIONS
# ============================================================

LEGITIMATE_PREFIXES = [
    "",
    "Notice: ",
    "Update: ",
    "Information: ",
    "Reminder: ",
    "Hello, ",
    "Please note: ",
    "For your information: ",
]

LEGITIMATE_SUFFIXES = [
    "",
    " Thank you.",
    " Please review the details when convenient.",
    " Contact the official office if you need assistance.",
    " Please use the official website or portal.",
]

PHISHING_PREFIXES = [
    "",
    "Urgent: ",
    "Important notice: ",
    "Immediate action required: ",
    "Security alert: ",
    "Warning: ",
    "Account notice: ",
]

PHISHING_SUFFIXES = [
    "",
    " Act now to avoid suspension.",
    " Verify your details immediately.",
    " Failure to respond may restrict access.",
    " Do not delay this action.",
]


# ============================================================
# 5. GENERATE THE DATASET
# ============================================================

def generate_messages():
    """Generate synthetic legitimate and phishing messages."""

    rows = []
    message_number = 1

    # Generate legitimate messages.
    for campaign_name, templates in LEGITIMATE_TEMPLATES.items():

        for _ in range(VARIANTS_PER_TEMPLATE):

            template = random.choice(templates)
            prefix = random.choice(LEGITIMATE_PREFIXES)
            suffix = random.choice(LEGITIMATE_SUFFIXES)

            message = f"{prefix}{template}{suffix}".strip()

            rows.append({
                "message_id": f"MSG-{message_number:05d}",
                "text": message,
                "label": "legitimate",
                "campaign": f"legitimate_{campaign_name}",
            })

            message_number += 1

    # Generate phishing messages.
    for campaign_name, templates in PHISHING_TEMPLATES.items():

        for _ in range(VARIANTS_PER_TEMPLATE):

            template = random.choice(templates)
            prefix = random.choice(PHISHING_PREFIXES)
            suffix = random.choice(PHISHING_SUFFIXES)

            message = f"{prefix}{template}{suffix}".strip()

            rows.append({
                "message_id": f"MSG-{message_number:05d}",
                "text": message,
                "label": "phishing",
                "campaign": f"phishing_{campaign_name}",
            })

            message_number += 1

    # Shuffle the rows reproducibly.
    random.shuffle(rows)

    dataset = pd.DataFrame(rows)

    # Save the CSV file.
    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    return dataset


# ============================================================
# 6. DISPLAY DATASET SUMMARY
# ============================================================

def main():

    print("=" * 60)
    print("PHISHSHIFT - SYNTHETIC DATASET GENERATOR")
    print("=" * 60)

    dataset = generate_messages()

    total_templates = (
        len(LEGITIMATE_TEMPLATES)
        + len(PHISHING_TEMPLATES)
    )

    print(f"\nDataset saved to:\n{OUTPUT_FILE}")

    print(f"\nTotal generated messages: {len(dataset)}")

    print(f"Total templates: {total_templates}")

    print(
        "Legitimate templates:",
        len(LEGITIMATE_TEMPLATES),
    )

    print(
        "Phishing templates:",
        len(PHISHING_TEMPLATES),
    )

    print("\nLabel distribution:")
    print(dataset["label"].value_counts().to_string())

    print("\nUnique campaign groups:")
    print(dataset["campaign"].nunique())

    print("\nMissing values:")
    print(dataset.isnull().sum().to_string())

    print("\nSample messages:")
    print(
        dataset[["text", "label", "campaign"]]
        .head(10)
        .to_string(index=False)
    )

    print("\nDataset generation completed successfully.")


# ============================================================
# 7. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()