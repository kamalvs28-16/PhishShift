
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

INPUT_FILE = DATA_DIR / "messages.csv"
OUTPUT_FILE = DATA_DIR / "messages_cleaned.csv"
REPORT_FILE = REPORTS_DIR / "duplicate_report.json"

SIMILARITY_THRESHOLD = 0.98

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. NORMALIZE MESSAGE TEXT
# ============================================================

def normalize_text(text):
    """
    Normalize text for duplicate detection.

    Converts text to lowercase, removes punctuation,
    and standardizes whitespace.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text


# ============================================================
# 3. STANDARDIZE COLUMN NAMES
# ============================================================

def standardize_columns(df):
    """
    Support the current dataset generator and legacy formats.

    Current columns:
        message_id, text, label, campaign

    Legacy columns:
        message_id, body_text, label, campaign_group
    """

    rename_map = {}

    if "text" in df.columns and "body_text" not in df.columns:
        rename_map["text"] = "body_text"

    if "campaign" in df.columns and "campaign_group" not in df.columns:
        rename_map["campaign"] = "campaign_group"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


# ============================================================
# 4. LOAD THE DATASET
# ============================================================

def load_dataset():
    """Load the CSV file and validate required columns."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}\n"
            "Run this command first:\n"
            "python src\\generate_dataset.py"
        )

    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as error:
        raise ValueError(
            f"Could not read the dataset: {error}"
        ) from error

    df = standardize_columns(df)

    required_columns = [
        "message_id",
        "body_text",
        "label",
        "campaign_group",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    return df


# ============================================================
# 5. VALIDATE AND CLEAN ROWS
# ============================================================

def validate_and_clean(df):
    """
    Validate records, normalize messages, and remove exact
    duplicates or messages with conflicting labels.
    """

    original_count = len(df)

    # Clean message IDs.
    df["message_id"] = (
        df["message_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Clean message text.
    df["body_text"] = (
        df["body_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Normalize labels.
    df["label"] = (
        df["label"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Clean campaign identifiers.
    df["campaign_group"] = (
        df["campaign_group"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Keep records with all required information.
    valid_mask = (
        (df["message_id"] != "")
        & (df["body_text"] != "")
        & (df["campaign_group"] != "")
        & (df["label"].isin(["legitimate", "phishing"]))
    )

    df = df.loc[valid_mask].copy()

    valid_count = len(df)
    invalid_count = original_count - valid_count

    # Normalize text for duplicate comparisons.
    df["normalized_text"] = (
        df["body_text"].apply(normalize_text)
    )

    # Remove messages that normalize to empty text.
    empty_mask = df["normalized_text"] == ""
    empty_normalized_count = int(empty_mask.sum())

    df = df.loc[~empty_mask].copy()

    # Find identical normalized text with conflicting labels.
    label_counts = (
        df.groupby("normalized_text")["label"].nunique()
    )

    conflicting_texts = set(
        label_counts[label_counts > 1].index
    )

    conflicting_message_count = int(
        df["normalized_text"].isin(conflicting_texts).sum()
    )

    # Remove conflicting groups instead of arbitrarily
    # retaining one label for identical message text.
    if conflicting_texts:
        df = df[
            ~df["normalized_text"].isin(conflicting_texts)
        ].copy()

    # Remove exact duplicate messages after normalization.
    duplicate_mask = df.duplicated(
        subset=["normalized_text"],
        keep="first",
    )

    exact_duplicates_removed = int(duplicate_mask.sum())

    cleaned = df.loc[~duplicate_mask].copy()

    statistics = {
        "original_message_count": int(original_count),
        "valid_message_count": int(valid_count),
        "invalid_message_count": int(invalid_count),
        "empty_normalized_message_count": int(
            empty_normalized_count
        ),
        "conflicting_label_message_count": int(
            conflicting_message_count
        ),
        "removed_exact_duplicate_count": int(
            exact_duplicates_removed
        ),
        "cleaned_message_count": int(len(cleaned)),
    }

    return cleaned, statistics


# ============================================================
# 6. FIND NEAR-DUPLICATE MESSAGES
# ============================================================

def find_near_duplicates(cleaned):
    """
    Identify highly similar messages using character n-gram
    TF-IDF and cosine similarity.

    Near-duplicate pairs are reported for review. They are
    not automatically deleted.
    """

    pairs = []

    if len(cleaned) < 2:
        return pairs

    texts = cleaned["normalized_text"].tolist()

    try:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
        )

        matrix = vectorizer.fit_transform(texts)

        # Calculate pairwise cosine similarities.
        similarities = cosine_similarity(matrix)

        records = cleaned.reset_index(drop=True)

        for i in range(len(records)):
            for j in range(i + 1, len(records)):

                similarity = float(similarities[i, j])

                if similarity >= SIMILARITY_THRESHOLD:

                    first = records.iloc[i]
                    second = records.iloc[j]

                    pairs.append({
                        "message_id_1": str(
                            first["message_id"]
                        ),
                        "message_id_2": str(
                            second["message_id"]
                        ),
                        "similarity": round(similarity, 4),
                        "same_label": bool(
                            first["label"] == second["label"]
                        ),
                        "same_campaign": bool(
                            first["campaign_group"]
                            == second["campaign_group"]
                        ),
                    })

    except ValueError as error:
        print(
            "WARNING: Near-duplicate detection was skipped:",
            error,
        )

    return pairs


# ============================================================
# 7. SAVE CLEANED DATASET AND REPORT
# ============================================================

def save_outputs(cleaned, statistics, near_duplicate_pairs):
    """Save the cleaned CSV and JSON report."""

    # Remove temporary processing columns.
    output_df = cleaned.drop(
        columns=["normalized_text"],
        errors="ignore",
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    label_distribution = {
        str(label): int(count)
        for label, count in (
            output_df["label"].value_counts().items()
        )
    }

    report = {
        **statistics,
        "near_duplicate_similarity_threshold": (
            SIMILARITY_THRESHOLD
        ),
        "near_duplicate_pair_count": len(
            near_duplicate_pairs
        ),
        "near_duplicate_pairs": near_duplicate_pairs,
        "label_distribution": label_distribution,
        "campaign_group_count": int(
            output_df["campaign_group"].nunique()
        ),
        "note": (
            "Near-duplicate pairs are flagged for review, "
            "not automatically removed. Exact duplicates are "
            "removed using normalized message text. This report "
            "does not by itself prove that campaign leakage "
            "has been eliminated."
        ),
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=4)

    return output_df


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

def display_results(output_df, statistics, near_duplicate_pairs):
    """Display preprocessing statistics."""

    print("\n" + "=" * 60)
    print("PREPROCESSING RESULTS")
    print("=" * 60)

    print(
        f"Original messages: "
        f"{statistics['original_message_count']}"
    )

    print(
        f"Valid messages: "
        f"{statistics['valid_message_count']}"
    )

    print(
        f"Invalid messages removed: "
        f"{statistics['invalid_message_count']}"
    )

    print(
        f"Empty normalized messages removed: "
        f"{statistics['empty_normalized_message_count']}"
    )

    print(
        f"Messages involved in conflicting labels: "
        f"{statistics['conflicting_label_message_count']}"
    )

    print(
        f"Exact duplicates removed: "
        f"{statistics['removed_exact_duplicate_count']}"
    )

    print(
        f"Cleaned messages: "
        f"{statistics['cleaned_message_count']}"
    )

    print(
        f"Near-duplicate pairs flagged: "
        f"{len(near_duplicate_pairs)}"
    )

    print(
        f"Unique campaign groups: "
        f"{output_df['campaign_group'].nunique()}"
    )

    print("\nLabel distribution:")

    if not output_df.empty:
        print(
            output_df["label"]
            .value_counts()
            .to_string()
        )
    else:
        print("WARNING: The cleaned dataset is empty.")

    print(f"\nCleaned dataset: {OUTPUT_FILE}")
    print(f"Duplicate report: {REPORT_FILE}")


# ============================================================
# 9. MAIN FUNCTION
# ============================================================

def main():

    print("=" * 60)
    print("PHISHSHIFT - DATA PREPROCESSING")
    print("=" * 60)

    try:
        # Load source data.
        df = load_dataset()

        # Validate and clean records.
        cleaned, statistics = validate_and_clean(df)

        # Prevent downstream model training on an empty dataset.
        if cleaned.empty:
            print(
                "\nERROR: No valid messages remain after cleaning."
            )
            return

        # Identify near-duplicate pairs.
        near_duplicate_pairs = find_near_duplicates(cleaned)

        # Save results.
        output_df = save_outputs(
            cleaned,
            statistics,
            near_duplicate_pairs,
        )

        # Display summary.
        display_results(
            output_df,
            statistics,
            near_duplicate_pairs,
        )

        print("\nPreprocessing completed successfully.")

    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"\nERROR: {error}")
        return


# ============================================================
# 10. EXECUTION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()