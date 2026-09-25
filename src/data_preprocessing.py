import pandas as pd
import re
import unicodedata
import logging
from pathlib import Path


# ============================================================
# Logging
# ============================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.INFO)

if not logger.handlers:

    console_handler = logging.StreamHandler()

    file_handler = logging.FileHandler(
        LOG_DIR / "data_preprocessing.log"
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ============================================================
# Paths
# ============================================================

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Replace &
    text = text.replace("&", " and ")

    # Keep Unicode characters
    # Important for Hindi and other languages
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# Business name normalization
# ============================================================

def normalize_name(text):

    text = normalize_text(text)

    replacements = {
        "corporation": "corp",
        "company": "co",
        "limited": "ltd",
        "private": "pvt",
        "incorporated": "inc",
    }

    words = text.split()

    words = [
        replacements.get(word, word)
        for word in words
    ]

    return " ".join(words)


# ============================================================
# Address normalization
# ============================================================

def normalize_address(text):

    text = normalize_text(text)

    replacements = {
        "road": "rd",
        "street": "st",
        "avenue": "ave",
        "boulevard": "blvd",
        "lane": "ln",
        "highway": "hwy",
        "apartment": "apt",
    }

    words = text.split()

    words = [
        replacements.get(word, word)
        for word in words
    ]

    return " ".join(words)


# ============================================================
# Source dataframe preprocessing
# ============================================================

def preprocess_source_dataframe(df):

    required_columns = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Business name
    df["name_clean"] = df["business_name"].apply(
        normalize_name
    )

    # Address
    df["address_clean"] = df["business_address"].apply(
        normalize_address
    )

    # Country
    df["country_clean"] = df["country"].apply(
        normalize_text
    )

    return df


# ============================================================
# Process source file
# ============================================================

def process_source_file(input_path, output_path):

    logger.info(
        "Processing source file: %s",
        input_path
    )

    df = pd.read_csv(
        input_path,
        sep="\t"
    )

    logger.info(
        "Original shape: %s",
        df.shape
    )

    df = preprocess_source_dataframe(df)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    logger.info(
        "Saved processed file: %s",
        output_path
    )

    logger.info(
        "Processed shape: %s",
        df.shape
    )


# ============================================================
# Process ground truth
# ============================================================

def process_ground_truth(input_path, output_path):

    logger.info(
        "Processing ground truth: %s",
        input_path
    )

    df = pd.read_csv(
        input_path,
        sep="\t"
    )

    required_columns = [
        "source1_entity_id",
        "matched_entity_ids",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Ground truth missing columns: {missing_columns}"
        )

    # DO NOT apply business-name preprocessing here.
    # Ground truth contains entity mappings.

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    logger.info(
        "Saved ground truth: %s",
        output_path
    )

    logger.info(
        "Ground truth shape: %s",
        df.shape
    )


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Training source files
    # ========================================================

    train_sources = [
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
    ]

    for file_name in train_sources:

        input_path = (
            RAW_DIR
            / "train"
            / file_name
        )

        output_path = (
            PROCESSED_DIR
            / "train"
            / file_name
        )

        process_source_file(
            input_path,
            output_path
        )

    # ========================================================
    # Ground truth
    # ========================================================

    ground_truth_input = (
        RAW_DIR
        / "train"
        / "train_ground_truth.tsv"
    )

    ground_truth_output = (
        PROCESSED_DIR
        / "train"
        / "train_ground_truth.tsv"
    )

    process_ground_truth(
        ground_truth_input,
        ground_truth_output
    )

    # ========================================================
    # Test source files
    # ========================================================

    test_sources = [
        "test_source1.tsv",
        "test_source2.tsv",
        "test_source3.tsv",
    ]

    for file_name in test_sources:

        input_path = (
            RAW_DIR
            / "test"
            / file_name
        )

        output_path = (
            PROCESSED_DIR
            / "test"
            / file_name
        )

        process_source_file(
            input_path,
            output_path
        )

    logger.info(
        "============================================"
    )

    logger.info(
        "DATA PREPROCESSING COMPLETED SUCCESSFULLY"
    )

    logger.info(
        "============================================"
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()