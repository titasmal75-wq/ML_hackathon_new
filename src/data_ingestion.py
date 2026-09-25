import shutil
import logging
from pathlib import Path


# ============================================================
# Logging
# ============================================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.INFO)

if not logger.handlers:

    console_handler = logging.StreamHandler()

    file_handler = logging.FileHandler(
        LOG_DIR / "data_ingestion.log"
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

# IMPORTANT:
# Your actual dataset is inside student_resource/dataset
SOURCE_DIR = Path("student_resource/dataset")

# DVC will store the ingested data here
RAW_DIR = Path("data/raw")


# ============================================================
# Training files
# ============================================================

TRAIN_FILES = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv",
]


# ============================================================
# Test files
# ============================================================

TEST_FILES = [
    "test_source1.tsv",
    "test_source2.tsv",
    "test_source3.tsv",
]


# ============================================================
# Copy files
# ============================================================

def copy_files():

    try:

        # Create directories

        train_raw_dir = RAW_DIR / "train"
        test_raw_dir = RAW_DIR / "test"

        train_raw_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        test_raw_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ====================================================
        # Copy training files
        # ====================================================

        logger.info("Starting training data ingestion...")

        for file_name in TRAIN_FILES:

            source_file = (
                SOURCE_DIR
                / "train"
                / file_name
            )

            destination_file = (
                train_raw_dir
                / file_name
            )

            if not source_file.exists():

                raise FileNotFoundError(
                    f"Training file not found: {source_file}"
                )

            shutil.copy2(
                source_file,
                destination_file
            )

            logger.info(
                "Copied: %s",
                source_file
            )

        # ====================================================
        # Copy test files
        # ====================================================

        logger.info("Starting test data ingestion...")

        for file_name in TEST_FILES:

            source_file = (
                SOURCE_DIR
                / "test"
                / file_name
            )

            destination_file = (
                test_raw_dir
                / file_name
            )

            if not source_file.exists():

                raise FileNotFoundError(
                    f"Test file not found: {source_file}"
                )

            shutil.copy2(
                source_file,
                destination_file
            )

            logger.info(
                "Copied: %s",
                source_file
            )

        # ====================================================
        # Completed
        # ====================================================

        logger.info(
            "Data ingestion completed successfully!"
        )

    except Exception as e:

        logger.error(
            "Data ingestion failed: %s",
            e
        )

        raise


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    copy_files()