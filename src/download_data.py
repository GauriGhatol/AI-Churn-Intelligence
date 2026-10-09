"""
Download the customer churn dataset for the project.
"""

from pathlib import Path
import shutil

import kagglehub


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# Public Kaggle dataset
DATASET_ID = "blastchar/telco-customer-churn"


def download_dataset():
    """Download the Telco Customer Churn dataset."""

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Downloading customer churn dataset...")

    dataset_path = Path(
        kagglehub.dataset_download(DATASET_ID)
    )

    print(f"Dataset downloaded to: {dataset_path}")

    csv_files = list(dataset_path.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV file was found in the downloaded dataset."
        )

    source_file = csv_files[0]

    destination_file = (
        RAW_DATA_DIR / "customer_churn.csv"
    )

    shutil.copy2(
        source_file,
        destination_file
    )

    print(f"Dataset copied to: {destination_file}")
    print("Dataset download completed successfully!")


if __name__ == "__main__":
    download_dataset()