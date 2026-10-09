"""
Inspect the raw customer churn dataset.
"""

from pathlib import Path

import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset location
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "customer_churn.csv"
)


def inspect_dataset():
    """Load and inspect the customer churn dataset."""

    print("=" * 60)
    print("CUSTOMER CHURN DATASET INSPECTION")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    print("\n1. Dataset shape:")
    print(df.shape)

    print("\n2. Column names:")
    print(df.columns.tolist())

    print("\n3. First 5 rows:")
    print(df.head())

    print("\n4. Data types:")
    print(df.dtypes)

    print("\n5. Missing values:")
    print(df.isnull().sum())

    print("\n6. Duplicate rows:")
    print(df.duplicated().sum())

    print("\n7. Statistical summary:")
    print(df.describe(include="all").transpose())

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    inspect_dataset()
