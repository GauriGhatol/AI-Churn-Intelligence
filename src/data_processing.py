"""
Data processing pipeline for the AI Churn Intelligence project.

Responsibilities:
1. Load raw customer data
2. Clean data
3. Convert data types
4. Encode the churn target
5. Save processed data
"""

from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DATA_FILE = RAW_DATA_DIR / "customer_churn.csv"
PROCESSED_DATA_FILE = (
    PROCESSED_DATA_DIR / "cleaned_customer_churn.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    """Load the raw customer churn dataset."""

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_FILE}"
        )

    df = pd.read_csv(RAW_DATA_FILE)

    print(f"Dataset loaded successfully: {df.shape}")

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the raw customer dataset.
    """

    cleaned_df = df.copy()

    # --------------------------------------------------------
    # 1. Remove duplicate rows
    # --------------------------------------------------------

    duplicates_before = cleaned_df.duplicated().sum()

    cleaned_df = cleaned_df.drop_duplicates()

    print(
        f"Duplicates removed: {duplicates_before}"
    )

    # --------------------------------------------------------
    # 2. Clean TotalCharges
    # --------------------------------------------------------

    cleaned_df["TotalCharges"] = pd.to_numeric(
        cleaned_df["TotalCharges"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # 3. Handle missing TotalCharges values
    # --------------------------------------------------------

    missing_total_charges = (
        cleaned_df["TotalCharges"].isna().sum()
    )

    print(
        f"Missing TotalCharges after conversion: "
        f"{missing_total_charges}"
    )

    # For customers with no recorded total charges,
    # use 0 because they have effectively no accumulated
    # charges yet.
    cleaned_df["TotalCharges"] = (
        cleaned_df["TotalCharges"].fillna(0)
    )

    # --------------------------------------------------------
    # 4. Convert SeniorCitizen to categorical values
    # --------------------------------------------------------

    cleaned_df["SeniorCitizen"] = cleaned_df[
        "SeniorCitizen"
    ].map({
        0: "No",
        1: "Yes"
    })

    # --------------------------------------------------------
    # 5. Convert Churn into machine-learning target
    # --------------------------------------------------------

    cleaned_df["Churn"] = cleaned_df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # --------------------------------------------------------
    # 6. Remove customerID
    # --------------------------------------------------------

    # Customer ID is an identifier, not a useful predictive
    # feature.
    cleaned_df = cleaned_df.drop(
        columns=["customerID"]
    )

    return cleaned_df


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_processed_data(df: pd.DataFrame) -> None:
    """Save the cleaned dataset."""

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_FILE,
        index=False
    )

    print(
        f"Processed dataset saved to: "
        f"{PROCESSED_DATA_FILE}"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """Run the complete data processing pipeline."""

    print("=" * 60)
    print("AI CHURN INTELLIGENCE - DATA PROCESSING")
    print("=" * 60)

    # Load
    df = load_data()

    print("\nOriginal columns:")
    print(df.columns.tolist())

    # Clean
    cleaned_df = clean_data(df)

    print("\nCleaned dataset shape:")
    print(cleaned_df.shape)

    print("\nCleaned data types:")
    print(cleaned_df.dtypes)

    print("\nChurn distribution:")
    print(cleaned_df["Churn"].value_counts())

    # Save
    save_processed_data(cleaned_df)

    print("\n" + "=" * 60)
    print("DATA PROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
