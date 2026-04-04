"""
feature_engineering.py — Derived feature creation for churn prediction.

New features engineered:
  - avg_monthly_revenue    : total_charges / tenure
  - num_services           : count of active add-on services
  - is_loyal               : tenure > 24 months flag
  - is_high_value          : monthly_charges > 75th percentile
  - no_support_services    : no tech support AND no online security
  - auto_payment           : automatic payment method flag
  - charge_per_service     : monthly_charges / (num_services + 1)
  - tenure_bucket          : binned tenure (0-6m, 6-12m, 12-24m, 24-48m, 48+m)

Run standalone:
  python src/feature_engineering.py
"""

import numpy as np
import pandas as pd
from src.utils import load_config, setup_logger

logger = setup_logger(__name__)

SERVICE_COLS = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]


def add_avg_monthly_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue per active month — handles new customers with tenure=0."""
    df["avg_monthly_revenue"] = df["total_charges"] / df["tenure"].replace(0, 1)
    return df


def add_num_services(df: pd.DataFrame) -> pd.DataFrame:
    """Count of subscribed add-on services (max 6)."""
    df["num_services"] = sum(
        (df[col] == "Yes").astype(int)
        for col in SERVICE_COLS
        if col in df.columns
    )
    return df


def add_loyalty_flag(df: pd.DataFrame, threshold: int = 24) -> pd.DataFrame:
    """1 if customer has been with the company > threshold months."""
    df["is_loyal"] = (df["tenure"] > threshold).astype(int)
    return df


def add_high_value_flag(df: pd.DataFrame, quantile: float = 0.75) -> pd.DataFrame:
    """1 if monthly charges exceed the 75th percentile."""
    cutoff = df["monthly_charges"].quantile(quantile)
    df["is_high_value"] = (df["monthly_charges"] > cutoff).astype(int)
    return df


def add_no_support_flag(df: pd.DataFrame) -> pd.DataFrame:
    """1 if customer has neither tech support nor online security."""
    df["no_support_services"] = (
        (df.get("tech_support", pd.Series("No", index=df.index)) == "No") &
        (df.get("online_security", pd.Series("No", index=df.index)) == "No")
    ).astype(int)
    return df


def add_auto_payment_flag(df: pd.DataFrame) -> pd.DataFrame:
    """1 if customer pays via automatic method (bank transfer or credit card)."""
    df["auto_payment"] = (
        df["payment_method"].str.contains("automatic", case=False, na=False)
    ).astype(int)
    return df


def add_charge_per_service(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly charges normalized by number of services used."""
    if "num_services" not in df.columns:
        df = add_num_services(df)
    df["charge_per_service"] = df["monthly_charges"] / (df["num_services"] + 1)
    return df


def add_tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Bin tenure into 5 meaningful customer lifecycle stages."""
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 48, 72],
        labels=["0-6m", "6-12m", "12-24m", "24-48m", "48+m"],
        include_lowest=True,
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in sequence.

    Args:
        df: Raw or cleaned DataFrame

    Returns:
        DataFrame with all engineered features added
    """
    logger.info("Engineering features...")

    df = df.copy()
    df = add_avg_monthly_revenue(df)
    df = add_num_services(df)
    df = add_loyalty_flag(df)
    df = add_high_value_flag(df)
    df = add_no_support_flag(df)
    df = add_auto_payment_flag(df)
    df = add_charge_per_service(df)
    df = add_tenure_bucket(df)

    new_features = [
        "avg_monthly_revenue", "num_services", "is_loyal",
        "is_high_value", "no_support_services", "auto_payment",
        "charge_per_service", "tenure_bucket",
    ]
    logger.info(f"Added features: {new_features}")
    logger.info(f"Dataset shape after FE: {df.shape}")

    return df


if __name__ == "__main__":
    config = load_config()

    try:
        df = pd.read_csv(config["data"]["raw_path"])
    except FileNotFoundError:
        from src.data_generator import generate_churn_dataset
        df = generate_churn_dataset(
            n=config["data"]["n_samples"],
            random_state=config["data"]["random_state"],
        )

    df = engineer_features(df)
    print(df[["tenure", "monthly_charges", "num_services",
              "charge_per_service", "is_loyal", "churn"]].head(10))
    print(f"\nFinal shape: {df.shape}")
