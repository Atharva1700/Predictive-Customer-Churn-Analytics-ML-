"""
data_generator.py — Generate a realistic 200K customer churn dataset.

Churn is statistically correlated with:
  - Month-to-month contracts (+28% churn probability)
  - Short tenure (< 6 months adds +20%)
  - High monthly charges (> $80 adds +12%)
  - No tech support or online security (+8% each)
  - Fiber optic internet (+8%)
  - Electronic check payment (+8%)
  - Senior citizen status (+5%)

Run standalone:
  python src/data_generator.py
"""

import numpy as np
import pandas as pd
from src.utils import load_config, setup_logger

logger = setup_logger(__name__)


def generate_churn_dataset(n: int = 200_000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic Telco-style churn dataset.

    Args:
        n: Number of records to generate
        random_state: Random seed for reproducibility

    Returns:
        pd.DataFrame with all raw features and binary churn label
    """
    np.random.seed(random_state)
    logger.info(f"Generating {n:,} customer records...")

    df = pd.DataFrame()

    # ── Demographics ─────────────────────────────────────────
    df["customer_id"] = [f"CUST{str(i).zfill(7)}" for i in range(n)]
    df["gender"] = np.random.choice(["Male", "Female"], n)
    df["senior_citizen"] = np.random.choice([0, 1], n, p=[0.84, 0.16])
    df["partner"] = np.random.choice(["Yes", "No"], n, p=[0.48, 0.52])
    df["dependents"] = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])

    # ── Tenure ───────────────────────────────────────────────
    df["tenure"] = (
        np.random.gamma(shape=2.5, scale=14, size=n).clip(1, 72).astype(int)
    )

    # ── Contract ─────────────────────────────────────────────
    df["contract"] = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        n,
        p=[0.55, 0.24, 0.21],
    )

    # ── Internet service ─────────────────────────────────────
    df["internet_service"] = np.random.choice(
        ["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22]
    )

    # ── Add-on services (conditional on internet) ────────────
    for col, prob in [
        ("online_security", 0.29),
        ("tech_support", 0.29),
        ("online_backup", 0.34),
        ("device_protection", 0.34),
        ("streaming_tv", 0.38),
        ("streaming_movies", 0.39),
    ]:
        df[col] = np.where(
            df["internet_service"] == "No",
            "No internet service",
            np.random.choice(["Yes", "No"], n, p=[prob, 1 - prob]),
        )

    # ── Phone service ────────────────────────────────────────
    df["phone_service"] = np.random.choice(["Yes", "No"], n, p=[0.90, 0.10])
    df["multiple_lines"] = np.where(
        df["phone_service"] == "No",
        "No phone service",
        np.random.choice(["Yes", "No"], n, p=[0.42, 0.58]),
    )

    # ── Billing ──────────────────────────────────────────────
    df["paperless_billing"] = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
    df["payment_method"] = np.random.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        n,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # ── Charges ──────────────────────────────────────────────
    base_charge = np.where(
        df["internet_service"] == "Fiber optic",
        np.random.normal(80, 20, n),
        np.where(
            df["internet_service"] == "DSL",
            np.random.normal(55, 15, n),
            np.random.normal(25, 8, n),
        ),
    )
    df["monthly_charges"] = base_charge.clip(18, 120).round(2)
    df["total_charges"] = (
        df["monthly_charges"] * df["tenure"] * np.random.uniform(0.9, 1.1, n)
    ).round(2)

    # ── Churn probability model ───────────────────────────────
    p = np.full(n, 0.05)  # base rate

    p += np.where(df["contract"] == "Month-to-month", 0.28, 0)
    p += np.where(df["contract"] == "One year", 0.05, 0)
    p += np.where(df["tenure"] <= 6, 0.20, 0)
    p += np.where(df["tenure"] <= 12, 0.10, 0)
    p -= np.where(df["tenure"] >= 48, 0.12, 0)
    p += np.where(df["monthly_charges"] > 80, 0.12, 0)
    p += np.where(df["tech_support"] == "No", 0.08, 0)
    p += np.where(df["online_security"] == "No", 0.07, 0)
    p += np.where(df["internet_service"] == "Fiber optic", 0.08, 0)
    p += np.where(df["payment_method"] == "Electronic check", 0.08, 0)
    p += df["senior_citizen"] * 0.05

    p = p.clip(0, 0.95)
    df["churn"] = (np.random.uniform(0, 1, n) < p).astype(int)

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Churn rate: {df['churn'].mean():.2%}")
    logger.info(f"Class distribution: {df['churn'].value_counts().to_dict()}")

    return df


if __name__ == "__main__":
    config = load_config()
    df = generate_churn_dataset(
        n=config["data"]["n_samples"],
        random_state=config["data"]["random_state"],
    )
    out_path = config["data"]["raw_path"]
    df.to_csv(out_path, index=False)
    logger.info(f"Saved raw dataset → {out_path}")
