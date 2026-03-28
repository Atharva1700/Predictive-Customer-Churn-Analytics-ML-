"""
predict.py — Load trained model and score new customers.

Usage:
  python src/predict.py --input data/new_customers.csv --output predictions.csv
  python src/predict.py  (uses built-in demo sample)
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.feature_engineering import engineer_features
from src.preprocessing import clean_data, encode_features
from src.utils import load_config, setup_logger

logger = setup_logger(__name__)


def load_model_and_scaler(config: dict):
    """Load the production stacking model and scaler from disk."""
    models_dir = Path(config["output"]["models_dir"])
    model_path = models_dir / config["output"]["model_filename"]
    scaler_path = models_dir / config["output"]["scaler_filename"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run `python src/train.py` first."
        )

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info(f"Loaded model: {model_path}")
    logger.info(f"Loaded scaler: {scaler_path}")
    return model, scaler


def preprocess_new_data(df: pd.DataFrame, config: dict, scaler, training_columns: list) -> np.ndarray:
    """
    Apply the same preprocessing pipeline to new customer data.
    Handles missing columns gracefully (fills with 0).
    """
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_features(df, config)

    # Remove target if accidentally included
    target = config["features"]["target"]
    if target in df.columns:
        df.drop(columns=[target], inplace=True)

    # Remove customer_id if present
    for col in config["features"]["drop_cols"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Align columns with training set — add missing as 0, drop extra
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[training_columns]

    return scaler.transform(df)


def predict(df: pd.DataFrame, model, X_scaled: np.ndarray,
            config: dict) -> pd.DataFrame:
    """
    Generate churn predictions and risk segments.

    Returns DataFrame with:
      - customer_id (if present)
      - churn_probability
      - churn_prediction (0/1)
      - risk_segment (Low/Medium/High)
    """
    seg_cfg = config["segmentation"]
    proba = model.predict_proba(X_scaled)[:, 1]
    pred = (proba >= config["evaluation"]["threshold"]).astype(int)

    risk = pd.cut(
        proba,
        bins=[0, seg_cfg["medium_risk_threshold"],
              seg_cfg["high_risk_threshold"], 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    result = pd.DataFrame({
        "churn_probability": proba.round(4),
        "churn_prediction": pred,
        "risk_segment": risk,
    })

    if "customer_id" in df.columns:
        result.insert(0, "customer_id", df["customer_id"].values)

    return result


def make_demo_sample() -> pd.DataFrame:
    """Create a small demo customer sample for testing inference."""
    data = {
        "customer_id": ["DEMO001", "DEMO002", "DEMO003", "DEMO004"],
        "gender": ["Male", "Female", "Male", "Female"],
        "senior_citizen": [0, 1, 0, 0],
        "partner": ["Yes", "No", "No", "Yes"],
        "dependents": ["No", "No", "Yes", "No"],
        "tenure": [2, 48, 12, 3],                      # New + loyal + mid + very new
        "phone_service": ["Yes", "Yes", "Yes", "Yes"],
        "multiple_lines": ["No", "Yes", "No", "No"],
        "internet_service": ["Fiber optic", "DSL", "Fiber optic", "Fiber optic"],
        "online_security": ["No", "Yes", "No", "No"],
        "online_backup": ["No", "Yes", "No", "No"],
        "device_protection": ["No", "Yes", "No", "No"],
        "tech_support": ["No", "Yes", "No", "No"],
        "streaming_tv": ["Yes", "No", "Yes", "Yes"],
        "streaming_movies": ["Yes", "No", "Yes", "Yes"],
        "contract": ["Month-to-month", "Two year", "Month-to-month", "Month-to-month"],
        "paperless_billing": ["Yes", "No", "Yes", "Yes"],
        "payment_method": ["Electronic check", "Bank transfer (automatic)",
                           "Electronic check", "Electronic check"],
        "monthly_charges": [89.95, 51.85, 79.65, 94.75],
        "total_charges": [179.90, 2488.80, 955.80, 284.25],
    }
    return pd.DataFrame(data)


def get_training_columns(config: dict) -> list:
    """
    Try to load training column names from processed dataset.
    Falls back to reconstructing from config.
    """
    try:
        df = pd.read_csv(config["data"]["processed_path"], nrows=0)
        cols = [c for c in df.columns
                if c != config["features"]["target"]
                and c not in config["features"]["drop_cols"]]
        return cols
    except FileNotFoundError:
        logger.warning("Processed data not found — using config to reconstruct columns")
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn Prediction Inference")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input CSV with new customer data")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Output CSV path for predictions")
    args = parser.parse_args()

    config = load_config()
    model, scaler = load_model_and_scaler(config)

    # Load input data
    if args.input:
        df_input = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df_input):,} records from {args.input}")
    else:
        logger.info("No input file provided — using demo sample")
        df_input = make_demo_sample()

    # Get training column alignment
    training_cols = get_training_columns(config)
    if not training_cols:
        logger.error("Cannot align features — run full pipeline first to generate processed data.")
        exit(1)

    # Preprocess
    X_new = preprocess_new_data(df_input.copy(), config, scaler, training_cols)

    # Predict
    predictions = predict(df_input, model, X_new, config)

    # Save & display
    predictions.to_csv(args.output, index=False)
    logger.info(f"\nPredictions saved → {args.output}")
    logger.info(f"\n{predictions.to_string(index=False)}")

    # Summary
    seg_counts = predictions["risk_segment"].value_counts()
    logger.info(f"\nRisk segment summary:\n{seg_counts.to_string()}")
