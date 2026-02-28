"""
main.py — Single entrypoint to run the complete churn prediction pipeline.

Steps:
  1. Generate 200K synthetic dataset
  2. Clean + feature engineer + encode
  3. Train base models + ensembles + Optuna tuning
  4. Evaluate all models + generate all plots
  5. Run business churn segmentation

Usage:
  python main.py                    # Full pipeline
  python main.py --skip-data        # Skip data generation (use existing CSV)
  python main.py --skip-train       # Skip training (evaluate existing models)
  python main.py --no-optuna        # Skip Optuna tuning (faster run)
"""

import argparse
import time
from src.utils import load_config, setup_logger, ensure_dirs

logger = setup_logger("main")


def parse_args():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data generation and use existing raw CSV")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training and load existing models")
    parser.add_argument("--no-optuna", action="store_true",
                        help="Skip Optuna hyperparameter tuning")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to config YAML file")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    ensure_dirs(config)

    start = time.time()
    logger.info("=" * 60)
    logger.info("  CUSTOMER CHURN PREDICTION — FULL PIPELINE")
    logger.info("=" * 60)

    # ── STEP 1: Data Generation ──────────────────────────────
    if not args.skip_data:
        logger.info("\n[STEP 1] Data Generation")
        from src.data_generator import generate_churn_dataset
        df = generate_churn_dataset(
            n=config["data"]["n_samples"],
            random_state=config["data"]["random_state"],
        )
        df.to_csv(config["data"]["raw_path"], index=False)
        logger.info(f"Saved: {config['data']['raw_path']}")
    else:
        logger.info("\n[STEP 1] Skipping data generation (--skip-data)")

    # ── STEP 2: Preprocessing + Feature Engineering ──────────
    logger.info("\n[STEP 2] Preprocessing & Feature Engineering")
    import pandas as pd
    from src.data_generator import generate_churn_dataset
    from src.preprocessing import clean_data, encode_features
    from src.feature_engineering import engineer_features

    try:
        df_raw = pd.read_csv(config["data"]["raw_path"])
    except FileNotFoundError:
        logger.warning("Raw data not found — regenerating...")
        df_raw = generate_churn_dataset(
            n=config["data"]["n_samples"],
            random_state=config["data"]["random_state"],
        )

    df_clean = clean_data(df_raw)
    df_fe = engineer_features(df_clean)
    df_encoded = encode_features(df_fe, config)
    df_encoded.to_csv(config["data"]["processed_path"], index=False)
    logger.info(f"Saved: {config['data']['processed_path']}")

    # EDA plots (use pre-encoded df for readability)
    logger.info("\n[STEP 2b] Generating EDA plots")
    from src.evaluate import plot_eda
    plot_eda(df_clean, save_dir=config["output"]["figures_dir"])

    # ── STEP 3: Training ─────────────────────────────────────
    if not args.skip_train:
        logger.info("\n[STEP 3] Model Training")

        # Patch config if --no-optuna
        if args.no_optuna:
            config["optuna"]["n_trials"] = 0
            logger.info("Optuna tuning disabled (--no-optuna)")

        from src.train import run_training_pipeline
        training_output = run_training_pipeline(config)
    else:
        logger.info("\n[STEP 3] Skipping training (--skip-train)")
        training_output = None

    # ── STEP 4: Evaluation ───────────────────────────────────
    logger.info("\n[STEP 4] Model Evaluation & Visualization")
    from src.evaluate import run_evaluation_pipeline
    run_evaluation_pipeline(training_output=training_output, config=config)

    # ── Done ─────────────────────────────────────────────────
    elapsed = time.time() - start
    logger.info("\n" + "=" * 60)
    logger.info(f"  ✅ PIPELINE COMPLETE — {elapsed / 60:.1f} minutes")
    logger.info("=" * 60)
    logger.info(f"\n  Output files:")
    logger.info(f"    Data      → {config['data']['processed_path']}")
    logger.info(f"    Models    → {config['output']['models_dir']}")
    logger.info(f"    Figures   → {config['output']['figures_dir']}")
    logger.info("\n  Next steps:")
    logger.info("    python src/predict.py          # Score new customers")
    logger.info("    pytest tests/ -v               # Run unit tests")


if __name__ == "__main__":
    main()
