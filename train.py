"""
train.py — Model training pipeline.

Trains:
  1. Base models: Random Forest, XGBoost, LightGBM, Logistic Regression
  2. Soft Voting Ensemble
  3. Stacking Ensemble (meta-learner = Logistic Regression)
  4. Optuna-tuned XGBoost (optional)

Saves all models to models/ directory.

Run standalone:
  python src/train.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.utils import load_config, setup_logger, ensure_dirs

logger = setup_logger(__name__)

# ── Optional imports ────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("xgboost not installed — using GradientBoostingClassifier")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("lightgbm not installed — ensemble will use available models")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("optuna not installed — skipping hyperparameter tuning")


# ─────────────────────────────────────────────────────────────
# BASE MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────

def build_base_models(config: dict) -> dict:
    """Instantiate all base models from config."""
    rf_cfg = config["models"]["random_forest"]
    lr_cfg = config["models"]["logistic_regression"]

    models = {
        "Random Forest": RandomForestClassifier(**rf_cfg),
        "Logistic Regression": LogisticRegression(**lr_cfg),
    }

    if HAS_XGB:
        xgb_cfg = config["models"]["xgboost"]
        models["XGBoost"] = XGBClassifier(**xgb_cfg)
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42
        )

    if HAS_LGB:
        lgb_cfg = config["models"]["lightgbm"]
        models["LightGBM"] = LGBMClassifier(**lgb_cfg)

    logger.info(f"Base models: {list(models.keys())}")
    return models


# ─────────────────────────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────

def cross_validate_models(models: dict, X_train, y_train, config: dict) -> dict:
    """Run 5-fold CV for each base model and report AUC-ROC."""
    cv_folds = config["evaluation"]["cv_folds"]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                         random_state=config["data"]["random_state"])

    cv_results = {}
    logger.info(f"Running {cv_folds}-fold cross-validation...")

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_results[name] = {"mean": scores.mean(), "std": scores.std()}
        logger.info(f"  {name:<25} AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    return cv_results


# ─────────────────────────────────────────────────────────────
# ENSEMBLE BUILDING
# ─────────────────────────────────────────────────────────────

def build_soft_voting(models: dict, X_train, y_train) -> VotingClassifier:
    """Soft Voting: average predicted probabilities across all base models."""
    logger.info("Building Soft Voting Ensemble...")

    estimators = [
        (name.lower().replace(" ", "_"), model)
        for name, model in models.items()
    ]
    voting = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    voting.fit(X_train, y_train)

    logger.info("Soft Voting Ensemble trained ✓")
    return voting


def build_stacking(models: dict, X_train, y_train, config: dict) -> StackingClassifier:
    """
    Stacking Classifier:
    - Level-0: all base models except Logistic Regression
    - Level-1 (meta-learner): Logistic Regression
    Uses 5-fold CV to generate out-of-fold predictions for Level-1.
    """
    logger.info("Building Stacking Ensemble...")

    stack_cfg = config["models"]["stacking"]

    # Exclude LR from base — it's the meta-learner
    estimators = [
        (name.lower().replace(" ", "_"), model)
        for name, model in models.items()
        if "Logistic" not in name
    ]

    meta_learner = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=stack_cfg["cv"],
        stack_method=stack_cfg["stack_method"],
        passthrough=stack_cfg["passthrough"],
        n_jobs=-1,
    )

    logger.info(f"  Base models: {[e[0] for e in estimators]}")
    logger.info("  Meta-learner: Logistic Regression")
    logger.info("  Fitting (5-fold CV internally)... this takes ~3–5 minutes")

    stacking.fit(X_train, y_train)
    logger.info("Stacking Ensemble trained ✓")
    return stacking


# ─────────────────────────────────────────────────────────────
# OPTUNA HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────

def tune_with_optuna(X_train, y_train, config: dict):
    """
    Tune XGBoost with Optuna TPE sampler.
    Optimizes AUC-ROC via 3-fold CV.
    Returns best model trained on full training data.
    """
    if not HAS_OPTUNA or not HAS_XGB:
        logger.warning("Optuna or XGBoost unavailable — skipping tuning")
        return None

    n_trials = config["optuna"]["n_trials"]
    logger.info(f"Running Optuna tuning — {n_trials} trials...")

    cv = StratifiedKFold(n_splits=3, shuffle=True,
                         random_state=config["data"]["random_state"])

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 6),
            "random_state": config["data"]["random_state"],
            "n_jobs": -1,
            "eval_metric": "logloss",
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        return scores.mean()

    study = optuna.create_study(
        direction=config["optuna"]["direction"],
        sampler=optuna.samplers.TPESampler(seed=config["data"]["random_state"]),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best AUC-ROC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    # Train final model on full training data with best params
    best_params = study.best_params
    best_params.update({
        "random_state": config["data"]["random_state"],
        "n_jobs": -1,
        "eval_metric": "logloss",
    })
    tuned_model = XGBClassifier(**best_params)
    tuned_model.fit(X_train, y_train)

    logger.info("Tuned XGBoost trained ✓")
    return tuned_model, study


# ─────────────────────────────────────────────────────────────
# SAVE MODELS
# ─────────────────────────────────────────────────────────────

def save_models(models_dict: dict, scaler, config: dict) -> None:
    """Serialize all models and scaler to disk."""
    out_dir = Path(config["output"]["models_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models_dict.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, out_dir / filename)
        logger.info(f"Saved: models/{filename}")

    joblib.dump(scaler, out_dir / config["output"]["scaler_filename"])
    logger.info(f"Saved: models/{config['output']['scaler_filename']}")


# ─────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────

def run_training_pipeline(config: dict = None):
    """
    Full training pipeline. Returns dict of trained models.
    Can be called from main.py or standalone.
    """
    if config is None:
        config = load_config()

    ensure_dirs(config)

    # ── Load & prepare data ──────────────────────────────────
    from src.data_generator import generate_churn_dataset
    from src.feature_engineering import engineer_features
    from src.preprocessing import clean_data, encode_features, split_and_scale

    try:
        df = pd.read_csv(config["data"]["processed_path"])
        logger.info(f"Loaded processed data from {config['data']['processed_path']}")
    except FileNotFoundError:
        logger.info("Processed data not found — running full preprocessing pipeline")
        try:
            df_raw = pd.read_csv(config["data"]["raw_path"])
        except FileNotFoundError:
            df_raw = generate_churn_dataset(
                n=config["data"]["n_samples"],
                random_state=config["data"]["random_state"],
            )
        df = clean_data(df_raw)
        df = engineer_features(df)
        df = encode_features(df, config)
        df.to_csv(config["data"]["processed_path"], index=False)

    (
        X_train_sc, X_test_sc,
        X_train_res, y_train_res,
        y_train, y_test,
        feature_names, scaler,
    ) = split_and_scale(df, config)

    # ── Train base models ────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("TRAINING BASE MODELS")
    logger.info("=" * 55)

    base_models = build_base_models(config)
    cv_results = cross_validate_models(base_models, X_train_res, y_train_res, config)

    trained_base = {}
    for name, model in base_models.items():
        logger.info(f"Fitting {name} on full training set...")
        model.fit(X_train_res, y_train_res)
        trained_base[name] = model

    # ── Ensemble models ──────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info("BUILDING ENSEMBLES")
    logger.info("=" * 55)

    voting = build_soft_voting(trained_base, X_train_res, y_train_res)
    stacking = build_stacking(trained_base, X_train_res, y_train_res, config)

    # ── Optuna tuning ────────────────────────────────────────
    tuned_xgb = None
    study = None
    optuna_result = tune_with_optuna(X_train_res, y_train_res, config)
    if optuna_result:
        tuned_xgb, study = optuna_result

    # ── Collect all trained models ───────────────────────────
    all_models = {**trained_base, "Soft Voting": voting, "Stacking Ensemble": stacking}
    if tuned_xgb:
        all_models["Tuned XGBoost"] = tuned_xgb

    # ── Save ─────────────────────────────────────────────────
    save_models(all_models, scaler, config)

    logger.info("\n✅ Training pipeline complete!")
    logger.info(f"   Models saved to: {config['output']['models_dir']}")

    return {
        "models": all_models,
        "X_train": X_train_sc,
        "X_test": X_test_sc,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_resampled": X_train_res,
        "y_train_resampled": y_train_res,
        "feature_names": feature_names,
        "scaler": scaler,
        "cv_results": cv_results,
        "optuna_study": study,
    }


if __name__ == "__main__":
    run_training_pipeline()
