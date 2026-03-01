"""
evaluate.py — Model evaluation: metrics, plots, SHAP, and business segmentation.

Generates:
  - roc_curves.png
  - precision_recall_curves.png
  - confusion_matrix.png
  - model_comparison.png
  - feature_importance.png
  - shap_summary.png
  - eda_overview.png
  - churn_segmentation.png

Run standalone:
  python src/evaluate.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
)

from src.utils import load_config, setup_logger

logger = setup_logger(__name__)
sns.set_theme(style="whitegrid", palette="muted")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

PALETTE = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]


# ─────────────────────────────────────────────────────────────
# METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_metrics(model, X_test, y_test, name: str = "Model") -> dict:
    """Compute full suite of evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "avg_precision": average_precision_score(y_test, y_prob),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    logger.info(f"\n{'─' * 50}")
    logger.info(f"  {name}")
    logger.info(f"{'─' * 50}")
    logger.info(f"  Accuracy:      {metrics['accuracy']:.4f}")
    logger.info(f"  AUC-ROC:       {metrics['auc_roc']:.4f}")
    logger.info(f"  F1 Score:      {metrics['f1']:.4f}")
    logger.info(f"  Avg Precision: {metrics['avg_precision']:.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churned'])}")

    return metrics


def evaluate_all_models(models: dict, X_test, y_test) -> list:
    """Evaluate all models and return sorted list of metric dicts."""
    results = []
    for name, model in models.items():
        result = compute_metrics(model, X_test, y_test, name)
        results.append(result)
    return sorted(results, key=lambda r: r["auc_roc"], reverse=True)


# ─────────────────────────────────────────────────────────────
# EDA PLOTS
# ─────────────────────────────────────────────────────────────

def plot_eda(df: pd.DataFrame, save_dir: str = "reports/figures") -> None:
    """6-panel EDA overview."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Customer Churn — EDA Overview", fontsize=16, fontweight="bold")

    # 1. Churn pie
    churn_counts = df["churn"].value_counts()
    axes[0, 0].pie(churn_counts, labels=["No Churn", "Churned"],
                   autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c"],
                   startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[0, 0].set_title("Churn Distribution", fontweight="bold")

    # 2. Tenure distribution
    for val, label, color in [(0, "No Churn", "#2ecc71"), (1, "Churned", "#e74c3c")]:
        axes[0, 1].hist(df[df["churn"] == val]["tenure"], bins=30,
                        alpha=0.7, color=color, label=label)
    axes[0, 1].set_title("Tenure by Churn", fontweight="bold")
    axes[0, 1].set_xlabel("Tenure (months)")
    axes[0, 1].legend()

    # 3. Monthly charges
    for val, label, color in [(0, "No Churn", "#2ecc71"), (1, "Churned", "#e74c3c")]:
        axes[0, 2].hist(df[df["churn"] == val]["monthly_charges"], bins=40,
                        alpha=0.6, color=color, label=label)
    axes[0, 2].set_title("Monthly Charges by Churn", fontweight="bold")
    axes[0, 2].set_xlabel("Monthly Charges ($)")
    axes[0, 2].legend()

    # 4. Contract type
    contract_churn = df.groupby("contract")["churn"].mean().sort_values(ascending=False)
    bars = axes[1, 0].bar(contract_churn.index, contract_churn.values,
                           color=["#e74c3c", "#f39c12", "#2ecc71"], edgecolor="black")
    axes[1, 0].set_title("Churn Rate by Contract", fontweight="bold")
    axes[1, 0].set_ylabel("Churn Rate")
    for bar in bars:
        axes[1, 0].annotate(f"{bar.get_height():.1%}",
                             (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             ha="center", va="bottom", fontsize=10)

    # 5. Internet service
    internet_churn = df.groupby("internet_service")["churn"].mean().sort_values(ascending=False)
    bars = axes[1, 1].bar(internet_churn.index, internet_churn.values,
                           color=PALETTE[:3], edgecolor="black")
    axes[1, 1].set_title("Churn Rate by Internet Service", fontweight="bold")
    axes[1, 1].set_ylabel("Churn Rate")
    for bar in bars:
        axes[1, 1].annotate(f"{bar.get_height():.1%}",
                             (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             ha="center", va="bottom", fontsize=10)

    # 6. Payment method
    payment_churn = df.groupby("payment_method")["churn"].mean().sort_values()
    axes[1, 2].barh(payment_churn.index, payment_churn.values,
                    color="#3498db", edgecolor="black")
    axes[1, 2].set_title("Churn Rate by Payment Method", fontweight="bold")
    axes[1, 2].set_xlabel("Churn Rate")

    plt.tight_layout()
    path = f"{save_dir}/eda_overview.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────
# EVALUATION PLOTS
# ─────────────────────────────────────────────────────────────

def plot_roc_curves(results: list, y_test, save_dir: str = "reports/figures") -> None:
    """ROC curves for all models."""
    plt.figure(figsize=(10, 7))

    for i, res in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        plt.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=2,
                 label=f"{res['name']} (AUC = {res['auc_roc']:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (0.50)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    path = f"{save_dir}/roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_precision_recall(results: list, y_test, save_dir: str = "reports/figures") -> None:
    """Precision-Recall curves for all models."""
    plt.figure(figsize=(10, 7))

    for i, res in enumerate(results):
        precision, recall, _ = precision_recall_curve(y_test, res["y_prob"])
        plt.plot(recall, precision, color=PALETTE[i % len(PALETTE)], lw=2,
                 label=f"{res['name']} (AP = {res['avg_precision']:.4f})")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    path = f"{save_dir}/precision_recall_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_confusion_matrix(y_test, y_pred, model_name: str,
                           save_dir: str = "reports/figures") -> None:
    """Annotated confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churned"],
                yticklabels=["No Churn", "Churned"],
                linewidths=0.5)
    plt.title(f"Confusion Matrix — {model_name}", fontweight="bold", fontsize=13)
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    path = f"{save_dir}/confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_model_comparison(results: list, save_dir: str = "reports/figures") -> None:
    """Grouped bar chart comparing all models across 4 metrics."""
    df_res = pd.DataFrame(results)[["name", "accuracy", "auc_roc", "f1", "avg_precision"]]
    df_res = df_res.set_index("name")

    ax = df_res.plot(kind="bar", figsize=(13, 6), width=0.7, edgecolor="black",
                     color=["#3498db", "#e74c3c", "#2ecc71", "#f39c12"])
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0.5, 1.02)
    ax.legend(["Accuracy", "AUC-ROC", "F1", "Avg Precision"], fontsize=10)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path = f"{save_dir}/model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_feature_importance(model, feature_names: list, model_name: str,
                             top_n: int = 20,
                             save_dir: str = "reports/figures") -> None:
    """Horizontal bar chart of native feature importances."""
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
    elif hasattr(model, "estimators_"):
        # Try to get importances from first base estimator in stacking
        for est in model.estimators_:
            est_obj = est[1] if isinstance(est, tuple) else est
            if hasattr(est_obj, "feature_importances_"):
                fi = est_obj.feature_importances_
                model_name = f"{model_name} (via {type(est_obj).__name__})"
                break
        else:
            logger.warning(f"No feature importances available for {model_name}")
            return
    else:
        logger.warning(f"No feature importances available for {model_name}")
        return

    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": fi})
        .sort_values("importance", ascending=True)
        .tail(top_n)
    )

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(fi_df)))
    plt.figure(figsize=(10, 8))
    plt.barh(fi_df["feature"], fi_df["importance"], color=colors, edgecolor="black", linewidth=0.4)
    plt.title(f"Top {top_n} Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    plt.xlabel("Importance Score", fontsize=12)
    plt.tight_layout()
    path = f"{save_dir}/feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_shap_summary(model, X_test, feature_names: list,
                      n_samples: int = 2000,
                      save_dir: str = "reports/figures") -> None:
    """SHAP beeswarm summary plot."""
    if not HAS_SHAP:
        logger.warning("shap not installed — skipping SHAP analysis. pip install shap")
        return

    logger.info(f"Computing SHAP values on {n_samples} samples...")
    X_sample = X_test[:n_samples]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        plt.figure(figsize=(12, 8))
        shap.summary_plot(sv, X_sample, feature_names=feature_names,
                          max_display=20, show=False, plot_size=(12, 8))
        plt.title("SHAP Feature Impact on Churn Prediction", fontsize=13, fontweight="bold")
        plt.tight_layout()
        path = f"{save_dir}/shap_summary.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {path}")

    except Exception as e:
        logger.warning(f"SHAP failed: {e}")


def plot_optuna_history(study, save_dir: str = "reports/figures") -> None:
    """Optuna optimization history plot."""
    if not HAS_OPTUNA or study is None:
        return
    try:
        ax = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.title("Optuna — XGBoost Tuning History", fontweight="bold")
        plt.tight_layout()
        path = f"{save_dir}/optuna_history.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved: {path}")
    except Exception as e:
        logger.warning(f"Optuna plot failed: {e}")


# ─────────────────────────────────────────────────────────────
# BUSINESS SEGMENTATION
# ─────────────────────────────────────────────────────────────

def churn_risk_segmentation(model, X_test, y_test,
                             config: dict,
                             save_dir: str = "reports/figures") -> pd.DataFrame:
    """
    Segment customers into High / Medium / Low churn risk.
    Simulate business retention impact.
    """
    seg_cfg = config["segmentation"]
    proba = model.predict_proba(X_test)[:, 1]

    seg_df = pd.DataFrame({
        "churn_probability": proba,
        "actual_churn": y_test.values if hasattr(y_test, "values") else y_test,
    })
    seg_df["risk_segment"] = pd.cut(
        proba,
        bins=[0, seg_cfg["medium_risk_threshold"],
              seg_cfg["high_risk_threshold"], 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )

    summary = seg_df.groupby("risk_segment").agg(
        count=("actual_churn", "count"),
        actual_churn_rate=("actual_churn", "mean"),
        avg_prob=("churn_probability", "mean"),
    ).reset_index()

    logger.info("\n" + "=" * 55)
    logger.info("CHURN RISK SEGMENTATION")
    logger.info("=" * 55)
    logger.info(f"\n{summary.to_string(index=False)}")

    # Business impact
    high_risk = seg_df[seg_df["risk_segment"] == "High Risk"]
    churners_caught = high_risk["actual_churn"].sum()
    total_churners = seg_df["actual_churn"].sum()
    coverage = churners_caught / total_churners if total_churners > 0 else 0
    saved = int(churners_caught * seg_cfg["retention_success_rate"])
    reduction = saved / total_churners if total_churners > 0 else 0

    logger.info(f"\nBusiness Impact Simulation:")
    logger.info(f"  Total churners:              {total_churners:,}")
    logger.info(f"  Churners identified (High):  {churners_caught:,} ({coverage:.1%} coverage)")
    logger.info(f"  Customers retained (30%):    {saved:,}")
    logger.info(f"  Churn reduction achieved:    {reduction:.1%}  ← target: ~18%")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    axes[0].pie(summary["count"], labels=summary["risk_segment"].values,
                autopct="%1.1f%%", colors=colors,
                wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[0].set_title("Customer Risk Segmentation", fontweight="bold", fontsize=13)

    bars = axes[1].bar(summary["risk_segment"], summary["actual_churn_rate"],
                       color=colors, edgecolor="black", linewidth=0.8)
    axes[1].set_title("Actual Churn Rate by Segment", fontweight="bold", fontsize=13)
    axes[1].set_ylabel("Churn Rate")
    for bar in bars:
        axes[1].annotate(f"{bar.get_height():.1%}",
                         (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005),
                         ha="center", fontsize=11)

    plt.tight_layout()
    path = f"{save_dir}/churn_segmentation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")

    return seg_df


# ─────────────────────────────────────────────────────────────
# MAIN EVALUATION PIPELINE
# ─────────────────────────────────────────────────────────────

def run_evaluation_pipeline(training_output: dict = None, config: dict = None) -> None:
    """Full evaluation pipeline. Can accept output from train.py."""
    if config is None:
        config = load_config()

    save_dir = config["output"]["figures_dir"]
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if training_output is None:
        logger.info("No training output provided — loading models from disk...")
        import joblib
        from pathlib import Path

        models_dir = Path(config["output"]["models_dir"])
        models = {}
        for pkl in models_dir.glob("*.pkl"):
            if "scaler" not in pkl.name:
                name = pkl.stem.replace("_", " ").title()
                models[name] = joblib.load(pkl)

        scaler = joblib.load(models_dir / config["output"]["scaler_filename"])
        df = pd.read_csv(config["data"]["processed_path"])

        from src.preprocessing import split_and_scale
        (X_train_sc, X_test_sc, _, _, y_train, y_test, feature_names, _) = split_and_scale(df, config)
    else:
        models = training_output["models"]
        X_test_sc = training_output["X_test"]
        y_test = training_output["y_test"]
        feature_names = training_output["feature_names"]
        study = training_output.get("optuna_study")

    # Evaluate all models
    results = evaluate_all_models(models, X_test_sc, y_test)

    # Plots
    plot_roc_curves(results, y_test, save_dir)
    plot_precision_recall(results, y_test, save_dir)
    plot_model_comparison(results, save_dir)

    # Best model details
    best = results[0]
    logger.info(f"\n🏆 Best model: {best['name']} | AUC: {best['auc_roc']:.4f}")
    plot_confusion_matrix(y_test, best["y_pred"], best["name"], save_dir)

    # Feature importance + SHAP (from best tree model)
    for res in results:
        model = models[res["name"]]
        if hasattr(model, "feature_importances_") or hasattr(model, "estimators_"):
            plot_feature_importance(model, feature_names, res["name"], save_dir=save_dir)
            plot_shap_summary(model, X_test_sc, feature_names,
                              config["evaluation"]["shap_sample_size"], save_dir)
            break

    # Optuna history
    if training_output and training_output.get("optuna_study"):
        plot_optuna_history(training_output["optuna_study"], save_dir)

    # Business segmentation (use stacking or best model)
    prod_model = models.get("Stacking Ensemble", models[list(models.keys())[0]])
    churn_risk_segmentation(prod_model, X_test_sc, y_test, config, save_dir)

    # Final summary table
    logger.info("\n" + "=" * 55)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 55)
    summary_df = pd.DataFrame(results)[["name", "accuracy", "auc_roc", "f1", "avg_precision"]]
    summary_df.columns = ["Model", "Accuracy", "AUC-ROC", "F1", "Avg Precision"]
    logger.info(f"\n{summary_df.to_string(index=False)}")
    logger.info("\n✅ Evaluation pipeline complete!")


if __name__ == "__main__":
    run_evaluation_pipeline()
