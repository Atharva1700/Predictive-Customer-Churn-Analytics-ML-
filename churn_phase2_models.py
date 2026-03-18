"""
============================================================
CUSTOMER CHURN PREDICTION — Phase 3 & 4
Ensemble Model Building + Evaluation + Hyperparameter Tuning
============================================================
Run AFTER churn_phase1_eda.py:
  python churn_phase2_models.py

Requires:
  pip install scikit-learn xgboost lightgbm imbalanced-learn shap optuna
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, classification_report,
    confusion_matrix, precision_recall_curve, average_precision_score, f1_score
)
import joblib

# Optional heavy libraries — gracefully degrade if missing
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[!] xgboost not installed. Will use GradientBoostingClassifier instead.")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[!] lightgbm not installed. Ensemble will use available models.")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[!] imbalanced-learn not installed. Using class_weight='balanced'.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[!] shap not installed. Skipping SHAP explanations.")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("[!] optuna not installed. Will skip hyperparameter tuning demo.")

np.random.seed(42)
sns.set_theme(style="whitegrid")


# ============================================================
# LOAD & PREPARE DATA
# ============================================================

def load_and_prepare():
    """Load the processed dataset from Phase 1."""
    print("Loading processed dataset...")
    try:
        df = pd.read_csv('churn_processed.csv')
    except FileNotFoundError:
        print("[!] churn_processed.csv not found. Regenerating from scratch...")
        # Inline mini dataset generation for standalone use
        from churn_phase1_eda import generate_churn_dataset, feature_engineering
        df = generate_churn_dataset(n=200_000)
        df = feature_engineering(df)

    X = df.drop(columns=['churn'])
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    if HAS_SMOTE:
        smote = SMOTE(random_state=42)
        X_train_sc, y_train = smote.fit_resample(X_train_sc, y_train)
        print(f"After SMOTE: {X_train_sc.shape[0]:,} training samples")

    print(f"Train: {X_train_sc.shape} | Test: {X_test_sc.shape}")
    return X_train_sc, X_test_sc, y_train, y_test, X.columns.tolist(), scaler


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_base_models():
    """Return well-tuned base models for the ensemble."""

    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            C=0.5,
            class_weight='balanced',
            max_iter=1000,
            solver='saga',
            random_state=42
        ),
    }

    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=3,   # handles imbalance
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    else:
        models['Gradient Boosting'] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42
        )

    if HAS_LGB:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

    return models


# ============================================================
# TRAINING & EVALUATION UTILITIES
# ============================================================

def evaluate_model(model, X_test, y_test, name="Model"):
    """Compute and print full evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_prob)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  AUC-ROC:           {auc:.4f}")
    print(f"  F1 Score:          {f1:.4f}")
    print(f"  Avg Precision:     {ap:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churned']))

    return {'name': name, 'acc': acc, 'auc': auc, 'f1': f1, 'ap': ap,
            'y_prob': y_prob, 'y_pred': y_pred}


def plot_roc_curves(results, y_test):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 7))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    for i, res in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                 label=f"{res['name']} (AUC = {res['auc']:.4f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC = 0.5)')
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.title('ROC Curves — All Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[Saved] roc_curves.png")


def plot_confusion_matrix(y_test, y_pred, model_name):
    """Heatmap confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churned'],
                yticklabels=['No Churn', 'Churned'],
                linewidths=0.5, linecolor='black')
    plt.title(f'Confusion Matrix — {model_name}', fontweight='bold', fontsize=13)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix_ensemble.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[Saved] confusion_matrix_ensemble.png")


def plot_precision_recall(results, y_test):
    """Precision-Recall curves."""
    plt.figure(figsize=(10, 7))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

    for i, res in enumerate(results):
        precision, recall, _ = precision_recall_curve(y_test, res['y_prob'])
        ap = res['ap']
        plt.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                 label=f"{res['name']} (AP = {ap:.4f})")

    plt.xlabel('Recall', fontsize=13)
    plt.ylabel('Precision', fontsize=13)
    plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig('precision_recall_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[Saved] precision_recall_curves.png")


def plot_model_comparison(results):
    """Bar chart comparing all models."""
    df_res = pd.DataFrame(results)[['name', 'acc', 'auc', 'f1', 'ap']]
    df_res = df_res.set_index('name')

    ax = df_res.plot(kind='bar', figsize=(12, 6), width=0.7, edgecolor='black',
                     color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('')
    ax.set_ylim(0.5, 1.0)
    ax.legend(['Accuracy', 'AUC-ROC', 'F1', 'Avg Precision'], fontsize=11)
    plt.xticks(rotation=15, ha='right')

    # Annotate best AUC
    for patch in ax.patches:
        ax.annotate(f'{patch.get_height():.3f}',
                    (patch.get_x() + patch.get_width()/2, patch.get_height() + 0.002),
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[Saved] model_comparison.png")


# ============================================================
# ENSEMBLE: STACKING CLASSIFIER
# ============================================================

def build_stacking_ensemble(base_models_dict, X_train, y_train, X_test, y_test):
    """
    Build a Stacking Classifier:
    - Level-0: all base models
    - Level-1 (meta-learner): Logistic Regression
    """
    print("\n" + "="*60)
    print("STACKING ENSEMBLE")
    print("="*60)

    # Exclude LogReg from base (it's the meta-learner)
    estimators = [
        (name.lower().replace(' ', '_'), model)
        for name, model in base_models_dict.items()
        if 'Logistic' not in name
    ]

    meta_learner = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')

    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1,
        passthrough=False
    )

    print(f"Base models: {[e[0] for e in estimators]}")
    print("Meta-learner: Logistic Regression")
    print("\nFitting stacking ensemble (this may take a few minutes)...")

    stacking.fit(X_train, y_train)
    result = evaluate_model(stacking, X_test, y_test, name="Stacking Ensemble")

    return stacking, result


# ============================================================
# SOFT VOTING ENSEMBLE
# ============================================================

def build_voting_ensemble(base_models_dict, X_train, y_train, X_test, y_test):
    """Soft Voting Ensemble — averages predicted probabilities."""
    print("\n" + "="*60)
    print("SOFT VOTING ENSEMBLE")
    print("="*60)

    estimators = [(name.lower().replace(' ', '_'), model)
                  for name, model in base_models_dict.items()]

    voting = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    voting.fit(X_train, y_train)
    result = evaluate_model(voting, X_test, y_test, name="Soft Voting Ensemble")

    return voting, result


# ============================================================
# FEATURE IMPORTANCE (SHAP + NATIVE)
# ============================================================

def plot_feature_importance(model, model_name, feature_names, X_test, top_n=20):
    """Plot native feature importance for tree-based models."""

    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
    elif hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'feature_importances_'):
        fi = model.estimators_[0].feature_importances_
    else:
        print(f"[!] No native feature importances for {model_name}")
        return

    fi_df = pd.DataFrame({'feature': feature_names, 'importance': fi})
    fi_df = fi_df.sort_values('importance', ascending=True).tail(top_n)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(fi_df)))
    plt.barh(fi_df['feature'], fi_df['importance'], color=colors, edgecolor='black', linewidth=0.5)
    plt.title(f'Top {top_n} Feature Importances — {model_name}', fontsize=13, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[Saved] feature_importance.png")


def run_shap_analysis(model, X_test, feature_names, n_samples=2000):
    """SHAP value analysis for model explainability."""
    if not HAS_SHAP:
        print("[!] SHAP not available. Skipping.")
        return

    print("\nRunning SHAP analysis (sample of 2,000 records)...")
    X_sample = X_test[:n_samples]

    try:
        # TreeExplainer works for RF, XGB, LGBM
        if HAS_XGB or HAS_LGB or hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_sample[:100])

        shap_values = explainer.shap_values(X_sample)

        # For binary classification, shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        plt.figure()
        shap.summary_plot(sv, X_sample, feature_names=feature_names,
                          max_display=20, show=False, plot_size=(12, 8))
        plt.title('SHAP Summary — Feature Impact on Churn', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("[Saved] shap_summary.png")

    except Exception as e:
        print(f"[!] SHAP failed: {e}")


# ============================================================
# OPTUNA HYPERPARAMETER TUNING
# ============================================================

def optuna_tune_xgb(X_train, y_train, n_trials=30):
    """Tune XGBoost with Optuna. Returns best params."""
    if not HAS_OPTUNA or not HAS_XGB:
        print("[!] Optuna or XGBoost not available. Skipping tuning.")
        return {}

    print("\n" + "="*60)
    print(f"OPTUNA HYPERPARAMETER TUNING (XGBoost, {n_trials} trials)")
    print("="*60)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 5),
            'random_state': 42,
            'n_jobs': -1,
            'use_label_encoder': False,
            'eval_metric': 'logloss',
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    print(f"\nBest AUC-ROC: {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    # Plot optimization history
    try:
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.title("Optuna Optimization History", fontweight='bold')
        plt.tight_layout()
        plt.savefig('optuna_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("[Saved] optuna_history.png")
    except Exception:
        pass

    return best_params


# ============================================================
# BUSINESS IMPACT: CHURN SEGMENTATION
# ============================================================

def churn_segmentation(model, X_test, y_test):
    """
    Translate model scores into actionable business segments:
    High / Medium / Low churn risk.
    Estimate retention intervention impact.
    """
    print("\n" + "="*60)
    print("BUSINESS IMPACT — CHURN RISK SEGMENTATION")
    print("="*60)

    proba = model.predict_proba(X_test)[:, 1]

    seg_df = pd.DataFrame({
        'churn_probability': proba,
        'actual_churn': y_test.values
    })

    seg_df['risk_segment'] = pd.cut(
        proba,
        bins=[0, 0.30, 0.60, 1.0],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )

    summary = seg_df.groupby('risk_segment').agg(
        count=('actual_churn', 'count'),
        actual_churn_rate=('actual_churn', 'mean'),
        avg_churn_prob=('churn_probability', 'mean')
    ).reset_index()

    print(f"\nCustomer Risk Segments (Test Set):")
    print(summary.to_string(index=False))

    # Simulate 18% churn reduction
    high_risk = seg_df[seg_df['risk_segment'] == 'High Risk']
    churners_identified = high_risk['actual_churn'].sum()
    total_churners = seg_df['actual_churn'].sum()
    coverage = churners_identified / total_churners if total_churners > 0 else 0

    print(f"\nSimulated Retention Impact:")
    print(f"  Total churners in test set:        {total_churners:,}")
    print(f"  Churners identified (High Risk):   {churners_identified:,} ({coverage:.1%} coverage)")

    # Assume 30% of targeted high-risk customers can be retained
    saved = int(churners_identified * 0.30)
    reduction_pct = saved / total_churners if total_churners > 0 else 0
    print(f"  Customers retained (30% success):  {saved:,}")
    print(f"  Churn reduction:                   {reduction_pct:.1%}  ← target: ~18%")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Segment pie chart
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    axes[0].pie(summary['count'], labels=summary['risk_segment'].values,
                autopct='%1.1f%%', colors=colors,
                wedgeprops=dict(edgecolor='white', linewidth=2))
    axes[0].set_title('Customer Risk Segmentation', fontweight='bold', fontsize=13)

    # Churn rate by segment
    axes[1].bar(summary['risk_segment'], summary['actual_churn_rate'],
                color=colors, edgecolor='black', linewidth=0.8)
    axes[1].set_title('Actual Churn Rate by Segment', fontweight='bold', fontsize=13)
    axes[1].set_ylabel('Churn Rate')
    for p in axes[1].patches:
        axes[1].annotate(f'{p.get_height():.1%}',
                         (p.get_x() + p.get_width()/2, p.get_height() + 0.005),
                         ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig('churn_segmentation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[Saved] churn_segmentation.png")

    return seg_df


# ============================================================
# MAIN PIPELINE
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("CUSTOMER CHURN PREDICTION — MODEL PIPELINE")
    print("="*60)

    # Load data
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_prepare()

    # ─── STEP 1: Train base models ───────────────────────────
    print("\n" + "="*60)
    print("TRAINING BASE MODELS")
    print("="*60)

    base_models = get_base_models()
    all_results = []

    for name, model in base_models.items():
        print(f"\nFitting {name}...")
        model.fit(X_train, y_train)
        result = evaluate_model(model, X_test, y_test, name=name)
        all_results.append(result)

    # ─── STEP 2: Soft Voting Ensemble ────────────────────────
    voting_model, voting_result = build_voting_ensemble(
        base_models, X_train, y_train, X_test, y_test
    )
    all_results.append(voting_result)

    # ─── STEP 3: Stacking Ensemble ───────────────────────────
    stacking_model, stacking_result = build_stacking_ensemble(
        base_models, X_train, y_train, X_test, y_test
    )
    all_results.append(stacking_result)

    # ─── STEP 4: Optional Optuna Tuning ──────────────────────
    if HAS_OPTUNA and HAS_XGB:
        best_params = optuna_tune_xgb(X_train, y_train, n_trials=30)
        if best_params:
            print("\nTraining tuned XGBoost with Optuna best params...")
            tuned_xgb = XGBClassifier(**best_params, use_label_encoder=False,
                                      eval_metric='logloss', n_jobs=-1)
            tuned_xgb.fit(X_train, y_train)
            result = evaluate_model(tuned_xgb, X_test, y_test, name="Tuned XGBoost")
            all_results.append(result)

    # ─── STEP 5: Visualizations ──────────────────────────────
    print("\nGenerating evaluation plots...")
    plot_roc_curves(all_results, y_test)
    plot_model_comparison(all_results)
    plot_precision_recall(all_results, y_test)

    # Best model = highest AUC
    best_result = max(all_results, key=lambda r: r['auc'])
    print(f"\n🏆 Best model: {best_result['name']} | AUC: {best_result['auc']:.4f}")

    # Confusion matrix for best model
    plot_confusion_matrix(y_test, best_result['y_pred'], best_result['name'])

    # ─── STEP 6: Feature Importance ──────────────────────────
    # Get best sklearn estimator
    best_sklearn_model = None
    for name, model in base_models.items():
        if hasattr(model, 'feature_importances_') and (
            best_sklearn_model is None or
            evaluate_model(model, X_test, y_test, name)['auc'] >
            evaluate_model(best_sklearn_model[1], X_test, y_test, best_sklearn_model[0])['auc']
        ):
            best_sklearn_model = (name, model)

    if best_sklearn_model:
        name, model = best_sklearn_model
        plot_feature_importance(model, name, feature_names, X_test)
        run_shap_analysis(model, X_test, feature_names)

    # ─── STEP 7: Business Impact ─────────────────────────────
    # Get the actual best model object for segmentation
    # Use stacking as final production model
    seg_df = churn_segmentation(stacking_model, X_test, y_test)

    # ─── STEP 8: Save models ─────────────────────────────────
    print("\nSaving models...")
    joblib.dump(stacking_model, 'churn_stacking_model.pkl')
    joblib.dump(voting_model, 'churn_voting_model.pkl')
    joblib.dump(scaler, 'churn_scaler.pkl')

    # Summary table
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    summary_df = pd.DataFrame(all_results)[['name', 'acc', 'auc', 'f1', 'ap']]
    summary_df.columns = ['Model', 'Accuracy', 'AUC-ROC', 'F1', 'Avg Precision']
    summary_df = summary_df.sort_values('AUC-ROC', ascending=False)
    print(summary_df.to_string(index=False))

    print("\n✅ Phase 3 & 4 Complete!")
    print("   Artifacts: roc_curves.png, model_comparison.png,")
    print("              confusion_matrix_ensemble.png, feature_importance.png,")
    print("              churn_segmentation.png, churn_stacking_model.pkl")
