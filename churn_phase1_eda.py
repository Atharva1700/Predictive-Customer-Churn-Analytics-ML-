"""
============================================================
CUSTOMER CHURN PREDICTION — Phase 1 & 2
Data Generation + EDA + Feature Engineering
============================================================
Run: python churn_phase1_eda.py
Requires: pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted")

# ============================================================
# STEP 1 — SYNTHETIC 200K DATASET GENERATION
# Modeled on real Telco churn patterns
# ============================================================

def generate_churn_dataset(n=200_000):
    """
    Generate a realistic 200K customer churn dataset.
    Churn is correlated with: high monthly charges, low tenure,
    month-to-month contracts, no tech support, fiber optic.
    """
    print(f"Generating {n:,} customer records...")

    # Demographics
    df = pd.DataFrame()
    df['customer_id'] = [f'CUST{str(i).zfill(7)}' for i in range(n)]
    df['gender'] = np.random.choice(['Male', 'Female'], n)
    df['senior_citizen'] = np.random.choice([0, 1], n, p=[0.84, 0.16])
    df['partner'] = np.random.choice(['Yes', 'No'], n, p=[0.48, 0.52])
    df['dependents'] = np.random.choice(['Yes', 'No'], n, p=[0.30, 0.70])

    # Service tenure (months)
    df['tenure'] = np.random.gamma(shape=2.5, scale=14, size=n).clip(1, 72).astype(int)

    # Contract type (key churn driver)
    df['contract'] = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.24, 0.21]
    )

    # Internet service
    df['internet_service'] = np.random.choice(
        ['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22]
    )

    # Add-on services
    df['online_security'] = np.where(
        df['internet_service'] == 'No', 'No internet service',
        np.random.choice(['Yes', 'No'], n, p=[0.29, 0.71])
    )
    df['tech_support'] = np.where(
        df['internet_service'] == 'No', 'No internet service',
        np.random.choice(['Yes', 'No'], n, p=[0.29, 0.71])
    )
    df['online_backup'] = np.where(
        df['internet_service'] == 'No', 'No internet service',
        np.random.choice(['Yes', 'No'], n, p=[0.34, 0.66])
    )
    df['device_protection'] = np.where(
        df['internet_service'] == 'No', 'No internet service',
        np.random.choice(['Yes', 'No'], n, p=[0.34, 0.66])
    )
    df['streaming_tv'] = np.where(
        df['internet_service'] == 'No', 'No internet service',
        np.random.choice(['Yes', 'No'], n, p=[0.38, 0.62])
    )
    df['streaming_movies'] = np.where(
        df['internet_service'] == 'No', 'No internet service',
        np.random.choice(['Yes', 'No'], n, p=[0.39, 0.61])
    )

    # Phone service
    df['phone_service'] = np.random.choice(['Yes', 'No'], n, p=[0.90, 0.10])
    df['multiple_lines'] = np.where(
        df['phone_service'] == 'No', 'No phone service',
        np.random.choice(['Yes', 'No'], n, p=[0.42, 0.58])
    )

    # Billing
    df['paperless_billing'] = np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41])
    df['payment_method'] = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        n, p=[0.34, 0.23, 0.22, 0.21]
    )

    # Monthly charges (fiber is more expensive)
    base_charge = np.where(df['internet_service'] == 'Fiber optic',
                           np.random.normal(80, 20, n),
                           np.where(df['internet_service'] == 'DSL',
                                    np.random.normal(55, 15, n),
                                    np.random.normal(25, 8, n)))
    df['monthly_charges'] = base_charge.clip(18, 120).round(2)
    df['total_charges'] = (df['monthly_charges'] * df['tenure'] *
                           np.random.uniform(0.9, 1.1, n)).round(2)

    # ---- CHURN LABEL (realistic probability model) ----
    churn_prob = np.zeros(n)
    churn_prob += 0.05  # base rate

    # Contract type is strongest predictor
    churn_prob += np.where(df['contract'] == 'Month-to-month', 0.28, 0)
    churn_prob += np.where(df['contract'] == 'One year', 0.05, 0)

    # Tenure (newer customers churn more)
    churn_prob += np.where(df['tenure'] <= 6, 0.20, 0)
    churn_prob += np.where(df['tenure'] <= 12, 0.10, 0)
    churn_prob -= np.where(df['tenure'] >= 48, 0.12, 0)

    # High monthly charges → churn
    churn_prob += np.where(df['monthly_charges'] > 80, 0.12, 0)

    # No support → churn
    churn_prob += np.where(df['tech_support'] == 'No', 0.08, 0)
    churn_prob += np.where(df['online_security'] == 'No', 0.07, 0)

    # Fiber optic customers churn more
    churn_prob += np.where(df['internet_service'] == 'Fiber optic', 0.08, 0)

    # Electronic check → churn
    churn_prob += np.where(df['payment_method'] == 'Electronic check', 0.08, 0)

    # Senior citizens churn slightly more
    churn_prob += df['senior_citizen'] * 0.05

    # Clip and sample
    churn_prob = churn_prob.clip(0, 0.95)
    df['churn'] = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

    print(f"Dataset shape: {df.shape}")
    print(f"Churn rate: {df['churn'].mean():.2%}")
    return df


# ============================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS
# ============================================================

def run_eda(df):
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)

    # Basic info
    print(f"\nShape: {df.shape}")
    print(f"\nChurn distribution:\n{df['churn'].value_counts()}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDtypes summary:\n{df.dtypes.value_counts()}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Customer Churn — EDA Overview', fontsize=16, fontweight='bold')

    # 1. Churn distribution
    churn_counts = df['churn'].value_counts()
    axes[0, 0].pie(churn_counts, labels=['No Churn', 'Churned'],
                   autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'],
                   startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
    axes[0, 0].set_title('Churn Distribution', fontweight='bold')

    # 2. Tenure by churn
    df.groupby('churn')['tenure'].plot.hist(
        ax=axes[0, 1], alpha=0.7, bins=30, legend=True,
        color=['#2ecc71', '#e74c3c']
    )
    axes[0, 1].set_title('Tenure Distribution by Churn', fontweight='bold')
    axes[0, 1].set_xlabel('Tenure (months)')
    axes[0, 1].legend(['No Churn', 'Churned'])

    # 3. Monthly charges by churn
    df[df['churn'] == 0]['monthly_charges'].plot.hist(
        ax=axes[0, 2], alpha=0.6, bins=40, color='#2ecc71', label='No Churn'
    )
    df[df['churn'] == 1]['monthly_charges'].plot.hist(
        ax=axes[0, 2], alpha=0.6, bins=40, color='#e74c3c', label='Churned'
    )
    axes[0, 2].set_title('Monthly Charges by Churn', fontweight='bold')
    axes[0, 2].set_xlabel('Monthly Charges ($)')
    axes[0, 2].legend()

    # 4. Contract type vs churn
    contract_churn = df.groupby('contract')['churn'].mean().sort_values(ascending=False)
    contract_churn.plot(kind='bar', ax=axes[1, 0], color=['#e74c3c', '#f39c12', '#2ecc71'],
                        edgecolor='black', rot=15)
    axes[1, 0].set_title('Churn Rate by Contract Type', fontweight='bold')
    axes[1, 0].set_ylabel('Churn Rate')
    axes[1, 0].set_xlabel('')
    for p in axes[1, 0].patches:
        axes[1, 0].annotate(f'{p.get_height():.1%}',
                            (p.get_x() + p.get_width()/2, p.get_height()),
                            ha='center', va='bottom', fontsize=10)

    # 5. Internet service vs churn
    internet_churn = df.groupby('internet_service')['churn'].mean().sort_values(ascending=False)
    internet_churn.plot(kind='bar', ax=axes[1, 1], color=['#3498db', '#e74c3c', '#2ecc71'],
                        edgecolor='black', rot=0)
    axes[1, 1].set_title('Churn Rate by Internet Service', fontweight='bold')
    axes[1, 1].set_ylabel('Churn Rate')
    axes[1, 1].set_xlabel('')
    for p in axes[1, 1].patches:
        axes[1, 1].annotate(f'{p.get_height():.1%}',
                            (p.get_x() + p.get_width()/2, p.get_height()),
                            ha='center', va='bottom', fontsize=10)

    # 6. Payment method vs churn
    payment_churn = df.groupby('payment_method')['churn'].mean().sort_values(ascending=False)
    payment_churn.plot(kind='barh', ax=axes[1, 2], color='#3498db', edgecolor='black')
    axes[1, 2].set_title('Churn Rate by Payment Method', fontweight='bold')
    axes[1, 2].set_xlabel('Churn Rate')

    plt.tight_layout()
    plt.savefig('eda_overview.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n[Saved] eda_overview.png")

    # Correlation heatmap (numeric only)
    print("\nGenerating correlation heatmap...")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()

    plt.figure(figsize=(10, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[Saved] correlation_heatmap.png")

    return df


# ============================================================
# STEP 3 — FEATURE ENGINEERING
# ============================================================

def feature_engineering(df):
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)

    df = df.copy()

    # ---- Derived features ----

    # Avg revenue per month (handles tenure=0 edge case)
    df['avg_monthly_revenue'] = df['total_charges'] / df['tenure'].replace(0, 1)

    # Service count (how many add-ons subscribed)
    service_cols = ['online_security', 'online_backup', 'device_protection',
                    'tech_support', 'streaming_tv', 'streaming_movies']
    df['num_services'] = sum([
        (df[col] == 'Yes').astype(int) for col in service_cols
    ])

    # Is customer loyal? (tenure > 24 months)
    df['is_loyal'] = (df['tenure'] > 24).astype(int)

    # High value customer flag
    df['is_high_value'] = (df['monthly_charges'] > df['monthly_charges'].quantile(0.75)).astype(int)

    # Has no support services (risk indicator)
    df['no_support_services'] = (
        (df['tech_support'] == 'No') & (df['online_security'] == 'No')
    ).astype(int)

    # Automatic payment (proxy for engagement)
    df['auto_payment'] = df['payment_method'].str.contains('automatic').astype(int)

    # Charge per service unit
    df['charge_per_service'] = df['monthly_charges'] / (df['num_services'] + 1)

    # Tenure bucket
    df['tenure_bucket'] = pd.cut(df['tenure'],
                                 bins=[0, 6, 12, 24, 48, 72],
                                 labels=['0-6m', '6-12m', '12-24m', '24-48m', '48+m'])

    print(f"New features added: avg_monthly_revenue, num_services, is_loyal,")
    print(f"  is_high_value, no_support_services, auto_payment, charge_per_service, tenure_bucket")
    print(f"Dataset shape after FE: {df.shape}")

    # ---- Encoding ----
    print("\nEncoding categorical features...")

    # Binary encoding
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    for col in ['gender', 'partner', 'dependents', 'phone_service', 'paperless_billing']:
        df[col] = df[col].map(binary_map)

    # Ordinal encoding for contract
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    df['contract'] = df['contract'].map(contract_map)

    # One-hot encoding for nominal
    df = pd.get_dummies(df, columns=[
        'internet_service', 'payment_method', 'multiple_lines',
        'online_security', 'online_backup', 'device_protection',
        'tech_support', 'streaming_tv', 'streaming_movies',
        'tenure_bucket'
    ], drop_first=False)

    # Drop ID column
    df.drop(columns=['customer_id'], inplace=True)

    # Convert boolean columns to int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"Final feature count: {df.shape[1] - 1} features + 1 target")
    print(f"Final dataset shape: {df.shape}")

    return df


# ============================================================
# STEP 4 — TRAIN/TEST SPLIT + CLASS IMBALANCE HANDLING
# ============================================================

def prepare_data(df):
    print("\n" + "="*60)
    print("TRAIN/TEST SPLIT & IMBALANCE HANDLING")
    print("="*60)

    X = df.drop(columns=['churn'])
    y = df['churn']

    print(f"Feature matrix: {X.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    print(f"Class imbalance ratio: {y.value_counts()[0]/y.value_counts()[1]:.1f}:1")

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTrain size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")
    print(f"Train churn rate: {y_train.mean():.2%} | Test churn rate: {y_test.mean():.2%}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply SMOTE on training set to balance classes
    try:
        from imblearn.over_sampling import SMOTE
        print("\nApplying SMOTE to balance training set...")
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        print(f"After SMOTE — Train size: {X_train_resampled.shape[0]:,}")
        print(f"After SMOTE — Class balance: {pd.Series(y_train_resampled).value_counts().to_dict()}")
    except ImportError:
        print("\n[!] imblearn not found. Using class_weight='balanced' in models instead.")
        print("    Install with: pip install imbalanced-learn")
        X_train_resampled = X_train_scaled
        y_train_resampled = y_train

    return (X_train_scaled, X_test_scaled, X_train_resampled,
            y_train, y_test, y_train_resampled, X.columns.tolist(), scaler)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Generate dataset
    df_raw = generate_churn_dataset(n=200_000)

    # Save raw data
    df_raw.to_csv('churn_raw_200k.csv', index=False)
    print("\n[Saved] churn_raw_200k.csv")

    # EDA
    df_eda = run_eda(df_raw)

    # Feature engineering
    df_processed = feature_engineering(df_eda)

    # Save processed data
    df_processed.to_csv('churn_processed.csv', index=False)
    print("\n[Saved] churn_processed.csv")

    # Prepare splits
    (X_train, X_test, X_train_resampled,
     y_train, y_test, y_train_resampled,
     feature_names, scaler) = prepare_data(df_processed)

    print("\n✅ Phase 1 & 2 Complete!")
    print("   → Run churn_phase2_models.py for model training")
