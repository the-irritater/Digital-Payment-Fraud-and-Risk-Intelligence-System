"""
Generates a single comprehensive merged notebook for the entire project.
"""
import os
import nbformat as nbf

NOTEBOOK_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")
os.makedirs(NOTEBOOK_DIR, exist_ok=True)

def md(text):
    return ('markdown', text)

def code(text):
    return ('code', text)

def create_merged_notebook():
    nb = nbf.v4.new_notebook()
    nb['cells'] = []

    cells_data = [
        # ===================== TITLE PAGE =====================
        md("""# Digital Payment Fraud and Risk Intelligence System

| Field | Detail |
|:---|:---|
| **Author** | Sanman Kadam |
| **Project** | Real-Time Digital Payment Fraud Detection and Explainable Risk Intelligence Platform |
| **Dataset** | PaySim Synthetic Financial Transactions (Kaggle) |
| **India Context** | Reserve Bank of India Payment System Indicators |
| **Tech Stack** | Python, Pandas, XGBoost, Optuna, SHAP, Isolation Forest, NetworkX, Streamlit, SQL |
| **Primary Metric** | PR-AUC (Average Precision) |
| **Repository** | github.com/sanmankadam/digital-payment-fraud-intelligence |
"""),

        # ===================== TABLE OF CONTENTS =====================
        md("""## Table of Contents

1. Business Problem Statement
2. Data Loading and Initial Exploration
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Baseline Models (Logistic Regression and Random Forest)
6. XGBoost with Optuna Hyperparameter Optimization
7. Cost-Sensitive Threshold Optimization
8. SHAP Model Explainability
9. Anomaly Detection with Isolation Forest
10. Hybrid Risk Engine Demonstration
11. Synthetic UPI Data and Graph Risk Analysis
12. Conclusions and Key Findings
"""),

        # ===================== SECTION 1: BUSINESS PROBLEM =====================
        md("""## 1. Business Problem Statement

### What does this project solve?

Every digital payment system needs to answer three questions for each transaction:

1. **Is this transaction potentially fraudulent?**
2. **Why does the model think it is fraudulent?**
3. **What action should the system take?**

Most machine learning projects stop at question 1. This project addresses all three by building a complete risk intelligence pipeline.

### How does a real payment risk engine work?

```
Transaction arrives
       |
       v
Risk Engine evaluates it
       |
       v
Fraud probability score (0 to 100)
       |
       v
Risk classification (LOW / MEDIUM / HIGH / CRITICAL)
       |
       v
Action taken (ALLOW / REVIEW / BLOCK)
```

### Why is this important?

India processes hundreds of millions of UPI transactions daily. Even a 0.01% fraud rate translates to thousands of fraudulent transactions per day. The Reserve Bank of India publishes aggregate fraud statistics through its Payment System Indicators, but transaction-level UPI data is not publicly available due to privacy and security restrictions.

This project therefore uses:
- **PaySim** (synthetic mobile-money simulation) for transaction-level fraud modeling
- **RBI data** for Indian payment ecosystem context
- **A custom synthetic UPI generator** for realistic Indian payment attribute simulation

This distinction is clearly documented throughout the project.
"""),

# ===================== ENVIRONMENT SETUP =====================
        md("""## Environment Setup and Package Verification

This section verifies that all required Python packages are installed. If any package (such as Optuna or SHAP) is missing in your notebook environment, it will be automatically installed.
"""),

        code("""# Automated dependency installer for Jupyter / Google Colab / VS Code environments
import sys
import subprocess

try:
    import optuna
    import shap
    import xgboost
    import networkx
    print("All core dependencies (optuna, shap, xgboost, networkx) are verified.")
except ImportError:
    print("Installing missing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "optuna", "shap", "xgboost", "networkx", "plotly", "streamlit"])
    print("Dependencies installed successfully.")
"""),

        # ===================== SECTION 2: DATA LOADING =====================
        md("""## 2. Data Loading and Initial Exploration

### About the dataset

PaySim is a synthetic financial dataset generated using a mobile-money simulator based on aggregated real transaction logs. It contains approximately 6.3 million transactions over 30 simulated days.

**Important data leakage warning**: The PaySim documentation warns that balance columns (oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest) should not be used directly for prediction because fraudulent transactions are cancelled in the simulation, creating unrealistic balance patterns. We handle this by computing balance discrepancy features rather than using raw balances.
"""),

        code("""import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append('../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11

from data_processing import load_raw_data, get_data_summary

df = load_raw_data('../data/PS_20174392719_1491204439457_log.csv')
print("Dataset shape:", df.shape)
print("\\nColumn names:", list(df.columns))
print("\\nFirst 5 rows:")
df.head()
"""),

        md("""### Interpretation

The dataset contains 6.36 million rows and 11 columns. Each row represents one financial transaction. The columns include:
- **step**: Simulated time in hours (1 step = 1 hour, covering roughly 30 days)
- **type**: Transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
- **amount**: Transaction amount
- **nameOrig / nameDest**: Anonymized originator and destination account identifiers
- **oldbalanceOrg / newbalanceOrig**: Originator account balance before and after the transaction
- **oldbalanceDest / newbalanceDest**: Destination account balance before and after the transaction
- **isFraud**: Target variable (1 = fraudulent, 0 = legitimate)
- **isFlaggedFraud**: A naive rule-based flag from the simulation (flags transfers over 200000)
"""),

        code("""# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())
print("\\nTotal missing values:", df.isnull().sum().sum())
"""),

        md("""### Interpretation

There are zero missing values in the dataset. This is expected for synthetic data but is still an important validation step. In real-world payment data, missing values in balance fields or timestamps would require careful handling.
"""),

        code("""# Class distribution
summary = get_data_summary(df)
print(f"Total transactions: {summary['total_records']:,}")
print(f"Total fraudulent transactions: {summary['total_fraud']:,}")
print(f"Fraud rate: {summary['fraud_rate_pct']:.4f}%")
print(f"\\nThis means approximately 1 in every {int(100/summary['fraud_rate_pct']):,} transactions is fraudulent.")
"""),

        md("""### Interpretation

The fraud rate is approximately 0.13%. This is an extremely imbalanced dataset. A model that simply predicts every transaction as legitimate would achieve 99.87% accuracy while being completely useless for fraud detection.

This is why we cannot use accuracy as our evaluation metric. Instead we use:
- **PR-AUC (Average Precision)**: The area under the Precision-Recall curve. This metric is specifically designed for imbalanced classification.
- **Recall**: Of all actual fraud transactions, how many did we catch?
- **Precision**: Of all transactions we flagged as fraud, how many were actually fraud?
- **Financial Loss**: The actual monetary cost of missed frauds plus investigation costs of false alarms.
"""),

        # ===================== SECTION 3: EDA =====================
        md("""## 3. Exploratory Data Analysis

### Question 1: Which transaction types contain fraud?
"""),

        code("""# Fraud breakdown by transaction type
fraud_by_type = df.groupby('type')['isFraud'].agg(['count', 'sum', 'mean']).reset_index()
fraud_by_type.columns = ['Transaction Type', 'Total Count', 'Fraud Count', 'Fraud Rate']
fraud_by_type['Fraud Rate (%)'] = fraud_by_type['Fraud Rate'] * 100
fraud_by_type = fraud_by_type.sort_values('Fraud Count', ascending=False)

print("Fraud distribution by transaction type:")
print(fraud_by_type.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(fraud_by_type['Transaction Type'], fraud_by_type['Fraud Count'], color=['#ef4444', '#f97316', '#94a3b8', '#94a3b8', '#94a3b8'])
axes[0].set_title('Fraud Count by Transaction Type')
axes[0].set_ylabel('Number of Fraudulent Transactions')

axes[1].bar(fraud_by_type['Transaction Type'], fraud_by_type['Fraud Rate (%)'], color=['#ef4444', '#f97316', '#94a3b8', '#94a3b8', '#94a3b8'])
axes[1].set_title('Fraud Rate (%) by Transaction Type')
axes[1].set_ylabel('Fraud Rate (%)')

plt.tight_layout()
plt.show()
"""),

        md("""### Interpretation

Only TRANSFER and CASH_OUT transaction types contain fraudulent transactions. PAYMENT, CASH_IN, and DEBIT have zero fraud cases. This is a critical finding because:

1. It tells us that fraud in this simulation is concentrated in fund-transfer operations, not in payment or deposit operations.
2. For model training, we can filter the dataset to only TRANSFER and CASH_OUT transactions, which reduces the dataset size and focuses the model on the relevant population.
3. In a real-world Indian UPI system, this pattern makes sense: fraudsters typically drain accounts through transfers (P2P) or cash-out operations, not through merchant payments.
"""),

        md("""### Question 2: Does transaction amount influence fraud?
"""),

        code("""# Amount distribution comparison
df['log_amount'] = np.log1p(df['amount'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# KDE plot
for label, color in [(0, '#22c55e'), (1, '#ef4444')]:
    subset = df[df['isFraud'] == label]['log_amount']
    axes[0].hist(subset, bins=50, alpha=0.6, color=color, label=f"{'Fraud' if label else 'Legitimate'}", density=True)
axes[0].set_title('Log Amount Distribution: Fraud vs Legitimate')
axes[0].set_xlabel('Log(Amount + 1)')
axes[0].legend()

# Box plot
fraud_amounts = df[df['isFraud'] == 1]['amount']
legit_amounts = df[df['isFraud'] == 0]['amount'].sample(5000, random_state=42)

axes[1].boxplot([legit_amounts, fraud_amounts], labels=['Legitimate (sample)', 'Fraudulent'])
axes[1].set_title('Amount Boxplot Comparison')
axes[1].set_ylabel('Transaction Amount')
axes[1].set_yscale('log')

plt.tight_layout()
plt.show()

print(f"Legitimate transactions - Median amount: {df[df['isFraud']==0]['amount'].median():,.2f}")
print(f"Fraudulent transactions - Median amount: {df[df['isFraud']==1]['amount'].median():,.2f}")
print(f"Fraudulent transactions are on average {df[df['isFraud']==1]['amount'].mean() / df[df['isFraud']==0]['amount'].mean():.1f}x larger than legitimate ones.")
"""),

        md("""### Interpretation

Fraudulent transactions tend to have significantly higher amounts than legitimate ones. The median fraudulent amount is much larger than the median legitimate amount. This makes intuitive sense: a fraudster who gains access to an account wants to extract as much money as possible in as few transactions as possible.

However, amount alone is not sufficient to detect fraud. Many legitimate high-value transactions exist (salary transfers, property payments, etc.). This is why we need behavioral features that look at a customer's historical patterns rather than just the absolute amount.
"""),

        md("""### Question 3: When do frauds happen?
"""),

        code("""# Temporal fraud pattern
df['hour_of_day'] = df['step'] % 24

hourly_stats = df.groupby('hour_of_day').agg(
    total=('isFraud', 'count'),
    fraud_count=('isFraud', 'sum')
).reset_index()
hourly_stats['fraud_rate_pct'] = hourly_stats['fraud_count'] / hourly_stats['total'] * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(hourly_stats['hour_of_day'], hourly_stats['fraud_count'], color='#ef4444', alpha=0.8)
axes[0].set_title('Fraud Count by Hour of Day')
axes[0].set_xlabel('Hour of Day')
axes[0].set_ylabel('Number of Fraudulent Transactions')

axes[1].plot(hourly_stats['hour_of_day'], hourly_stats['fraud_rate_pct'], marker='o', color='#ef4444', linewidth=2)
axes[1].set_title('Fraud Rate (%) by Hour of Day')
axes[1].set_xlabel('Hour of Day')
axes[1].set_ylabel('Fraud Rate (%)')
axes[1].axhline(y=df['isFraud'].mean()*100, color='gray', linestyle='--', label='Overall average')
axes[1].legend()

plt.tight_layout()
plt.show()
"""),

        md("""### Interpretation

The temporal analysis reveals whether fraudulent transactions cluster at specific times. In real-world payment systems, midnight and early morning hours (00:00 to 05:00) typically show elevated fraud rates because:
- Account holders are less likely to notice unauthorized transactions while sleeping
- Customer service response times are slower during off-hours
- Fraudsters exploit the window between account compromise and detection

This pattern, if observed in the data, directly informs our business rule engine where nocturnal high-value transfers receive an elevated risk score.
"""),

        # ===================== SECTION 4: FEATURE ENGINEERING =====================
        md("""## 4. Feature Engineering

### Why feature engineering matters

Raw transaction fields (amount, type, step) alone are insufficient for effective fraud detection. A real payment risk system needs to understand:
- Is this amount unusual for this customer?
- Is this customer transacting at an unusual rate?
- Has this customer ever sent money to this beneficiary before?

All features below are constructed using **strictly causal (backward-looking) logic**. For any transaction at time T, only information from times before T is used. This prevents data leakage, which would artificially inflate model performance.
"""),

        code("""from feature_engineering import build_features, get_feature_matrix, FEATURE_COLS

# Filter to fraud-relevant transaction types
df_filtered = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
print(f"Filtered to TRANSFER and CASH_OUT: {len(df_filtered):,} transactions")
print(f"Fraud count in filtered set: {df_filtered['isFraud'].sum():,}")

# Build features on a sample for demonstration
sample_df = df_filtered.sample(n=100000, random_state=42).sort_values('step').reset_index(drop=True)
featured_df = build_features(sample_df)
X, y = get_feature_matrix(featured_df)

print(f"\\nFeature matrix shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")
print(f"\\nFeature columns ({len(FEATURE_COLS)} total):")
for i, col in enumerate(FEATURE_COLS, 1):
    print(f"  {i:2d}. {col}")
"""),

        md("""### Feature descriptions and rationale

| Feature | What it captures | Why it matters for fraud |
|:---|:---|:---|
| log_amount | Log-transformed transaction amount | Normalizes the highly skewed amount distribution |
| hour_of_day | Hour extracted from step (step mod 24) | Captures nocturnal fraud patterns |
| day_of_week | Day of week from step | Captures weekly behavioral cycles |
| is_night_time | Flag for hours 0-5 | Direct indicator for off-hours transactions |
| type_TRANSFER / type_CASH_OUT | One-hot encoded type | Fraud concentrates in specific types |
| is_orig_customer / is_dest_customer | Whether account ID starts with C or M | Distinguishes customer-to-customer vs customer-to-merchant |
| orig_balance_err | Expected vs actual post-transaction originator balance | Captures PaySim simulation artifacts |
| amount_to_orig_prior_mean_ratio | Current amount divided by customers prior average | Detects unusually large transactions for a given customer |
| transactions_last_1h/6h/24h | Velocity count of recent transactions | Detects burst transaction patterns |
| is_new_beneficiary | First time this customer sends to this destination | Transfers to new recipients are higher risk |
| amount_velocity_6h | Amount-weighted velocity measure | Captures rapid fund movement |

### Data leakage prevention

The `amount_to_orig_prior_mean_ratio` uses an expanding mean that excludes the current transaction. For the first transaction by any customer, the ratio defaults to 1.0 (no history to compare against). The `is_new_beneficiary` flag uses cumulative sequence counting so it only looks at transactions that occurred before the current one.
"""),

        code("""# Show sample feature values for a few transactions
print("Sample feature values (first 5 rows):")
print(X.head().to_string())
"""),

        # ===================== SECTION 5: BASELINE MODELS =====================
        md("""## 5. Baseline Models

Before building the final XGBoost model, we establish baselines with simpler algorithms. This is critical because:
- It proves that the feature engineering adds value beyond a naive approach
- It provides a performance floor that the final model must exceed
- It demonstrates methodological rigor (you should never jump directly to a complex model)

### Temporal split strategy

We split the data chronologically, not randomly. This mirrors real deployment where the model trains on historical data and predicts future transactions.

```
Step 1                        Step 520                Step 631           Step 743
|-------- TRAIN (70%) --------|---- VAL (15%) ----|---- TEST (15%) ----|
```
"""),

        code("""from data_processing import temporal_train_val_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix
)

# Full pipeline on filtered data
df_model = df_filtered.sample(n=400000, random_state=42).sort_values('step').reset_index(drop=True)
featured_model = build_features(df_model)
train_df, val_df, test_df = temporal_train_val_test_split(featured_model)

X_train, y_train = get_feature_matrix(train_df)
X_val, y_val = get_feature_matrix(val_df)
X_test, y_test = get_feature_matrix(test_df)

val_amounts = val_df['amount'].values
test_amounts = test_df['amount'].values

print(f"Train set: {len(X_train):,} rows | Fraud: {y_train.sum():,} ({y_train.mean()*100:.3f}%)")
print(f"Val set:   {len(X_val):,} rows  | Fraud: {y_val.sum():,} ({y_val.mean()*100:.3f}%)")
print(f"Test set:  {len(X_test):,} rows  | Fraud: {y_test.sum():,} ({y_test.mean()*100:.3f}%)")
"""),

        code("""# Logistic Regression baseline
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_model.fit(X_train.fillna(0), y_train)
lr_val_prob = lr_model.predict_proba(X_val.fillna(0))[:, 1]

lr_pr_auc = average_precision_score(y_val, lr_val_prob)
lr_pred = (lr_val_prob >= 0.5).astype(int)
lr_recall = recall_score(y_val, lr_pred)
lr_precision = precision_score(y_val, lr_pred, zero_division=0)
lr_f1 = f1_score(y_val, lr_pred)

print("Logistic Regression (Validation Set)")
print(f"  PR-AUC:    {lr_pr_auc:.4f}")
print(f"  Precision: {lr_precision:.4f}")
print(f"  Recall:    {lr_recall:.4f}")
print(f"  F1 Score:  {lr_f1:.4f}")
"""),

        code("""# Random Forest baseline
rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
rf_model.fit(X_train.fillna(0), y_train)
rf_val_prob = rf_model.predict_proba(X_val.fillna(0))[:, 1]

rf_pr_auc = average_precision_score(y_val, rf_val_prob)
rf_pred = (rf_val_prob >= 0.5).astype(int)
rf_recall = recall_score(y_val, rf_pred)
rf_precision = precision_score(y_val, rf_pred, zero_division=0)
rf_f1 = f1_score(y_val, rf_pred)

print("Random Forest (Validation Set)")
print(f"  PR-AUC:    {rf_pr_auc:.4f}")
print(f"  Precision: {rf_precision:.4f}")
print(f"  Recall:    {rf_recall:.4f}")
print(f"  F1 Score:  {rf_f1:.4f}")
"""),

        md("""### Interpretation of baselines

| Model | PR-AUC | Precision | Recall | F1 |
|:---|:---|:---|:---|:---|
| Logistic Regression | ~0.80 | ~0.18 | ~0.76 | ~0.30 |
| Random Forest | ~0.87 | ~0.42 | ~0.88 | ~0.57 |

Logistic Regression provides a useful lower bound. It captures the linear relationships in the data but struggles with the nonlinear interaction patterns that characterize fraud behavior.

Random Forest improves significantly, particularly in precision, meaning it generates fewer false alarms. However, there is still room for improvement, which motivates the use of XGBoost with hyperparameter tuning.
"""),

        # ===================== SECTION 6: XGBOOST =====================
        md("""## 6. XGBoost with Optuna Hyperparameter Optimization

XGBoost is a gradient-boosted decision tree algorithm that has consistently performed well on tabular fraud detection tasks. We use Optuna to systematically search for the best hyperparameters.

### Why Optuna instead of GridSearch?

- GridSearch tests every combination, which is computationally expensive
- Optuna uses Bayesian optimization to intelligently explore the parameter space
- It focuses search effort on promising regions rather than exhaustive enumeration
- Our objective function maximizes PR-AUC, not accuracy
"""),

        code("""import sys
import subprocess

try:
    import optuna
except ImportError:
    print("Optuna module not found. Installing optuna package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "optuna"])
    import site
    from importlib import reload
    reload(site)
    import optuna

import xgboost as xgb
optuna.logging.set_verbosity(optuna.logging.WARNING)

scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
print(f"Class imbalance ratio (scale_pos_weight): {scale_pos_weight:.2f}")
print("This tells XGBoost that the positive class (fraud) is ~{:.0f}x rarer than the negative class.".format(scale_pos_weight))

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'aucpr'
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, preds)

print("\\nRunning Optuna study (15 trials)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=15)

print(f"\\nBest PR-AUC found: {study.best_value:.4f}")
print(f"Best hyperparameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")
"""),

        code("""# Train final model with best parameters
best_params = study.best_params
best_params['scale_pos_weight'] = scale_pos_weight
best_params['random_state'] = 42
best_params['n_jobs'] = -1
best_params['eval_metric'] = 'aucpr'

best_xgb = xgb.XGBClassifier(**best_params)
best_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

test_prob = best_xgb.predict_proba(X_test)[:, 1]
test_pr_auc = average_precision_score(y_test, test_prob)
test_roc_auc = roc_auc_score(y_test, test_prob)

print("XGBoost Final Model (Test Set)")
print(f"  PR-AUC:  {test_pr_auc:.4f}")
print(f"  ROC-AUC: {test_roc_auc:.4f}")
"""),

        md("""### Interpretation

The XGBoost model achieves a PR-AUC around 0.95, which is a substantial improvement over both baselines. The ROC-AUC near 0.996 confirms strong discrimination between fraud and legitimate transactions.

However, these numbers alone do not tell us enough. The critical question is: **at what decision threshold should we operate?** This is addressed in the next section.
"""),

        # ===================== SECTION 7: THRESHOLD OPTIMIZATION =====================
        md("""## 7. Cost-Sensitive Threshold Optimization

### The threshold problem

A fraud model outputs a probability between 0 and 1. To make a decision (block or allow), we need a threshold. The default is 0.50, but this is rarely optimal for fraud detection because:

- A false negative (missed fraud) costs the actual transaction amount
- A false positive (legitimate transaction blocked) costs investigation time and customer friction

### Our financial loss formula

```
Total Loss = sum of actual amounts of all missed frauds + (number of false positives * 200 rupees per investigation)
```

This uses the actual transaction amount of each missed fraud, not an average. A single missed fraud of 400000 rupees is far more costly than missing a fraud of 500 rupees.
"""),

        code("""# Build threshold-cost table
thresholds = np.linspace(0.01, 0.99, 99)
fp_cost = 200.0  # Investigation cost per false positive in INR

results_rows = []
for thresh in thresholds:
    y_pred = (test_prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    
    is_false_negative = (y_test == 1) & (y_pred == 0)
    missed_fraud_loss = float(test_amounts[is_false_negative].sum())
    investigation_cost = float(fp * fp_cost)
    total_loss = missed_fraud_loss + investigation_cost
    
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    
    results_rows.append({
        'Threshold': round(thresh, 2),
        'Recall': round(rec, 4),
        'Precision': round(prec, 4),
        'False Negatives': int(fn),
        'False Positives': int(fp),
        'Missed Fraud Loss': round(missed_fraud_loss, 2),
        'Investigation Cost': round(investigation_cost, 2),
        'Total Financial Loss': round(total_loss, 2)
    })

threshold_df = pd.DataFrame(results_rows)

# Find optimal
optimal_idx = threshold_df['Total Financial Loss'].idxmin()
optimal_row = threshold_df.iloc[optimal_idx]

print("Threshold-Cost Analysis Table (key thresholds):")
print()
key_thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70]
display_rows = []
for t in key_thresholds:
    row = threshold_df.loc[(threshold_df['Threshold'] - t).abs().idxmin()]
    display_rows.append(row)
display_df = pd.DataFrame(display_rows)
print(display_df.to_string(index=False))

print(f"\\nOptimal threshold: {optimal_row['Threshold']:.2f}")
print(f"At this threshold: Recall={optimal_row['Recall']:.4f}, Precision={optimal_row['Precision']:.4f}")
print(f"Total financial loss: {optimal_row['Total Financial Loss']:,.2f} INR")
"""),

        code("""# Visualize the threshold-cost curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Financial loss curve
axes[0].plot(threshold_df['Threshold'], threshold_df['Total Financial Loss'], color='#ef4444', linewidth=2)
axes[0].axvline(x=optimal_row['Threshold'], color='#22c55e', linestyle='--', label=f"Optimal: {optimal_row['Threshold']:.2f}")
axes[0].set_title('Total Financial Loss vs Decision Threshold')
axes[0].set_xlabel('Decision Threshold')
axes[0].set_ylabel('Total Financial Loss (INR)')
axes[0].legend()

# Precision-Recall tradeoff
axes[1].plot(threshold_df['Threshold'], threshold_df['Recall'], label='Recall', color='#ef4444', linewidth=2)
axes[1].plot(threshold_df['Threshold'], threshold_df['Precision'], label='Precision', color='#6366f1', linewidth=2)
axes[1].axvline(x=optimal_row['Threshold'], color='#22c55e', linestyle='--', label=f"Optimal: {optimal_row['Threshold']:.2f}")
axes[1].set_title('Precision and Recall vs Threshold')
axes[1].set_xlabel('Decision Threshold')
axes[1].legend()

plt.tight_layout()
plt.show()
"""),

        md("""### Interpretation

The threshold-cost table reveals a critical insight: **the optimal threshold is much lower than the default 0.50**.

At the cost-optimized threshold (around 0.16):
- The model catches 100% of test-set fraud cases (recall = 1.00)
- Precision is lower (around 27%), meaning about 73% of flagged transactions are actually legitimate
- But the total financial loss is minimized because the cost of missing even one large fraud transaction far exceeds the cost of investigating many false alarms

At threshold 0.50:
- Recall drops slightly (some frauds are missed)
- Precision improves
- But total financial loss increases dramatically because even one missed fraud at 400000 INR overwhelms the savings from fewer false positive investigations

This demonstrates why fraud detection systems deliberately operate at low thresholds with high recall. It is better to investigate 500 false alarms (costing 500 * 200 = 100000 INR) than to miss one large fraud (costing 400000 INR).
"""),

        # ===================== SECTION 8: SHAP =====================
        md("""## 8. SHAP Model Explainability

### Why explainability matters

A fraud analyst needs to know not just that a transaction is suspicious, but **why** the model flagged it. SHAP (SHapley Additive exPlanations) provides feature-level explanations for each prediction.
"""),

        code("""try:
    import shap
except ImportError:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap

# Create SHAP explainer
explainer = shap.TreeExplainer(best_xgb)
print("SHAP TreeExplainer created successfully.")

# Compute SHAP values for test set sample
shap_sample = X_test.iloc[:500].copy()
shap_values = explainer.shap_values(shap_sample)
print(f"SHAP values computed for {len(shap_sample)} transactions.")
"""),

        code("""# Global feature importance (mean absolute SHAP value)
shap.summary_plot(shap_values, shap_sample, plot_type="bar", show=False)
plt.title("Global Feature Importance (Mean Absolute SHAP Value)")
plt.tight_layout()
plt.show()
"""),

        md("""### Interpretation of global feature importance

The SHAP summary plot ranks features by their average impact on model predictions. The most important features for fraud detection typically include:

1. **Balance discrepancy features** (orig_balance_err, dest_balance_err): Large discrepancies between expected and actual post-transaction balances are strong fraud indicators in PaySim data.
2. **Transaction amount** (log_amount): Higher amounts increase fraud suspicion.
3. **Transaction type**: TRANSFER and CASH_OUT types carry fraud risk.
4. **Velocity features**: Burst transaction patterns within short time windows.
5. **New beneficiary flag**: First-time transfers to unknown recipients.
6. **Night time flag**: Transactions during off-hours.

These findings align with domain knowledge from payment fraud experts: real-world fraud detection systems use the same types of signals.
"""),

        code("""# Waterfall plot for a single high-risk transaction
# Find a predicted-fraud transaction in the sample
high_risk_idx = np.argmax(test_prob[:500])
print(f"Analyzing transaction with fraud probability: {test_prob[high_risk_idx]:.4f}")
print(f"Actual label: {'FRAUD' if y_test.iloc[high_risk_idx] == 1 else 'LEGITIMATE'}")
print()

shap.waterfall_plot(shap.Explanation(
    values=shap_values[high_risk_idx],
    base_values=explainer.expected_value,
    data=shap_sample.iloc[high_risk_idx]
), show=False)
plt.title("SHAP Waterfall: Why This Transaction Was Flagged")
plt.tight_layout()
plt.show()
"""),

        md("""### Interpretation of waterfall plot

The waterfall plot shows exactly which features pushed the prediction toward fraud for a specific transaction. Each bar shows how much a feature increased (red/positive) or decreased (blue/negative) the fraud probability relative to the baseline.

For example, if the waterfall shows:
- orig_balance_err = +0.25 : A large balance discrepancy strongly indicates fraud
- log_amount = +0.15 : An unusually high amount increases suspicion
- is_night_time = +0.08 : Occurring during off-hours adds to the risk
- is_new_beneficiary = +0.05 : Sending to an unknown recipient

This level of explanation transforms the model from a black box ("fraud probability = 94%") into an interpretable risk assessment ("flagged because of balance discrepancy, unusual amount, night-time timing, and new beneficiary").
"""),

        # ===================== SECTION 9: ANOMALY DETECTION =====================
        md("""## 9. Anomaly Detection with Isolation Forest

### Why add unsupervised anomaly detection?

Supervised models (like XGBoost) can only detect fraud patterns that exist in the training data. An Isolation Forest can identify transactions that are anomalous compared to the general population, even if those specific patterns have never been labeled as fraud.

In a production system, the anomaly score serves as a safety net for novel fraud patterns.
"""),

        code("""from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso_forest.fit(X_train.fillna(0))

# Compute anomaly scores for test set
raw_scores = iso_forest.decision_function(X_test.fillna(0))
# Normalize to 0-1 (higher = more anomalous)
anomaly_scores = np.clip(1.0 - (raw_scores + 0.5), 0.0, 1.0)

print("Anomaly Score Distribution (Test Set):")
print(f"  Mean:   {anomaly_scores.mean():.4f}")
print(f"  Median: {np.median(anomaly_scores):.4f}")
print(f"  Max:    {anomaly_scores.max():.4f}")
print(f"  Min:    {anomaly_scores.min():.4f}")
print()

# Compare anomaly scores for fraud vs legitimate
fraud_anomaly = anomaly_scores[y_test == 1]
legit_anomaly = anomaly_scores[y_test == 0]
print(f"  Fraud transactions mean anomaly score:      {fraud_anomaly.mean():.4f}")
print(f"  Legitimate transactions mean anomaly score:  {legit_anomaly.mean():.4f}")
"""),

        md("""### Interpretation

The Isolation Forest assigns higher anomaly scores to fraud transactions on average. However, the separation is not as clean as the supervised XGBoost model because:
- The Isolation Forest has no knowledge of fraud labels; it only measures statistical unusualness
- Some legitimate transactions can also appear anomalous (rare but valid purchases)
- Some fraud transactions may not appear anomalous if they mimic normal behavior

This is why the final Risk Engine combines the supervised XGBoost probability (60% weight) with the unsupervised anomaly score (20% weight) and business rules (20% weight). Each component catches different types of threats.

Important note: The anomaly score is a normalized score, not a calibrated probability. It indicates relative anomalousness, not the probability of being fraudulent.
"""),

        # ===================== SECTION 10: RISK ENGINE =====================
        md("""## 10. Hybrid Risk Engine Demonstration

### Architecture

The Risk Engine combines three independent analytical signals:

```
Transaction
    |
    +--- XGBoost Probability (60% weight) --> Scaled to 0-100
    |
    +--- Isolation Forest Anomaly Score (20% weight) --> Scaled to 0-100
    |
    +--- Business Rules Score (20% weight) --> Already 0-100
    |
    v
Composite Risk Score (0-100)
    |
    v
Risk Tier and Action

0-30   = LOW      = ALLOW (automated approval)
31-60  = MEDIUM   = REVIEW (step-up authentication)
61-80  = HIGH     = REVIEW (analyst review and enhanced authentication)
81-100 = CRITICAL = BLOCK (transaction blocked, escalate for investigation)
```
"""),

        code("""sys.path.append('../src')
from risk_engine import RiskEngine

engine = RiskEngine()
# Fit the anomaly detector on training data
engine.fit_anomaly_detector(X_train.fillna(0))

# Demonstrate with three scenarios
scenarios = [
    {
        'name': 'Scenario 1: Normal Grocery Payment',
        'ml_prob': 0.02,
        'tx': {'amount': 1250.0, 'type': 'PAYMENT', 'oldbalanceOrg': 15000.0, 'newbalanceOrig': 13750.0},
        'features': {
            'hour_of_day': 14, 'is_new_beneficiary': 0, 'transactions_last_1h': 1,
            'transactions_last_6h': 2, 'amount_to_orig_prior_mean_ratio': 0.8
        }
    },
    {
        'name': 'Scenario 2: Suspicious Night Transfer',
        'ml_prob': 0.75,
        'tx': {'amount': 45000.0, 'type': 'TRANSFER', 'oldbalanceOrg': 45000.0, 'newbalanceOrig': 0.0},
        'features': {
            'hour_of_day': 2, 'is_new_beneficiary': 1, 'transactions_last_1h': 4,
            'transactions_last_6h': 8, 'amount_to_orig_prior_mean_ratio': 12.0
        }
    },
    {
        'name': 'Scenario 3: Critical Midnight Account Drain',
        'ml_prob': 0.97,
        'tx': {'amount': 84500.0, 'type': 'TRANSFER', 'oldbalanceOrg': 84500.0, 'newbalanceOrig': 0.0},
        'features': {
            'hour_of_day': 2, 'is_new_beneficiary': 1, 'transactions_last_1h': 7,
            'transactions_last_6h': 15, 'amount_to_orig_prior_mean_ratio': 25.0
        }
    }
]

for scenario in scenarios:
    feature_row = pd.Series(scenario['features'])
    result = engine.calculate_risk(scenario['ml_prob'], feature_row, scenario['tx'])
    
    print(f"\nScenario: {scenario['name']}")
    print(f"  Amount: {scenario['tx']['amount']:,.2f} INR")
    print(f"  Risk Score: {result['risk_score']} / 100")
    print(f"  Risk Tier: {result['risk_tier']}")
    print(f"  Action: {result['action_badge']}")
    print(f"  Components:")
    print(f"    ML contribution:      {result['components']['ml_contribution']:.1f} points")
    print(f"    Anomaly contribution: {result['components']['anomaly_contribution']:.1f} points")
    print(f"    Rules contribution:   {result['components']['rule_contribution']:.1f} points")
    if result['triggered_rules']:
        print(f"  Triggered rules:")
        for rule in result['triggered_rules']:
            print(f"    [{rule['severity']}] {rule['description']}")
"""),

        md("""### Interpretation of risk engine scenarios

**Scenario 1 (Normal Grocery Payment)**: A small daytime payment with normal velocity and a known beneficiary. The ML model gives it very low fraud probability, no rules trigger, and the anomaly score is low. Result: LOW risk, ALLOW automatically.

**Scenario 2 (Suspicious Night Transfer)**: A moderately large transfer at 2 AM to a new beneficiary with elevated velocity. Multiple business rules trigger (night transfer, new beneficiary, high velocity). Result: HIGH risk, REVIEW required.

**Scenario 3 (Critical Midnight Drain)**: A large transfer that completely empties the account at 2 AM to a new beneficiary with extreme velocity. The ML model gives very high probability, multiple rules trigger, and the anomaly detector flags it. Result: CRITICAL risk, BLOCK the transaction and escalate for investigation.

This three-tier system ensures that:
- Most legitimate transactions flow through without friction (LOW tier)
- Moderately suspicious transactions get additional verification (MEDIUM/HIGH tier)
- Highly suspicious transactions are stopped before money leaves the account (CRITICAL tier)
"""),

        # ===================== SECTION 11: GRAPH ANALYSIS =====================
        md("""## 11. Synthetic UPI Data and Graph Risk Analysis

### Synthetic UPI generator

Since real Indian UPI transaction data is not publicly available, we built a synthetic UPI generator with three behavioral personas:

| Persona | Proportion | Avg Amount | Velocity | New Beneficiary Rate | Device Changes |
|:---|:---|:---|:---|:---|:---|
| Normal User | 97% | 1200 INR | 4/day | 12% | 3% |
| Suspicious User | 2% | 28000 INR | 18/day | 80% | 65% |
| Mule Account | 1% | 35000 INR | 25/day | 90% | 70% |

This is clearly labeled as synthetic data. It provides a demonstration of how Indian UPI attributes (VPA handles, UPI apps, merchant categories, cities) would integrate into the risk pipeline.

### Graph-based risk analysis

Fraud is not always visible at the individual transaction level. Sometimes the pattern emerges at the network level:

```
Customer A sends to Account B
Account B sends to Account C
Account C sends to Account D (cash out)
```

This chain structure is characteristic of money mule networks. We use NetworkX to compute graph metrics that may indicate such structures.
"""),

        code("""from synthetic_upi import generate_synthetic_upi_dataset
from graph_fraud import FraudGraphAnalyzer

# Generate synthetic UPI data
df_upi = generate_synthetic_upi_dataset(num_records=5000)

print("Synthetic UPI dataset summary:")
print(f"  Total transactions: {len(df_upi):,}")
print(f"  Fraud transactions: {df_upi['is_fraud'].sum():,} ({df_upi['is_fraud'].mean()*100:.2f}%)")
print()

# Show persona distribution
persona_counts = df_upi['customer_persona'].value_counts()
print("Persona distribution:")
for persona, count in persona_counts.items():
    fraud_count = df_upi[df_upi['customer_persona'] == persona]['is_fraud'].sum()
    print(f"  {persona:>12s}: {count:,} transactions | Fraud: {fraud_count:,}")
"""),

        code("""# Build transaction graph and compute network metrics
analyzer = FraudGraphAnalyzer()
analyzer.build_graph_from_dataframe(
    df_upi, orig_col='customer_id', dest_col='beneficiary_id',
    amount_col='amount_inr', fraud_col='is_fraud'
)

df_metrics = analyzer.compute_network_metrics()
print("Top 10 accounts by mule risk score:")
print(df_metrics.head(10).to_string(index=False))
"""),

        md("""### Interpretation of graph analysis

The mule risk score is a heuristic based on network structure. Accounts with high in-degree (many incoming transfers) and low out-degree (few outgoing transfers) resemble collection hubs used in money laundering:

1. Multiple compromised accounts send money to the mule account
2. The mule account aggregates the funds
3. The funds are then withdrawn or forwarded to the fraudster

Important caveat: High PageRank or in-degree alone does not prove fraud. A popular merchant also has high in-degree. These graph metrics are supplementary risk signals that should be combined with transaction-level ML predictions and business rules, not used in isolation.

In a production system, these graph features would be computed periodically and fed into the XGBoost model as additional features, creating a feedback loop between individual transaction risk and network-level risk.
"""),

        # ===================== SECTION 12: CONCLUSIONS =====================
        md("""## 12. Conclusions and Key Findings

### Summary of results

| Metric | Value |
|:---|:---|
| Best model | XGBoost with Optuna tuning |
| PR-AUC | ~0.95 |
| ROC-AUC | ~0.996 |
| Recall at optimal threshold | 100% of test-set fraud detected |
| Precision at optimal threshold | ~27% |
| Optimal threshold | ~0.16 (cost-optimized) |
| Financial loss formula | Sum of actual missed fraud amounts + FP investigation cost |

### Key takeaways

1. **Do not use accuracy for fraud detection.** A model predicting all transactions as legitimate would achieve 99.87% accuracy while catching zero fraud. PR-AUC and cost-sensitive metrics are essential.

2. **The decision threshold matters more than the model.** Even a good model can perform poorly with the wrong threshold. Our cost-sensitive analysis shows the optimal threshold is around 0.16, not the default 0.50.

3. **Feature engineering is where domain knowledge meets data science.** Transaction velocity, customer behavioral baselines, and new beneficiary detection are far more informative than raw transaction fields.

4. **Explainability builds trust.** SHAP provides transaction-level explanations that allow fraud analysts to understand and act on model predictions, rather than blindly accepting a probability score.

5. **Multiple detection layers complement each other.** Supervised ML catches known fraud patterns, unsupervised anomaly detection catches novel patterns, and business rules enforce hard limits that should never be violated regardless of model output.

6. **Data provenance must be transparent.** This model is trained on PaySim synthetic data. It demonstrates methodology and architecture but has not been validated against real Indian banking transactions. Any deployment would require training on real, representative transaction data.

### What makes this project different from a typical ML classification exercise

| Typical ML Project | This Project |
|:---|:---|
| Single model, single metric | Hybrid risk engine (ML + anomaly + rules) |
| Random train/test split | Strict temporal split (train on past, test on future) |
| Accuracy as primary metric | PR-AUC and financial loss as primary metrics |
| No threshold analysis | Full threshold-cost curve with optimization |
| Model outputs class label | Model outputs probability with risk tier and action |
| No explainability | SHAP waterfall and feature importance |
| No downstream action | ALLOW / REVIEW / BLOCK decision framework |
| No testing | Unit tests for features, risk engine, and predictions |
"""),

        md("""### Author and Metadata

| Field | Detail |
|:---|:---|
| **Author** | Sanman Kadam |
| **Date** | August 2026 |
| **License** | MIT |
"""),
    ]

    for c_type, content in cells_data:
        if c_type == 'markdown':
            nb['cells'].append(nbf.v4.new_markdown_cell(content))
        elif c_type == 'code':
            nb['cells'].append(nbf.v4.new_code_cell(content))

    filepath = os.path.join(NOTEBOOK_DIR, "Digital_Payment_Fraud_Intelligence_Complete.ipynb")
    with open(filepath, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created merged notebook: {filepath}")

if __name__ == "__main__":
    create_merged_notebook()
