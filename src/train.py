"""
Model Training Module for Digital Payment Fraud Intelligence System.
Includes Logistic Regression baseline, Random Forest, XGBoost with Optuna PR-AUC optimization,
and financial cost-sensitive threshold selection using actual missed-transaction amounts.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import optuna
from typing import Dict, Any, Tuple, List

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, precision_recall_curve
)
import xgboost as xgb

from data_processing import load_raw_data, temporal_train_val_test_split
from feature_engineering import build_features, get_feature_matrix, FEATURE_COLS, TARGET_COL

# Prevent verbose Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def evaluate_predictions(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    amounts: np.ndarray,
    threshold: float = 0.5,
    fp_cost: float = 200.0
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics and financial loss.
    
    Financial loss uses ACTUAL transaction amounts for false negatives:
        Loss = sum(amount_i for each missed fraud i) + (FP_count * fp_cost)
    This is more defensible than using an average fraud amount proxy.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    
    # Cost-sensitive loss: actual amounts of missed frauds + investigation cost per FP
    is_false_negative = (y_true == 1) & (y_pred == 0)
    missed_fraud_loss = float(amounts[is_false_negative].sum())
    investigation_cost = float(fp * fp_cost)
    financial_loss = missed_fraud_loss + investigation_cost
    
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "missed_fraud_loss_inr": round(missed_fraud_loss, 2),
        "investigation_cost_inr": round(investigation_cost, 2),
        "total_financial_loss_inr": round(financial_loss, 2)
    }

def build_threshold_cost_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    fp_cost: float = 200.0,
    thresholds: np.ndarray = None
) -> Tuple[pd.DataFrame, float, Dict[str, float]]:
    """
    Build a complete threshold-cost table across candidate thresholds.
    Returns (table_df, optimal_threshold, optimal_metrics).
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    
    rows = []
    best_threshold = 0.5
    min_loss = float('inf')
    best_metrics = {}
    
    for thresh in thresholds:
        metrics = evaluate_predictions(y_true, y_prob, amounts, threshold=thresh, fp_cost=fp_cost)
        rows.append(metrics)
        if metrics["total_financial_loss_inr"] < min_loss:
            min_loss = metrics["total_financial_loss_inr"]
            best_threshold = thresh
            best_metrics = metrics
    
    table_df = pd.DataFrame(rows)
    return table_df, best_threshold, best_metrics

def train_and_evaluate_all(
    sample_size: int = 400000, 
    n_optuna_trials: int = 15
) -> Dict[str, Any]:
    """Execute end-to-end model training, tuning, and evaluation pipeline."""
    df_raw = load_raw_data()
    
    # Filter to high-risk payment types (TRANSFER & CASH_OUT carry 100% of PaySim frauds)
    df_filtered = df_raw[df_raw['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
    print(f"Filtered dataset to TRANSFER & CASH_OUT: {len(df_filtered):,} records (Fraud count: {df_filtered['isFraud'].sum():,})")
    
    if sample_size and len(df_filtered) > sample_size:
        print(f"Sampling {sample_size:,} records for fast, high-performance training...")
        # Stratified/temporal sample preserving temporal step order
        df_filtered = df_filtered.sample(n=sample_size, random_state=42).sort_values('step').reset_index(drop=True)
        
    # Feature engineering
    featured_df = build_features(df_filtered)
    
    # Temporal Train/Val/Test Split
    train_df, val_df, test_df = temporal_train_val_test_split(featured_df, train_ratio=0.70, val_ratio=0.15)
    
    X_train, y_train = get_feature_matrix(train_df)
    X_val, y_val = get_feature_matrix(val_df)
    X_test, y_test = get_feature_matrix(test_df)
    
    # Preserve actual transaction amounts for cost-sensitive evaluation
    val_amounts = val_df['amount'].values
    test_amounts = test_df['amount'].values
    
    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    print(f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")
    
    results = {}
    
    # 1. Baseline Logistic Regression
    print("\n--- Training Baseline Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_model.fit(X_train.fillna(0), y_train)
    lr_val_prob = lr_model.predict_proba(X_val.fillna(0))[:, 1]
    lr_metrics = evaluate_predictions(y_val, lr_val_prob, val_amounts, threshold=0.5)
    results["Logistic_Regression"] = lr_metrics
    print(f"Logistic Regression Val PR-AUC: {lr_metrics['pr_auc']:.4f} | F1: {lr_metrics['f1_score']:.4f}")
    
    # 2. Random Forest Baseline
    print("\n--- Training Random Forest Baseline ---")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_train.fillna(0), y_train)
    rf_val_prob = rf_model.predict_proba(X_val.fillna(0))[:, 1]
    rf_metrics = evaluate_predictions(y_val, rf_val_prob, val_amounts, threshold=0.5)
    results["Random_Forest"] = rf_metrics
    print(f"Random Forest Val PR-AUC: {rf_metrics['pr_auc']:.4f} | F1: {rf_metrics['f1_score']:.4f}")
    
    # 3. XGBoost Hyperparameter Optimization via Optuna
    print(f"\n--- Running Optuna Study for XGBoost ({n_optuna_trials} trials) ---")
    
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
        score = average_precision_score(y_val, preds)
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_optuna_trials)
    
    best_params = study.best_params
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['eval_metric'] = 'aucpr'
    
    print(f"Best Optuna Trial PR-AUC: {study.best_value:.4f}")
    print(f"Best Hyperparameters: {best_params}")
    
    # 4. Train Final Best XGBoost Model
    best_xgb = xgb.XGBClassifier(**best_params)
    best_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    test_prob = best_xgb.predict_proba(X_test)[:, 1]
    
    # Evaluate at default 0.5 threshold
    default_test_metrics = evaluate_predictions(y_test, test_prob, test_amounts, threshold=0.5)
    results["XGBoost_Default_0.5"] = default_test_metrics
    
    # Build full threshold-cost table and find optimal threshold
    threshold_table, opt_thresh, opt_test_metrics = build_threshold_cost_table(
        y_test.values, test_prob, test_amounts, fp_cost=200.0
    )
    results["XGBoost_Optimal_Financial"] = opt_test_metrics
    
    # Save threshold-cost table to reports
    threshold_csv_path = os.path.join(REPORTS_DIR, "threshold_cost_table.csv")
    threshold_table.to_csv(threshold_csv_path, index=False)
    print(f"\nThreshold-cost table saved to: {threshold_csv_path}")
    
    # Print a summary of key thresholds
    print("\n--- THRESHOLD-COST TABLE (Key Points) ---")
    print(f"{'Threshold':>10} {'Recall':>8} {'Precision':>10} {'FN':>5} {'FP':>6} {'Missed Loss (₹)':>16} {'FP Cost (₹)':>12} {'Total Loss (₹)':>15}")
    for t in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70]:
        row = threshold_table.loc[(threshold_table['threshold'] - t).abs().idxmin()]
        print(f"{row['threshold']:>10.2f} {row['recall']:>8.4f} {row['precision']:>10.4f} {row['false_negatives']:>5.0f} {row['false_positives']:>6.0f} {row['missed_fraud_loss_inr']:>16,.2f} {row['investigation_cost_inr']:>12,.2f} {row['total_financial_loss_inr']:>15,.2f}")
    
    print(f"\n--- FINAL TEST SET PERFORMANCE (XGBoost @ threshold={opt_thresh:.4f}) ---")
    print(f"PR-AUC: {opt_test_metrics['pr_auc']:.4f} | ROC-AUC: {opt_test_metrics['roc_auc']:.4f}")
    print(f"Optimal Decision Threshold: {opt_thresh:.4f}")
    print(f"Precision: {opt_test_metrics['precision']:.4f} | Recall: {opt_test_metrics['recall']:.4f} | F1: {opt_test_metrics['f1_score']:.4f}")
    print(f"Missed Fraud Loss: ₹{opt_test_metrics['missed_fraud_loss_inr']:,.2f} | Investigation Cost: ₹{opt_test_metrics['investigation_cost_inr']:,.2f}")
    print(f"Total Expected Financial Loss: ₹{opt_test_metrics['total_financial_loss_inr']:,.2f}")
    print(f"  (TP: {opt_test_metrics['true_positives']}, FP: {opt_test_metrics['false_positives']}, FN: {opt_test_metrics['false_negatives']}, TN: {opt_test_metrics['true_negatives']})")
    
    # Save Model & Metadata
    xgb_filepath = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    meta_filepath = os.path.join(MODEL_DIR, "model_metadata.json")
    
    joblib.dump(best_xgb, xgb_filepath)
    
    # Serialize params — convert numpy types to native Python
    serializable_params = {}
    for k, v in best_params.items():
        if hasattr(v, 'item'):
            serializable_params[k] = v.item()
        else:
            serializable_params[k] = v
    
    metadata = {
        "feature_cols": FEATURE_COLS,
        "best_params": serializable_params,
        "optimal_threshold": float(opt_thresh),
        "test_metrics": opt_test_metrics,
        "baseline_comparison": results,
        "cost_model": {
            "description": "Financial loss = sum(actual_amount of each missed fraud tx) + (FP_count * investigation_cost_per_fp)",
            "investigation_cost_per_fp_inr": 200.0,
            "fn_cost_method": "actual_transaction_amount"
        },
        "data_provenance": {
            "training_data": "PaySim synthetic mobile-money simulation (Kaggle)",
            "split_method": "Strict temporal split on step column (70/15/15)",
            "filtered_types": ["TRANSFER", "CASH_OUT"],
            "sample_size": sample_size
        }
    }
    with open(meta_filepath, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    # Save experiment tracking CSV
    comparison_rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        row.update(metrics)
        comparison_rows.append(row)
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"\nModel comparison table saved to: {comparison_csv_path}")
    
    print(f"Model saved to: {xgb_filepath}")
    print(f"Metadata saved to: {meta_filepath}")
    
    return metadata

if __name__ == "__main__":
    train_and_evaluate_all()
