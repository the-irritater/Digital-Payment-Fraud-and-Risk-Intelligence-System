"""
Model Training Module for Digital Payment Fraud Intelligence System.

Methodology Safeguards Implemented:
1. Strict Temporal Split (Train 70%, Val 15%, Test 15%).
2. Optuna Bayesian Hyperparameter Optimization on Validation Set.
3. Cost-Sensitive Threshold Selection LOCKED on Validation Set (NO Test Set Leakage).
4. Probability Calibration via CalibratedClassifierCV on Validation Set.
5. Feature Ablation Study (Model With vs Without PaySim Balance Error Artifacts).
6. Isolation Forest training and persistence for deployed anomaly detection.
7. Risk Engine weight optimization on Validation Set.
8. Calibration quality metrics (Brier Score, Expected Calibration Error).
"""

import os
import sys
import json
import platform
import joblib
import pandas as pd
import numpy as np
import optuna
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, brier_score_loss
)
import xgboost as xgb

from data_processing import load_raw_data, temporal_train_val_test_split, get_data_summary
from feature_engineering import build_features, get_feature_matrix, FEATURE_COLS, TARGET_COL, FEATURE_SCHEMA_VERSION

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Model versioning
MODEL_VERSION = "3.0.0"

def _compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) for probability predictions."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / len(y_true))

def evaluate_predictions(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    amounts: np.ndarray,
    threshold: float = 0.5,
    fp_cost: float = 200.0
) -> Dict[str, float]:
    """
    Calculate evaluation metrics and financial loss using actual transaction amounts for false negatives.
    Includes calibration metrics (Brier Score, ECE).
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = _compute_expected_calibration_error(y_true, y_prob)
    
    is_false_negative = (y_true == 1) & (y_pred == 0)
    missed_fraud_loss = float(amounts[is_false_negative].sum())
    investigation_cost = float(fp * fp_cost)
    financial_loss = missed_fraud_loss + investigation_cost
    
    # Business KPIs
    total_flagged = tp + fp
    total_transactions = len(y_true)
    total_fraud = int(y_true.sum())
    review_rate = total_flagged / total_transactions if total_transactions > 0 else 0.0
    fraud_capture_rate = recall  # same as recall
    blocked_value = float(amounts[(y_true == 1) & (y_pred == 1)].sum())
    
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 6),
        "expected_calibration_error": round(float(ece), 6),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "missed_fraud_loss_inr": round(missed_fraud_loss, 2),
        "investigation_cost_inr": round(investigation_cost, 2),
        "total_financial_loss_inr": round(financial_loss, 2),
        "review_rate": round(float(review_rate), 4),
        "fraud_capture_rate": round(float(fraud_capture_rate), 4),
        "blocked_fraud_value_inr": round(blocked_value, 2)
    }

def build_threshold_cost_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    fp_cost: float = 200.0,
    thresholds: np.ndarray = None
) -> Tuple[pd.DataFrame, float, Dict[str, float]]:
    """
    Build threshold-cost table over validation set to select optimal operating threshold.
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

def run_ablation_experiment(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series, val_amounts: np.ndarray,
    X_test: pd.DataFrame, y_test: pd.Series, test_amounts: np.ndarray,
    best_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Run ablation study comparing XGBoost performance WITH vs WITHOUT PaySim balance error features.
    """
    print("\n--- Running Feature Ablation Study ---")
    balance_cols = ['orig_balance_err', 'dest_balance_err', 'is_zero_orig_balance', 'is_zero_dest_balance']
    
    # Model A: Full Feature Set
    model_full = xgb.XGBClassifier(**best_params)
    model_full.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    probs_full = model_full.predict_proba(X_test)[:, 1]
    metrics_full = evaluate_predictions(y_test, probs_full, test_amounts, threshold=0.5)
    metrics_full['experiment'] = 'Full_Features_With_Balance_Errors'
    
    # Model B: Without Balance Error Features
    X_tr_no_bal = X_train.drop(columns=balance_cols, errors='ignore')
    X_val_no_bal = X_val.drop(columns=balance_cols, errors='ignore')
    X_te_no_bal = X_test.drop(columns=balance_cols, errors='ignore')
    
    model_no_bal = xgb.XGBClassifier(**best_params)
    model_no_bal.fit(X_tr_no_bal, y_train, eval_set=[(X_val_no_bal, y_val)], verbose=False)
    probs_no_bal = model_no_bal.predict_proba(X_te_no_bal)[:, 1]
    metrics_no_bal = evaluate_predictions(y_test, probs_no_bal, test_amounts, threshold=0.5)
    metrics_no_bal['experiment'] = 'Ablation_Without_Balance_Errors'
    
    ablation_df = pd.DataFrame([metrics_full, metrics_no_bal])
    ablation_csv_path = os.path.join(REPORTS_DIR, "ablation_study.csv")
    ablation_df.to_csv(ablation_csv_path, index=False)
    print(f"Ablation study saved to: {ablation_csv_path}")
    return ablation_df

def _train_and_save_isolation_forest(X_train: pd.DataFrame) -> IsolationForest:
    """Train Isolation Forest on training features and save to disk."""
    print("\n--- Training Isolation Forest Anomaly Detector ---")
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42
    )
    iso_forest.fit(X_train.fillna(0))
    
    iso_path = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    joblib.dump(iso_forest, iso_path)
    print(f"Isolation Forest saved to: {iso_path}")
    return iso_forest

def _run_weight_optimization(
    y_val: np.ndarray,
    val_ml_probs: np.ndarray,
    val_anomaly_scores: np.ndarray,
    val_amounts: np.ndarray,
    fp_cost: float = 200.0
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Optimize risk engine component weights on validation set by evaluating
    candidate weight combinations against expected financial loss.
    """
    print("\n--- Running Risk Engine Weight Optimization ---")
    
    candidate_weights = [
        {"w_ml": 0.60, "w_anomaly": 0.20, "w_rules": 0.20},
        {"w_ml": 0.50, "w_anomaly": 0.25, "w_rules": 0.25},
        {"w_ml": 0.70, "w_anomaly": 0.15, "w_rules": 0.15},
        {"w_ml": 0.80, "w_anomaly": 0.10, "w_rules": 0.10},
        {"w_ml": 0.65, "w_anomaly": 0.25, "w_rules": 0.10},
        {"w_ml": 0.55, "w_anomaly": 0.15, "w_rules": 0.30},
    ]
    
    results = []
    best_weights = candidate_weights[0]
    best_loss = float('inf')
    
    for weights in candidate_weights:
        # Approximate composite score using ML and anomaly (rules require full tx data)
        composite_prob = (
            val_ml_probs * weights["w_ml"] + 
            val_anomaly_scores * weights["w_anomaly"] +
            0.0 * weights["w_rules"]  # rules contribution approximated as 0 for optimization
        )
        # Normalize to [0, 1] for threshold evaluation
        composite_prob = np.clip(composite_prob, 0.0, 1.0)
        
        _, _, opt_metrics = build_threshold_cost_table(y_val, composite_prob, val_amounts, fp_cost)
        
        row = {
            "w_ml": weights["w_ml"],
            "w_anomaly": weights["w_anomaly"],
            "w_rules": weights["w_rules"],
            "optimal_loss": opt_metrics.get("total_financial_loss_inr", float('inf')),
            "optimal_recall": opt_metrics.get("recall", 0.0),
            "optimal_precision": opt_metrics.get("precision", 0.0),
        }
        results.append(row)
        
        if row["optimal_loss"] < best_loss:
            best_loss = row["optimal_loss"]
            best_weights = weights
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(REPORTS_DIR, "weight_optimization.csv"), index=False)
    print(f"Best weights: ML={best_weights['w_ml']}, Anomaly={best_weights['w_anomaly']}, Rules={best_weights['w_rules']}")
    print(f"Best expected loss: ₹{best_loss:,.2f}")
    
    return best_weights, results_df

def _save_dataset_summary(df_raw: pd.DataFrame, df_filtered: pd.DataFrame, 
                          train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Generate and save dataset summary JSON for dashboard consumption."""
    summary = get_data_summary(df_raw)
    
    # Add filtered/split info
    summary["filtered_types"] = ["TRANSFER", "CASH_OUT"]
    summary["filtered_records"] = len(df_filtered)
    summary["splits"] = {
        "train": {"records": len(train_df), "fraud": int(train_df[TARGET_COL].sum()), 
                  "fraud_rate_pct": round(train_df[TARGET_COL].mean() * 100, 4)},
        "validation": {"records": len(val_df), "fraud": int(val_df[TARGET_COL].sum()),
                       "fraud_rate_pct": round(val_df[TARGET_COL].mean() * 100, 4)},
        "test": {"records": len(test_df), "fraud": int(test_df[TARGET_COL].sum()),
                 "fraud_rate_pct": round(test_df[TARGET_COL].mean() * 100, 4)},
    }
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    
    summary_path = os.path.join(REPORTS_DIR, "dataset_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4, default=str)
    print(f"Dataset summary saved to: {summary_path}")

def train_and_evaluate_all(
    sample_size: int = 400000, 
    n_optuna_trials: int = 15
) -> Dict[str, Any]:
    """Execute end-to-end model training, tuning, calibration, and evaluation pipeline."""
    training_start = datetime.now(timezone.utc)
    
    df_raw = load_raw_data()
    df_filtered = df_raw[df_raw['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
    print(f"Filtered dataset to TRANSFER & CASH_OUT: {len(df_filtered):,} records")
    
    if sample_size and len(df_filtered) > sample_size:
        print(f"Sampling {sample_size:,} records for fast, high-performance training...")
        df_filtered = df_filtered.sample(n=sample_size, random_state=42).sort_values('step').reset_index(drop=True)
        
    featured_df = build_features(df_filtered)
    train_df, val_df, test_df = temporal_train_val_test_split(featured_df, train_ratio=0.70, val_ratio=0.15)
    
    # Save dataset summary for dashboard
    _save_dataset_summary(df_raw, df_filtered, train_df, val_df, test_df)
    
    X_train, y_train = get_feature_matrix(train_df)
    X_val, y_val = get_feature_matrix(val_df)
    X_test, y_test = get_feature_matrix(test_df)
    
    val_amounts = val_df['amount'].values
    test_amounts = test_df['amount'].values
    
    scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    results = {}
    
    # 1. Baseline Logistic Regression
    print("\n--- Training Baseline Logistic Regression ---")
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_model.fit(X_train.fillna(0), y_train)
    lr_val_prob = lr_model.predict_proba(X_val.fillna(0))[:, 1]
    results["Logistic_Regression"] = evaluate_predictions(y_val, lr_val_prob, val_amounts, threshold=0.5)
    
    # 2. Random Forest Baseline
    print("\n--- Training Random Forest Baseline ---")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model.fit(X_train.fillna(0), y_train)
    rf_val_prob = rf_model.predict_proba(X_val.fillna(0))[:, 1]
    results["Random_Forest"] = evaluate_predictions(y_val, rf_val_prob, val_amounts, threshold=0.5)
    
    # 3. XGBoost Hyperparameter Optimization via Optuna on VALIDATION SET
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
        return average_precision_score(y_val, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_optuna_trials)
    
    best_params = study.best_params
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1
    best_params['eval_metric'] = 'aucpr'
    
    # 4. Fit Base XGBoost and Calibrate Probabilities on Validation Set
    print("\n--- Fitting Base XGBoost & Calibrating Probabilities ---")
    base_xgb = xgb.XGBClassifier(**best_params)
    base_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Calibrated Classifier (Platt / Sigmoid scaling on Validation set)
    calibrated_xgb = CalibratedClassifierCV(estimator=base_xgb, method='sigmoid', cv='prefit')
    calibrated_xgb.fit(X_val, y_val)
    
    val_cal_prob = calibrated_xgb.predict_proba(X_val)[:, 1]
    
    # 5. LOCK Threshold Selection on VALIDATION SET (NO TEST SET LEAKAGE)
    val_threshold_table, locked_threshold, val_opt_metrics = build_threshold_cost_table(
        y_val.values, val_cal_prob, val_amounts, fp_cost=200.0
    )
    print(f"LOCKED Optimal Threshold selected on Validation Set: {locked_threshold:.4f}")
    
    # Save Validation Threshold Cost Table
    val_threshold_csv = os.path.join(REPORTS_DIR, "threshold_cost_table.csv")
    val_threshold_table.to_csv(val_threshold_csv, index=False)
    print(f"Threshold-cost table saved to: {val_threshold_csv}")
    
    # 6. ONE-TIME Final Evaluation on UNTOUCHED TEST SET
    test_cal_prob = calibrated_xgb.predict_proba(X_test)[:, 1]
    final_test_metrics = evaluate_predictions(
        y_test.values, test_cal_prob, test_amounts, threshold=locked_threshold, fp_cost=200.0
    )
    results["XGBoost_Calibrated_Locked_Threshold"] = final_test_metrics
    
    # Also record XGBoost at default 0.5 threshold for comparison
    default_test_metrics = evaluate_predictions(
        y_test.values, test_cal_prob, test_amounts, threshold=0.5, fp_cost=200.0
    )
    results["XGBoost_Default_0.5"] = default_test_metrics
    
    print(f"\n--- UNTOUCHED TEST SET EVALUATION (@ Locked Threshold={locked_threshold:.4f}) ---")
    print(f"PR-AUC: {final_test_metrics['pr_auc']:.4f} | ROC-AUC: {final_test_metrics['roc_auc']:.4f}")
    print(f"Precision: {final_test_metrics['precision']:.4f} | Recall: {final_test_metrics['recall']:.4f} | F1: {final_test_metrics['f1_score']:.4f}")
    print(f"Brier Score: {final_test_metrics['brier_score']:.6f} | ECE: {final_test_metrics['expected_calibration_error']:.6f}")
    print(f"Missed Fraud Loss: ₹{final_test_metrics['missed_fraud_loss_inr']:,.2f} | Investigation Cost: ₹{final_test_metrics['investigation_cost_inr']:,.2f}")
    print(f"Total Expected Financial Loss: ₹{final_test_metrics['total_financial_loss_inr']:,.2f}")
    
    # 7. Run Feature Ablation Experiment
    run_ablation_experiment(
        X_train, y_train, X_val, y_val, val_amounts, X_test, y_test, test_amounts, best_params
    )
    
    # 8. Train and Save Isolation Forest
    iso_forest = _train_and_save_isolation_forest(X_train)
    
    # 9. Run Risk Engine Weight Optimization
    # Compute anomaly scores on validation set
    val_anomaly_raw = iso_forest.decision_function(X_val.fillna(0))
    val_anomaly_scores = np.clip(1.0 - (val_anomaly_raw + 0.5), 0.0, 1.0)
    
    best_weights, _ = _run_weight_optimization(
        y_val.values, val_cal_prob, val_anomaly_scores, val_amounts, fp_cost=200.0
    )
    
    # Save Calibrated Model in both pickle and native XGBoost JSON
    xgb_pkl_path = os.path.join(MODEL_DIR, "xgboost_model.pkl")
    xgb_json_path = os.path.join(MODEL_DIR, "xgboost_model.json")
    meta_filepath = os.path.join(MODEL_DIR, "model_metadata.json")
    
    joblib.dump(calibrated_xgb, xgb_pkl_path)
    base_xgb.save_model(xgb_json_path)
    
    training_end = datetime.now(timezone.utc)
    serializable_params = {k: (v.item() if hasattr(v, 'item') else v) for k, v in best_params.items()}
    
    metadata = {
        "model_version": MODEL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_cols": FEATURE_COLS,
        "best_params": serializable_params,
        "optimal_threshold": float(locked_threshold),
        "threshold_selection_split": "Validation",
        "test_metrics": final_test_metrics,
        "baseline_comparison": results,
        "calibration": {
            "method": "sigmoid (Platt scaling)",
            "cv": "prefit on validation set",
            "brier_score": final_test_metrics["brier_score"],
            "expected_calibration_error": final_test_metrics["expected_calibration_error"]
        },
        "risk_engine_weights": best_weights,
        "cost_model": {
            "description": "Financial loss = sum(actual_amount of each missed fraud tx) + (FP_count * investigation_cost_per_fp)",
            "investigation_cost_per_fp_inr": 200.0,
            "fn_cost_method": "actual_transaction_amount",
            "threshold_selection_split": "Validation set ONLY (Locked before test evaluation)"
        },
        "data_provenance": {
            "training_data": "PaySim synthetic mobile-money simulation (Kaggle)",
            "split_method": "Strict temporal split on step column (70/15/15)",
            "filtered_types": ["TRANSFER", "CASH_OUT"],
            "sample_size": sample_size
        },
        "software_versions": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "sklearn": __import__('sklearn').__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "optuna": optuna.__version__
        },
        "training_timestamp": training_start.isoformat(),
        "training_duration_seconds": round((training_end - training_start).total_seconds(), 1)
    }
    with open(meta_filepath, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    comparison_rows = [{"model": k, **v} for k, v in results.items()]
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    
    print(f"\nModel saved to: {xgb_pkl_path}")
    print(f"XGBoost native JSON saved to: {xgb_json_path}")
    print(f"Metadata saved to: {meta_filepath}")
    return metadata

if __name__ == "__main__":
    train_and_evaluate_all()
