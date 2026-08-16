"""
Feature Engineering Module for Digital Payment Fraud Intelligence System.
Transforms raw transaction logs into rich behavioral, velocity, structural, and temporal risk signals.

IMPORTANT: All features are constructed using STRICTLY CAUSAL logic.
For any transaction at time T, only information from times <= T is used.
This prevents temporal data leakage between train/validation/test sets.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

FEATURE_SCHEMA_VERSION = "2.0.0"

# Core feature columns used in downstream ML models
FEATURE_COLS = [
    'log_amount',
    'hour_of_day',
    'day_of_week',
    'is_night_time',
    'type_TRANSFER',
    'type_CASH_OUT',
    'type_PAYMENT',
    'type_CASH_IN',
    'type_DEBIT',
    'is_orig_customer',
    'is_dest_customer',
    'is_dest_merchant',
    'orig_balance_err',
    'dest_balance_err',
    'is_zero_orig_balance',
    'is_zero_dest_balance',
    'amount_to_orig_prior_mean_ratio',
    'transactions_last_1h',
    'transactions_last_6h',
    'transactions_last_24h',
    'is_new_beneficiary',
    'amount_velocity_6h'
]

TARGET_COL = 'isFraud'

def _compute_rolling_features_for_group(group: pd.DataFrame) -> pd.DataFrame:
    """
    Compute exact causal rolling window velocity and prior customer statistics for a single customer group.
    """
    steps = group['step'].values
    amounts = group['amount'].values
    n = len(steps)
    
    # 1. Rolling window transaction counts (1h, 6h, 24h)
    idx_1h = np.searchsorted(steps, steps - 1, side='left')
    idx_6h = np.searchsorted(steps, steps - 6, side='left')
    idx_24h = np.searchsorted(steps, steps - 24, side='left')
    
    counts_1h = np.arange(n) - idx_1h + 1
    counts_6h = np.arange(n) - idx_6h + 1
    counts_24h = np.arange(n) - idx_24h + 1
    
    # 2. Rolling window amount sum (6h)
    cum_amounts = np.cumsum(np.insert(amounts, 0, 0.0))
    amount_vel_6h = cum_amounts[np.arange(n) + 1] - cum_amounts[idx_6h]
    
    # 3. Expanding prior mean (strictly prior to current transaction)
    cum_prior_amounts = cum_amounts[:n]  # cumsum up to prior row
    prior_counts = np.arange(n)           # 0, 1, 2, ...
    
    prior_mean = np.where(prior_counts > 0, cum_prior_amounts / np.maximum(prior_counts, 1), amounts)
    amount_ratio = np.where(prior_mean > 0, amounts / prior_mean, 1.0)
    amount_ratio = np.clip(amount_ratio, 0.0, 100.0)
    
    res = group.copy()
    res['transactions_last_1h'] = counts_1h
    res['transactions_last_6h'] = counts_6h
    res['transactions_last_24h'] = counts_24h
    res['amount_velocity_6h'] = amount_vel_6h
    res['amount_to_orig_prior_mean_ratio'] = amount_ratio
    
    return res

def build_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Build high-leverage fraud risk features from raw dataframe.
    
    All behavioral/historical features use strictly causal (backward-looking)
    construction: for transaction at step T, only data from steps <= T is used.
    This prevents future-information leakage.
    """
    print(f"Building features for dataset with {len(df):,} records...")
    data = df.copy()
    
    # Preserve original ordering index
    data['_orig_idx'] = np.arange(len(data))
    
    # 1. Log Amount Transformation
    data['log_amount'] = np.log1p(data['amount'])
    
    # 2. Temporal Features
    data['hour_of_day'] = (data['step'] % 24).astype(int)
    data['day_of_week'] = ((data['step'] // 24) % 7).astype(int)
    data['is_night_time'] = data['hour_of_day'].isin([0, 1, 2, 3, 4, 5]).astype(int)
    
    # 3. Transaction Type One-Hot Encoding
    for t_type in ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT']:
        data[f'type_{t_type}'] = (data['type'] == t_type).astype(int)
        
    # 4. Identifier Type Flags
    orig_type = data['nameOrig'].astype(str).str[0]
    dest_type = data['nameDest'].astype(str).str[0]
    data['is_orig_customer'] = (orig_type == 'C').astype(int)
    data['is_dest_customer'] = (dest_type == 'C').astype(int)
    data['is_dest_merchant'] = (dest_type == 'M').astype(int)
    
    # 5. Domain Balance Errors & Zero Flags
    data['orig_balance_err'] = (data['oldbalanceOrg'] - data['amount']) - data['newbalanceOrig']
    data['dest_balance_err'] = (data['oldbalanceDest'] + data['amount']) - data['newbalanceDest']
    data['is_zero_orig_balance'] = (data['oldbalanceOrg'] == 0).astype(int)
    data['is_zero_dest_balance'] = (data['oldbalanceDest'] == 0).astype(int)
    
    # 6. CAUSAL Customer Historical Mean, Rolling Velocity, and Amount Sum
    data = data.sort_values(['nameOrig', 'step']).reset_index(drop=True)
    
    # Apply rolling feature computation per customer
    groups = []
    for _, group in data.groupby('nameOrig', sort=False):
        groups.append(_compute_rolling_features_for_group(group))
    data = pd.concat(groups, axis=0, ignore_index=True)
    
    # 7. CAUSAL Beneficiary Tracking (is_new_beneficiary)
    data['_pair_seq'] = data.groupby(['nameOrig', 'nameDest']).cumcount()
    data['is_new_beneficiary'] = (data['_pair_seq'] == 0).astype(int)
    data.drop(columns=['_pair_seq'], errors='ignore', inplace=True)
    
    # Sort back to original input order
    data = data.sort_values('_orig_idx').reset_index(drop=True)
    data.drop(columns=['_orig_idx'], errors='ignore', inplace=True)
    
    print(f"Feature engineering completed. Total features: {len(FEATURE_COLS)}")
    return data

def get_feature_matrix(df_featured: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract X matrix and y target series."""
    X = df_featured[FEATURE_COLS].copy()
    y = df_featured[TARGET_COL].copy() if TARGET_COL in df_featured.columns else None
    return X, y

def get_feature_schema() -> Dict[str, Any]:
    """Return structured metadata about the feature schema for validation and documentation."""
    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": len(FEATURE_COLS),
        "feature_columns": FEATURE_COLS,
        "target_column": TARGET_COL,
        "features": {
            "log_amount": {"type": "continuous", "description": "Log-transformed transaction amount", "leakage_risk": "none"},
            "hour_of_day": {"type": "categorical", "description": "Hour extracted from step (step % 24)", "leakage_risk": "none"},
            "day_of_week": {"type": "categorical", "description": "Day of week from step", "leakage_risk": "none"},
            "is_night_time": {"type": "binary", "description": "1 if hour in [0,1,2,3,4,5]", "leakage_risk": "none"},
            "type_TRANSFER": {"type": "binary", "description": "Transaction type one-hot", "leakage_risk": "none"},
            "type_CASH_OUT": {"type": "binary", "description": "Transaction type one-hot", "leakage_risk": "none"},
            "type_PAYMENT": {"type": "binary", "description": "Transaction type one-hot", "leakage_risk": "none"},
            "type_CASH_IN": {"type": "binary", "description": "Transaction type one-hot", "leakage_risk": "none"},
            "type_DEBIT": {"type": "binary", "description": "Transaction type one-hot", "leakage_risk": "none"},
            "is_orig_customer": {"type": "binary", "description": "Originator is customer account (C prefix)", "leakage_risk": "none"},
            "is_dest_customer": {"type": "binary", "description": "Destination is customer account", "leakage_risk": "none"},
            "is_dest_merchant": {"type": "binary", "description": "Destination is merchant account (M prefix)", "leakage_risk": "none"},
            "orig_balance_err": {"type": "continuous", "description": "Discrepancy: (old_bal - amount) - new_bal", "leakage_risk": "PaySim artifact"},
            "dest_balance_err": {"type": "continuous", "description": "Discrepancy: (old_bal + amount) - new_bal", "leakage_risk": "PaySim artifact"},
            "is_zero_orig_balance": {"type": "binary", "description": "Originator balance is zero before tx", "leakage_risk": "none"},
            "is_zero_dest_balance": {"type": "binary", "description": "Destination balance is zero before tx", "leakage_risk": "none"},
            "amount_to_orig_prior_mean_ratio": {"type": "continuous", "description": "Current amount / expanding prior mean for customer", "leakage_risk": "mitigated (causal expanding window)"},
            "transactions_last_1h": {"type": "count", "description": "Rolling count of customer transactions in prior 1 hour", "leakage_risk": "mitigated (backward-looking searchsorted)"},
            "transactions_last_6h": {"type": "count", "description": "Rolling count of customer transactions in prior 6 hours", "leakage_risk": "mitigated (backward-looking searchsorted)"},
            "transactions_last_24h": {"type": "count", "description": "Rolling count of customer transactions in prior 24 hours", "leakage_risk": "mitigated (backward-looking searchsorted)"},
            "is_new_beneficiary": {"type": "binary", "description": "First occurrence of (originator, destination) pair", "leakage_risk": "mitigated (cumcount)"},
            "amount_velocity_6h": {"type": "continuous", "description": "Total amount transferred by customer in prior 6 hours", "leakage_risk": "mitigated (backward-looking cumsum)"},
        }
    }

if __name__ == "__main__":
    from data_processing import load_raw_data
    df = load_raw_data()
    sample_df = df.iloc[:50000].copy()
    featured_df = build_features(sample_df)
    X, y = get_feature_matrix(featured_df)
    print("X Shape:", X.shape)
    print("y Shape:", y.shape if y is not None else "None")
    print("\nFeature Columns Sample:")
    print(X.head())
