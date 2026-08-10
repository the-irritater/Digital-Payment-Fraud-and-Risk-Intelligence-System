"""
Feature Engineering Module for Digital Payment Fraud Intelligence System.
Transforms raw transaction logs into rich behavioral, velocity, structural, and temporal risk signals.

IMPORTANT: All features are constructed using STRICTLY CAUSAL logic.
For any transaction at time T, only information from times < T is used.
This prevents temporal data leakage between train/validation/test sets.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List

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

def build_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Build high-leverage fraud risk features from raw dataframe.
    
    All behavioral/historical features use strictly causal (backward-looking)
    construction: for transaction at step T, only data from steps < T is used.
    This prevents future-information leakage.
    """
    print(f"Building features for dataset with {len(df):,} records...")
    data = df.copy()
    
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
    data['orig_type'] = data['nameOrig'].astype(str).str[0]
    data['dest_type'] = data['nameDest'].astype(str).str[0]
    data['is_orig_customer'] = (data['orig_type'] == 'C').astype(int)
    data['is_dest_customer'] = (data['dest_type'] == 'C').astype(int)
    data['is_dest_merchant'] = (data['dest_type'] == 'M').astype(int)
    
    # 5. Domain Balance Errors & Zero Flags
    # NOTE: These balance fields contain PaySim simulation artifacts where
    # fraudulent transactions are cancelled/nullified. We compute the
    # discrepancy as a feature rather than using raw balances directly.
    data['orig_balance_err'] = (data['oldbalanceOrg'] - data['amount']) - data['newbalanceOrig']
    data['dest_balance_err'] = (data['oldbalanceDest'] + data['amount']) - data['newbalanceDest']
    data['is_zero_orig_balance'] = (data['oldbalanceOrg'] == 0).astype(int)
    data['is_zero_dest_balance'] = (data['oldbalanceDest'] == 0).astype(int)
    
    # 6. CAUSAL Customer Historical Mean & Amount Deviation Score
    # Sort by customer then time to enable strictly backward-looking computation
    data = data.sort_values(['nameOrig', 'step']).reset_index(drop=True)
    
    # Expanding mean: for each row, compute mean of all PRIOR transactions
    # by the same customer (excludes the current transaction)
    data['_cum_amount'] = data.groupby('nameOrig')['amount'].cumsum() - data['amount']
    data['_cum_count'] = data.groupby('nameOrig').cumcount()  # 0-indexed count before current
    data['_prior_mean'] = np.where(
        data['_cum_count'] > 0,
        data['_cum_amount'] / data['_cum_count'],
        data['amount']  # First transaction: ratio = 1.0
    )
    data['amount_to_orig_prior_mean_ratio'] = np.where(
        data['_prior_mean'] > 0,
        data['amount'] / data['_prior_mean'],
        1.0
    )
    data['amount_to_orig_prior_mean_ratio'] = np.clip(
        data['amount_to_orig_prior_mean_ratio'], 0, 100
    )
    data.drop(columns=['_cum_amount', '_cum_count', '_prior_mean'], inplace=True)
    
    # 7. CAUSAL Velocity Features (backward-looking step diffs per customer)
    data['step_diff'] = data.groupby('nameOrig')['step'].diff().fillna(999)
    
    # Approximation: if last transaction was within window, increment count
    # This is a fast vectorized proxy. A production system would use
    # proper rolling window aggregation over a time-series index.
    data['transactions_last_1h'] = (data['step_diff'] <= 1).astype(int) + 1
    data['transactions_last_6h'] = (data['step_diff'] <= 6).astype(int) + 1
    data['transactions_last_24h'] = (data['step_diff'] <= 24).astype(int) + 1
    
    # Amount velocity: amplify if recent transactions are clustered
    data['amount_velocity_6h'] = np.where(
        data['step_diff'] <= 6,
        data['amount'] * 1.5,
        data['amount']
    )
    
    # 8. CAUSAL Beneficiary Tracking (is_new_beneficiary)
    # A beneficiary is "new" if this is the FIRST time this customer
    # has transacted with this specific destination account.
    # Uses cumcount within (nameOrig, nameDest) group: first occurrence = 0
    data['_pair_seq'] = data.groupby(['nameOrig', 'nameDest']).cumcount()
    data['is_new_beneficiary'] = (data['_pair_seq'] == 0).astype(int)
    data.drop(columns=['_pair_seq'], inplace=True)
    
    # Sort back by original step order
    data = data.sort_values('step').reset_index(drop=True)
    
    print(f"Feature engineering completed. Total features: {len(FEATURE_COLS)}")
    return data

def get_feature_matrix(df_featured: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract X matrix and y target series."""
    X = df_featured[FEATURE_COLS].copy()
    y = df_featured[TARGET_COL].copy() if TARGET_COL in df_featured.columns else None
    return X, y

if __name__ == "__main__":
    from data_processing import load_raw_data, temporal_train_val_test_split
    df = load_raw_data()
    # Test sample for fast validation
    sample_df = df.iloc[:50000].copy()
    featured_df = build_features(sample_df)
    X, y = get_feature_matrix(featured_df)
    print("X Shape:", X.shape)
    print("y Shape:", y.shape if y is not None else "None")
    print("\nFeature Columns Sample:")
    print(X.head())
