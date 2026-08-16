"""
Unit tests for Feature Engineering Pipeline.
Validates causal feature construction, data leakage prevention, and output correctness.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_engineering import build_features, get_feature_matrix, get_feature_schema, FEATURE_COLS, TARGET_COL, FEATURE_SCHEMA_VERSION

def _make_sample_df():
    """Create a minimal sample dataframe for testing."""
    return pd.DataFrame({
        'step': [1, 2, 3, 10, 11, 12],
        'type': ['TRANSFER', 'CASH_OUT', 'TRANSFER', 'TRANSFER', 'CASH_OUT', 'TRANSFER'],
        'amount': [1000.0, 5000.0, 2000.0, 50000.0, 3000.0, 80000.0],
        'nameOrig': ['C100', 'C100', 'C100', 'C200', 'C200', 'C100'],
        'oldbalanceOrg': [10000.0, 9000.0, 4000.0, 50000.0, 47000.0, 2000.0],
        'newbalanceOrig': [9000.0, 4000.0, 2000.0, 0.0, 44000.0, 0.0],
        'nameDest': ['C300', 'M400', 'C300', 'C500', 'M400', 'C600'],
        'oldbalanceDest': [0.0, 0.0, 1000.0, 0.0, 0.0, 0.0],
        'newbalanceDest': [1000.0, 5000.0, 3000.0, 50000.0, 3000.0, 80000.0],
        'isFraud': [0, 0, 0, 1, 0, 1]
    })

def test_build_features_output_shape():
    """All expected feature columns should be present after feature engineering."""
    df = _make_sample_df()
    featured = build_features(df)
    X, y = get_feature_matrix(featured)
    
    assert X.shape[1] == len(FEATURE_COLS), f"Expected {len(FEATURE_COLS)} features, got {X.shape[1]}"
    assert len(X) == len(df), "Row count should be preserved"
    for col in FEATURE_COLS:
        assert col in X.columns, f"Missing feature column: {col}"

def test_no_null_features():
    """Feature matrix should have no NaN values."""
    df = _make_sample_df()
    featured = build_features(df)
    X, _ = get_feature_matrix(featured)
    null_cols = X.columns[X.isnull().any()].tolist()
    assert len(null_cols) == 0, f"Null values found in columns: {null_cols}"

def test_causal_customer_mean():
    """
    For the first transaction of any customer, the amount-to-prior-mean ratio
    should be 1.0 (no prior history to compare against).
    """
    df = _make_sample_df()
    featured = build_features(df)
    # C100's first transaction (step=1, amount=1000)
    c100_first = featured[
        (featured['nameOrig'] == 'C100') & (featured['step'] == 1)
    ]
    ratio = c100_first['amount_to_orig_prior_mean_ratio'].values[0]
    assert ratio == 1.0, f"First transaction ratio should be 1.0, got {ratio}"

def test_new_beneficiary_flag():
    """
    First transaction between (C100, C300) should be marked as new beneficiary.
    Second transaction between (C100, C300) should NOT be marked as new.
    """
    df = _make_sample_df()
    featured = build_features(df)
    c100_to_c300 = featured[
        (featured['nameOrig'] == 'C100') & (featured['nameDest'] == 'C300')
    ].sort_values('step')
    
    assert len(c100_to_c300) == 2
    assert c100_to_c300.iloc[0]['is_new_beneficiary'] == 1, "First occurrence should be new"
    assert c100_to_c300.iloc[1]['is_new_beneficiary'] == 0, "Second occurrence should NOT be new"

def test_log_amount_positive():
    """log_amount should be positive for all positive transaction amounts."""
    df = _make_sample_df()
    featured = build_features(df)
    assert (featured['log_amount'] > 0).all(), "log_amount should be > 0 for positive amounts"

def test_night_time_flag():
    """Transactions in steps mapping to hours 0-5 should be flagged as night time."""
    df = _make_sample_df()
    featured = build_features(df)
    for _, row in featured.iterrows():
        hour = row['hour_of_day']
        expected_night = 1 if hour in [0, 1, 2, 3, 4, 5] else 0
        assert row['is_night_time'] == expected_night, f"Night flag wrong for hour {hour}"

def test_temporal_leakage_prevention():
    """
    Changing a FUTURE transaction's amount must NOT affect feature values of EARLIER transactions.
    This validates that all features are strictly backward-looking (causal).
    """
    df = _make_sample_df()
    featured_original = build_features(df.copy())
    
    # Modify the LAST transaction (step=12, C100→C600, amount=80000)
    df_modified = df.copy()
    df_modified.loc[df_modified['step'] == 12, 'amount'] = 999999.0
    featured_modified = build_features(df_modified)
    
    # All features for EARLIER transactions (steps < 12) should be identical
    earlier_cols = FEATURE_COLS
    for step in [1, 2, 3, 10, 11]:
        orig_row = featured_original[featured_original['step'] == step][earlier_cols].iloc[0]
        mod_row = featured_modified[featured_modified['step'] == step][earlier_cols].iloc[0]
        diff = (orig_row - mod_row).abs().sum()
        assert diff < 1e-10, f"Temporal leakage detected at step {step}: features changed when future transaction was modified"

def test_feature_schema_version_exists():
    """Feature schema version should be a non-empty string."""
    assert isinstance(FEATURE_SCHEMA_VERSION, str)
    assert len(FEATURE_SCHEMA_VERSION) > 0

def test_get_feature_schema_structure():
    """get_feature_schema should return a well-formed dict with all required keys."""
    schema = get_feature_schema()
    assert "schema_version" in schema
    assert "feature_count" in schema
    assert "feature_columns" in schema
    assert "target_column" in schema
    assert "features" in schema
    assert schema["feature_count"] == len(FEATURE_COLS)
    assert schema["target_column"] == TARGET_COL
    
    # Every feature column should have metadata
    for col in FEATURE_COLS:
        assert col in schema["features"], f"Missing schema entry for feature: {col}"
        entry = schema["features"][col]
        assert "type" in entry
        assert "description" in entry
        assert "leakage_risk" in entry

def test_velocity_counts_are_integers():
    """Velocity count features should produce integer-like values."""
    df = _make_sample_df()
    featured = build_features(df)
    for col in ['transactions_last_1h', 'transactions_last_6h', 'transactions_last_24h']:
        values = featured[col].values
        assert np.all(values >= 0), f"{col} should be non-negative"
        assert np.all(values == values.astype(int)), f"{col} should be integer counts"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
