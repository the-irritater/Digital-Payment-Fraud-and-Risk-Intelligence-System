"""
Unit tests for CustomerStateStore, DatabaseManager, and Probability Calibration.
"""

import pytest
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sql")))

from customer_state import CustomerStateStore
from db_manager import DatabaseManager

def test_customer_state_store_realtime_features():
    store = CustomerStateStore()
    
    # Customer makes first transaction at step 10 for 5000 INR
    tx1 = {'nameOrig': 'C100', 'nameDest': 'C200', 'step': 10, 'amount': 5000.0}
    feats1 = store.compute_realtime_features(tx1)
    
    assert feats1['transactions_last_1h'] == 1
    assert feats1['transactions_last_24h'] == 1
    assert feats1['is_new_beneficiary'] == 1
    assert feats1['amount_to_orig_prior_mean_ratio'] == 1.0
    
    store.update_state(tx1)
    
    # Customer makes second transaction at step 12 to SAME beneficiary for 10000 INR
    tx2 = {'nameOrig': 'C100', 'nameDest': 'C200', 'step': 12, 'amount': 10000.0}
    feats2 = store.compute_realtime_features(tx2)
    
    assert feats2['transactions_last_24h'] == 2
    assert feats2['is_new_beneficiary'] == 0
    assert feats2['amount_to_orig_prior_mean_ratio'] == 2.0  # 10000 / 5000 = 2.0

def test_database_manager_sqlite_logging(tmp_path):
    db_file = str(tmp_path / "test_fraud.db")
    db_mgr = DatabaseManager(db_path=db_file)
    
    tx = {'step': 5, 'type': 'TRANSFER', 'amount': 45000.0, 'nameOrig': 'C111', 'oldbalanceOrg': 45000.0, 'newbalanceOrig': 0.0, 'nameDest': 'C222', 'oldbalanceDest': 0.0, 'newbalanceDest': 0.0}
    risk_res = {
        'risk_score': 88.5,
        'risk_tier': 'CRITICAL',
        'action': 'BLOCK',
        'action_badge': 'BLOCK TRANSACTION',
        'components': {'ml_probability': 0.92, 'normalized_anomaly_score': 0.75, 'rule_score': 40.0, 'ml_contribution': 55.2, 'anomaly_contribution': 15.0, 'rule_contribution': 8.0},
        'triggered_rules': [{'rule_id': 'R001', 'severity': 'HIGH', 'description': 'Nocturnal drain'}]
    }
    
    case_id = db_mgr.log_evaluation(tx, risk_res)
    assert case_id.startswith("CASE_")
    
    df_queue = db_mgr.get_investigation_queue()
    assert not df_queue.empty
    assert "C111" in df_queue["Originator"].values
