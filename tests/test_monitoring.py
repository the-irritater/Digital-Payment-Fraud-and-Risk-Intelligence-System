"""
Unit tests for Monitoring module (Drift Detection & Business KPIs).
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "monitoring"))

from drift_detector import calculate_psi, DriftDetector
from business_kpis import BusinessKPITracker

def test_calculate_psi_identical_distributions():
    """Identical distributions should produce PSI close to 0.0."""
    np.random.seed(42)
    data = np.random.normal(100, 15, 1000)
    psi = calculate_psi(data, data)
    assert psi < 0.05, f"PSI for identical distribution should be ~0.0, got {psi}"

def test_calculate_psi_shifted_distributions():
    """Shifted distribution should produce high PSI (> 0.25)."""
    np.random.seed(42)
    base = np.random.normal(100, 15, 1000)
    curr = np.random.normal(150, 25, 1000)
    psi = calculate_psi(base, curr)
    assert psi > 0.25, f"PSI for shifted distribution should be > 0.25, got {psi}"

def test_drift_detector_evaluates_features():
    """DriftDetector should return per-feature status and overall status."""
    np.random.seed(42)
    base_df = pd.DataFrame({
        "amt": np.random.normal(5000, 1000, 500),
        "vel": np.random.poisson(3, 500)
    })
    curr_df = pd.DataFrame({
        "amt": np.random.normal(5000, 1000, 500),
        "vel": np.random.poisson(12, 500)  # shifted velocity
    })
    detector = DriftDetector(base_df)
    report = detector.detect_feature_drift(curr_df)
    
    assert "overall_status" in report
    assert "feature_details" in report
    assert "amt" in report["feature_details"]
    assert "vel" in report["feature_details"]
    assert report["feature_details"]["amt"]["status"] == "STABLE"
    assert report["feature_details"]["vel"]["status"] in ("MODERATE_DRIFT", "SIGNIFICANT_DRIFT")

def test_business_kpi_tracker():
    """BusinessKPITracker should correctly compute financial KPIs."""
    df_evals = pd.DataFrame({
        'amount': [1000.0, 50000.0, 2000.0, 30000.0],
        'is_fraud': [0, 1, 0, 1],
        'action': ['ALLOW', 'BLOCK', 'ALLOW', 'REVIEW']
    })
    tracker = BusinessKPITracker(fp_investigation_cost=200.0)
    kpis = tracker.calculate_business_kpis(df_evals)
    
    assert kpis["total_transactions"] == 4
    assert kpis["total_fraud_transactions"] == 2
    assert kpis["financial_kpis"]["fraud_capture_rate_pct"] == 100.0
    assert kpis["financial_kpis"]["blocked_fraud_value_inr"] == 80000.0
    assert kpis["financial_kpis"]["missed_fraud_loss_inr"] == 0.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
