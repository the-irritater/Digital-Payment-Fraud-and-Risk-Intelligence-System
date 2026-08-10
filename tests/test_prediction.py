"""
Unit tests for Prediction Pipeline.
Validates inference module loading and output structure.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import FraudPredictor

def test_predictor_loads():
    """FraudPredictor should initialize without errors."""
    predictor = FraudPredictor()
    assert predictor is not None
    assert isinstance(predictor.feature_cols, list)
    assert len(predictor.feature_cols) > 0

def test_predict_proba_returns_array():
    """predict_proba should return a numpy array of the same length as input."""
    predictor = FraudPredictor()
    # Create a dummy feature DataFrame matching expected columns
    dummy_data = {col: [0.0] for col in predictor.feature_cols}
    X = pd.DataFrame(dummy_data)
    probs = predictor.predict_proba(X)
    assert isinstance(probs, np.ndarray)
    assert len(probs) == 1

def test_predict_proba_bounded():
    """All predicted probabilities should be in [0, 1]."""
    predictor = FraudPredictor()
    dummy_data = {col: [0.0, 1.0, 5.0] for col in predictor.feature_cols}
    X = pd.DataFrame(dummy_data)
    probs = predictor.predict_proba(X)
    assert (probs >= 0.0).all(), "Probabilities should be >= 0"
    assert (probs <= 1.0).all(), "Probabilities should be <= 1"

def test_optimal_threshold_is_valid():
    """Optimal threshold should be a float in (0, 1)."""
    predictor = FraudPredictor()
    assert 0.0 < predictor.optimal_threshold < 1.0, \
        f"Threshold {predictor.optimal_threshold} should be in (0, 1)"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
