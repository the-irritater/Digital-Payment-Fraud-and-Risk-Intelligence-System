"""
Unit tests for Prediction Pipeline.
Validates inference module loading, output structure, and error handling.
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
    assert (
        0.0 < predictor.optimal_threshold < 1.0
    ), f"Threshold {predictor.optimal_threshold} should be in (0, 1)"


def test_missing_model_raises_error():
    """FraudPredictor should raise FileNotFoundError when model binary is missing."""
    with pytest.raises(FileNotFoundError, match="FATAL"):
        FraudPredictor(model_path="/nonexistent/model.pkl")


def test_predict_single_output_structure():
    """predict_single should return a well-structured dict with all expected keys."""
    predictor = FraudPredictor()
    tx = {
        "step": 5,
        "type": "TRANSFER",
        "amount": 5000.0,
        "nameOrig": "C_TEST_1",
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 5000.0,
        "nameDest": "C_TEST_2",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    result = predictor.predict_single(tx, update_state=False)

    assert "fraud_probability" in result
    assert "threshold" in result
    assert "is_flagged_fraud" in result
    assert "raw_features" in result
    assert "prediction_metadata" in result

    meta = result["prediction_metadata"]
    assert "model_version" in meta
    assert "feature_schema_version" in meta
    assert "predicted_at" in meta


def test_predict_single_probability_bounded():
    """Single prediction fraud probability should be in [0, 1]."""
    predictor = FraudPredictor()
    tx = {
        "step": 5,
        "type": "TRANSFER",
        "amount": 5000.0,
        "nameOrig": "C_TEST_3",
        "oldbalanceOrg": 10000.0,
        "newbalanceOrig": 5000.0,
        "nameDest": "C_TEST_4",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    result = predictor.predict_single(tx, update_state=False)
    assert 0.0 <= result["fraud_probability"] <= 1.0


def test_model_version_tracking():
    """Predictor should expose model_version and feature_schema_version."""
    predictor = FraudPredictor()
    assert isinstance(predictor.model_version, str)
    assert isinstance(predictor.feature_schema_version, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
