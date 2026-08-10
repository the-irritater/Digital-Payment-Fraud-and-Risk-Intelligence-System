"""
Inference Module for Digital Payment Fraud Intelligence System.
Provides fast inference capabilities for individual transactions or batch transaction streams.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Union

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
XGB_PATH = os.path.join(MODEL_DIR, "xgboost_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

class FraudPredictor:
    def __init__(self, model_path: str = XGB_PATH, meta_path: str = META_PATH):
        self.model_path = model_path
        self.meta_path = meta_path
        self.model = None
        self.meta = None
        self.feature_cols = []
        self.optimal_threshold = 0.5
        self._load_artifacts()

    def _load_artifacts(self):
        """Load trained XGBoost model and configuration metadata."""
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r') as f:
                self.meta = json.load(f)
                self.feature_cols = self.meta.get("feature_cols", [])
                self.optimal_threshold = self.meta.get("optimal_threshold", 0.5)
        
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"[FraudPredictor] Successfully loaded model from {self.model_path}")
        else:
            print(f"[FraudPredictor] Warning: Model binary not found at {self.model_path}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud probabilities for input feature DataFrame."""
        if self.model is None:
            # Fallback heuristic probability if model binary is building
            return np.full(len(X), 0.05)
        
        # Ensure correct column ordering
        X_aligned = X[self.feature_cols].fillna(0)
        probs = self.model.predict_proba(X_aligned)[:, 1]
        return probs

    def predict_single(self, transaction_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single transaction payload and return prediction details."""
        df_single = pd.DataFrame([transaction_dict])
        from feature_engineering import build_features
        featured_df = build_features(df_single, is_training=False)
        prob = float(self.predict_proba(featured_df)[0])
        is_flagged = bool(prob >= self.optimal_threshold)
        
        return {
            "fraud_probability": round(prob, 4),
            "threshold": round(self.optimal_threshold, 4),
            "is_flagged_fraud": is_flagged,
            "raw_features": featured_df[self.feature_cols].iloc[0].to_dict()
        }

if __name__ == "__main__":
    predictor = FraudPredictor()
    sample_tx = {
        'step': 12,
        'type': 'TRANSFER',
        'amount': 84500.00,
        'nameOrig': 'C928310',
        'oldbalanceOrg': 84500.00,
        'newbalanceOrig': 0.0,
        'nameDest': 'C102938',
        'oldbalanceDest': 0.0,
        'newbalanceDest': 0.0,
        'isFraud': 1
    }
    result = predictor.predict_single(sample_tx)
    print("Sample Single Prediction Result:")
    print(json.dumps(result, indent=2))
