"""
Inference Module for Digital Payment Fraud Intelligence System.
Provides fast inference capabilities for individual transactions or batch transaction streams,
integrated with CustomerStateStore for genuine stateful historical velocity and behavioral features.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Union
from customer_state import get_state_store, CustomerStateStore

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
        self.model_version = "unknown"
        self.feature_schema_version = "unknown"
        self.state_store = get_state_store()
        self._load_artifacts()

    def _load_artifacts(self):
        """Load trained XGBoost model and configuration metadata."""
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                self.meta = json.load(f)
                self.feature_cols = self.meta.get("feature_cols", [])
                self.optimal_threshold = self.meta.get("optimal_threshold", 0.5)
                self.model_version = self.meta.get("model_version", "unknown")
                self.feature_schema_version = self.meta.get(
                    "feature_schema_version", "unknown"
                )

        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"[FraudPredictor] Successfully loaded model from {self.model_path}")
        else:
            raise FileNotFoundError(
                f"[FraudPredictor] FATAL: Model binary not found at {self.model_path}. "
                f"A fraud scoring system must not operate without a trained model. "
                f"Run 'python src/train.py' to train and save the model first."
            )

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud probabilities for input feature DataFrame."""
        if self.model is None:
            raise RuntimeError(
                "[FraudPredictor] Model is not loaded. Cannot produce fraud probabilities. "
                "Ensure the model was trained and saved before calling predict."
            )

        # Ensure correct column ordering
        cols_to_use = self.feature_cols if self.feature_cols else list(X.columns)
        X_aligned = X.reindex(columns=cols_to_use, fill_value=0.0).fillna(0)
        probs = self.model.predict_proba(X_aligned)[:, 1]
        return probs

    def predict_single(
        self, transaction_dict: Dict[str, Any], update_state: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single transaction payload using CustomerStateStore for genuine
        stateful velocity and historical baseline features.
        """
        df_single = pd.DataFrame([transaction_dict])
        from feature_engineering import build_features

        featured_df = build_features(df_single, is_training=False)

        # Override single-row proxies with genuine CustomerStateStore state
        realtime_state = self.state_store.compute_realtime_features(transaction_dict)
        for col, val in realtime_state.items():
            if col in featured_df.columns:
                featured_df[col] = val

        prob = float(self.predict_proba(featured_df)[0])
        is_flagged = bool(prob >= self.optimal_threshold)

        if update_state:
            self.state_store.update_state(transaction_dict)

        return {
            "fraud_probability": round(prob, 4),
            "threshold": round(self.optimal_threshold, 4),
            "is_flagged_fraud": is_flagged,
            "raw_features": featured_df[self.feature_cols].iloc[0].to_dict(),
            "prediction_metadata": {
                "model_version": self.model_version,
                "feature_schema_version": self.feature_schema_version,
                "predicted_at": datetime.now(timezone.utc).isoformat(),
            },
        }


if __name__ == "__main__":
    predictor = FraudPredictor()
    sample_tx = {
        "step": 12,
        "type": "TRANSFER",
        "amount": 84500.00,
        "nameOrig": "C928310",
        "oldbalanceOrg": 84500.00,
        "newbalanceOrig": 0.0,
        "nameDest": "C102938",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "isFraud": 1,
    }
    result = predictor.predict_single(sample_tx)
    print("Sample Single Prediction Result:")
    print(json.dumps(result, indent=2))
