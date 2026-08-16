"""
Hybrid Risk Engine for Digital Payment Fraud Intelligence System.
Fuses Supervised ML probabilities, Isolation Forest normalized anomaly scores,
and dynamic Business Rules to generate a composite Risk Score (0-100),
Risk Tier, and Decision Action (ALLOW, REVIEW, BLOCK).

NOTE: The Isolation Forest output is a normalized anomaly score, NOT a
calibrated probability. The XGBoost output is a calibrated probability
via Platt scaling. The composite score is a weighted heuristic — not a
statistical probability.
"""
from sklearn.utils.validation import check_is_fitted

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import IsolationForest

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ISO_FOREST_PATH = os.path.join(MODEL_DIR, "isolation_forest.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

class RiskEngine:
    def __init__(self, w_ml: float = None, w_anomaly: float = None, w_rules: float = None):
        """
        Initialize Risk Engine. If weights are not provided, attempts to load
        optimized weights from model_metadata.json. Falls back to 60/20/20.
        """
        self.iso_forest = None
        self._iso_forest_active = False
        
        # Load optimized weights from metadata if not explicitly provided
        default_weights = self._load_optimized_weights()
        self.w_ml = w_ml if w_ml is not None else default_weights.get("w_ml", 0.60)
        self.w_anomaly = w_anomaly if w_anomaly is not None else default_weights.get("w_anomaly", 0.20)
        self.w_rules = w_rules if w_rules is not None else default_weights.get("w_rules", 0.20)
        
        self._validate_weights()
        self._load_anomaly_model()

    def _load_optimized_weights(self) -> Dict[str, float]:
        """Load optimized weights from model metadata if available."""
        if os.path.exists(META_PATH):
            try:
                with open(META_PATH, 'r') as f:
                    meta = json.load(f)
                return meta.get("risk_engine_weights", {})
            except (json.JSONDecodeError, KeyError):
                pass
        return {}

    def _validate_weights(self):
        """Validate that component weights sum to 1.0."""
        total = self.w_ml + self.w_anomaly + self.w_rules
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Risk engine weights must sum to 1.0, got {total:.4f} "
                f"(ml={self.w_ml}, anomaly={self.w_anomaly}, rules={self.w_rules})"
            )

    def _load_anomaly_model(self):
        """Load pre-trained Isolation Forest from disk."""
        if os.path.exists(ISO_FOREST_PATH):
            try:
                self.iso_forest = joblib.load(ISO_FOREST_PATH)
                check_is_fitted(self.iso_forest)
                self._iso_forest_active = True
                print(f"[RiskEngine] Isolation Forest loaded from {ISO_FOREST_PATH}")
            except Exception as e:
                print(f"[RiskEngine] Warning: Could not load Isolation Forest: {e}")
                self._init_fallback_anomaly_model()
        else:
            print(f"[RiskEngine] Isolation Forest not found at {ISO_FOREST_PATH}. "
                  f"Anomaly scores will use unfitted baseline (0.20).")
            self._init_fallback_anomaly_model()

    def _init_fallback_anomaly_model(self):
        """Initialize unfitted Isolation Forest as fallback."""
        self.iso_forest = IsolationForest(
            n_estimators=100, 
            contamination=0.01, 
            random_state=42
        )
        self._iso_forest_active = False

    def fit_anomaly_detector(self, X: pd.DataFrame):
        """Fit Isolation Forest on historical feature matrix."""
        print("[RiskEngine] Fitting Isolation Forest anomaly detector...")
        self.iso_forest.fit(X.fillna(0))
        self._iso_forest_active = True

    @property
    def is_anomaly_active(self) -> bool:
        """Whether the Isolation Forest is fitted and producing real scores."""
        return self._iso_forest_active

    def compute_anomaly_score(self, feature_row: pd.Series) -> float:
        """
        Compute NORMALIZED anomaly score between 0.0 (normal) and 1.0 (highly anomalous).
        
        This is NOT a calibrated probability. The Isolation Forest decision_function
        returns raw scores roughly in [-0.5, 0.5]. We invert and clip to [0, 1]
        so that higher values indicate greater anomalousness.
        
        Returns 0.20 baseline if the model is not fitted.
        """
        if not self._iso_forest_active:
            return 0.20
        
        try:
            if hasattr(self.iso_forest, "feature_names_in_"):
                full_row = {col: float(feature_row.get(col, 0.0)) for col in self.iso_forest.feature_names_in_}
                df_row = pd.DataFrame([full_row])
            else:
                df_row = pd.DataFrame([feature_row.fillna(0)])
            raw_score = self.iso_forest.decision_function(df_row)[0]
            # Invert & scale: lower decision_function → more anomalous → higher score
            norm_score = float(np.clip(1.0 - (raw_score + 0.5), 0.0, 1.0))
            return round(norm_score, 4)
        except Exception:
            return 0.20

    def evaluate_business_rules(self, tx: Dict[str, Any], feature_row: pd.Series) -> Tuple[float, List[Dict[str, str]]]:
        """
        Evaluate deterministic business fraud rules.
        Returns (rule_score_0_to_100, list_of_triggered_rules).
        """
        rule_score = 0.0
        triggered_rules = []

        amount = float(tx.get('amount', 0.0))
        tx_type = str(tx.get('type', '')).upper()
        hour = int(feature_row.get('hour_of_day', 0))
        is_new_ben = int(feature_row.get('is_new_beneficiary', 0))
        vel_1h = int(feature_row.get('transactions_last_1h', 1))
        vel_6h = int(feature_row.get('transactions_last_6h', 1))
        amount_ratio = float(feature_row.get('amount_to_orig_prior_mean_ratio', 1.0))
        old_bal_org = float(tx.get('oldbalanceOrg', 0.0))
        new_bal_org = float(tx.get('newbalanceOrig', 0.0))

        # Rule 1: High Velocity Spike
        if vel_1h >= 5 or vel_6h >= 10:
            rule_score += 35.0
            triggered_rules.append({
                "rule_id": "RULE_01_VELOCITY_SPIKE",
                "severity": "HIGH",
                "description": f"Extremely high transaction velocity ({vel_1h} txs/1h, {vel_6h} txs/6h)."
            })

        # Rule 2: Large Amount + New Beneficiary
        if amount >= 50000.0 and is_new_ben == 1:
            rule_score += 30.0
            triggered_rules.append({
                "rule_id": "RULE_02_NEW_BENEFICIARY_HIGH_VAL",
                "severity": "HIGH",
                "description": f"High value transfer (₹{amount:,.2f}) to previously unseen beneficiary account."
            })

        # Rule 3: Night-Time Large Transfer
        if hour in [0, 1, 2, 3, 4, 5] and tx_type == 'TRANSFER' and amount >= 25000.0:
            rule_score += 25.0
            triggered_rules.append({
                "rule_id": "RULE_03_OFF_HOURS_TRANSFER",
                "severity": "MEDIUM",
                "description": f"Off-hours nocturnal transfer (Hour {hour:02d}:00) exceeding ₹25,000."
            })

        # Rule 4: Total Account Balance Drain
        if old_bal_org > 0 and new_bal_org == 0 and abs(amount - old_bal_org) < 1.0:
            rule_score += 20.0
            triggered_rules.append({
                "rule_id": "RULE_04_COMPLETE_BALANCE_DRAIN",
                "severity": "HIGH",
                "description": f"Complete account balance liquidation (100% of old balance ₹{old_bal_org:,.2f} emptied)."
            })

        # Rule 5: Extreme Amount Deviation vs Historical Customer Mean
        if amount_ratio >= 10.0:
            rule_score += 20.0
            triggered_rules.append({
                "rule_id": "RULE_05_HISTORICAL_AMOUNT_ANOMALY",
                "severity": "MEDIUM",
                "description": f"Current amount is {amount_ratio:.1f}x higher than customer's prior transaction average."
            })

        rule_score_normalized = min(rule_score, 100.0)
        return rule_score_normalized, triggered_rules

    def calculate_risk(
        self, 
        ml_prob: float, 
        feature_row: pd.Series, 
        tx_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesize ML probability, normalized anomaly score, and rule triggers
        into a composite Risk Score (0-100).
        
        Composite Score = (ml_prob * 100 * w_ml) + (anomaly_score * 100 * w_anomaly) + (rule_score * w_rules)
        
        All three components are scaled to 0-100 before weighting:
            - ml_prob: XGBoost calibrated output (0-1) → scaled to 0-100
            - anomaly_score: Normalized IF score (0-1) → scaled to 0-100
            - rule_score: Already on 0-100 scale
        """
        anomaly_score = self.compute_anomaly_score(feature_row)
        rule_score, triggered_rules = self.evaluate_business_rules(tx_dict, feature_row)

        # Composite score calculation (0 to 100 scale)
        ml_component = (ml_prob * 100.0) * self.w_ml
        anomaly_component = (anomaly_score * 100.0) * self.w_anomaly
        rules_component = rule_score * self.w_rules

        final_risk_score = min(max(ml_component + anomaly_component + rules_component, 0.0), 100.0)
        final_risk_score = round(final_risk_score, 1)

        # Assign Risk Tier & Action Directive
        if final_risk_score >= 81.0:
            risk_tier = "CRITICAL"
            action = "BLOCK"
            action_badge = "BLOCK TRANSACTION / ESCALATE FOR INVESTIGATION"
        elif final_risk_score >= 61.0:
            risk_tier = "HIGH"
            action = "REVIEW"
            action_badge = "STEP-UP AUTHENTICATION AND ANALYST REVIEW"
        elif final_risk_score >= 31.0:
            risk_tier = "MEDIUM"
            action = "REVIEW"
            action_badge = "STEP-UP AUTHENTICATION"
        else:
            risk_tier = "LOW"
            action = "ALLOW"
            action_badge = "AUTOMATED APPROVAL"

        return {
            "risk_score": final_risk_score,
            "risk_tier": risk_tier,
            "action": action,
            "action_badge": action_badge,
            "components": {
                "ml_probability": round(ml_prob, 4),
                "ml_contribution": round(ml_component, 1),
                "normalized_anomaly_score": round(anomaly_score, 4),
                "anomaly_contribution": round(anomaly_component, 1),
                "anomaly_model_active": self._iso_forest_active,
                "rule_score": round(rule_score, 1),
                "rule_contribution": round(rules_component, 1)
            },
            "triggered_rules": triggered_rules,
            "engine_config": {
                "w_ml": self.w_ml,
                "w_anomaly": self.w_anomaly,
                "w_rules": self.w_rules
            }
        }

if __name__ == "__main__":
    engine = RiskEngine()
    dummy_feature_row = pd.Series({
        'hour_of_day': 2,
        'is_new_beneficiary': 1,
        'transactions_last_1h': 6,
        'transactions_last_6h': 12,
        'amount_to_orig_prior_mean_ratio': 15.5
    })
    dummy_tx = {
        'amount': 84500.0,
        'type': 'TRANSFER',
        'oldbalanceOrg': 84500.0,
        'newbalanceOrig': 0.0
    }
    result = engine.calculate_risk(ml_prob=0.945, feature_row=dummy_feature_row, tx_dict=dummy_tx)
    print("Risk Engine Evaluation Result:")
    import json as _json
    print(_json.dumps(result, indent=2))
