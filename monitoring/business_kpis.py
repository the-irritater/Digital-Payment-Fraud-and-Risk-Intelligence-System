"""
Business KPI Tracking Layer for Digital Payment Fraud Intelligence System.
Calculates production fraud performance metrics, cost per detection, and analyst efficiency.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

class BusinessKPITracker:
    def __init__(self, fp_investigation_cost: float = 200.0):
        self.fp_investigation_cost = fp_investigation_cost

    def calculate_business_kpis(
        self,
        df_evaluations: pd.DataFrame,
        actual_fraud_col: str = 'is_fraud',
        predicted_action_col: str = 'action',
        amount_col: str = 'amount'
    ) -> Dict[str, Any]:
        """
        Compute business-oriented KPIs from live evaluation logs.
        
        Args:
            df_evaluations: DataFrame containing actual fraud status, risk engine action, and transaction amounts.
        """
        if df_evaluations.empty:
            return {"error": "Empty evaluations DataFrame"}

        total_tx = len(df_evaluations)
        total_fraud_tx = int(df_evaluations[actual_fraud_col].sum()) if actual_fraud_col in df_evaluations.columns else 0
        total_volume = float(df_evaluations[amount_col].sum())
        total_fraud_volume = float(df_evaluations[df_evaluations[actual_fraud_col] == 1][amount_col].sum()) if actual_fraud_col in df_evaluations.columns else 0.0

        # Action breakdown
        blocked_tx = df_evaluations[df_evaluations[predicted_action_col] == 'BLOCK']
        reviewed_tx = df_evaluations[df_evaluations[predicted_action_col] == 'REVIEW']
        allowed_tx = df_evaluations[df_evaluations[predicted_action_col] == 'ALLOW']

        num_blocked = len(blocked_tx)
        num_reviewed = len(reviewed_tx)
        num_allowed = len(allowed_tx)

        # Value blocked vs missed
        if actual_fraud_col in df_evaluations.columns:
            captured_fraud_tx = df_evaluations[
                (df_evaluations[actual_fraud_col] == 1) & 
                (df_evaluations[predicted_action_col].isin(['BLOCK', 'REVIEW']))
            ]
            missed_fraud_tx = df_evaluations[
                (df_evaluations[actual_fraud_col] == 1) & 
                (df_evaluations[predicted_action_col] == 'ALLOW')
            ]
            fp_tx = df_evaluations[
                (df_evaluations[actual_fraud_col] == 0) & 
                (df_evaluations[predicted_action_col].isin(['BLOCK', 'REVIEW']))
            ]

            fraud_capture_rate = len(captured_fraud_tx) / max(total_fraud_tx, 1)
            blocked_value = float(captured_fraud_tx[amount_col].sum())
            missed_value = float(missed_fraud_tx[amount_col].sum())
            fp_count = len(fp_tx)
            investigation_cost = fp_count * self.fp_investigation_cost
            total_financial_loss = missed_value + investigation_cost
            cost_per_detected_fraud = (total_financial_loss / max(len(captured_fraud_tx), 1))
        else:
            fraud_capture_rate = 0.0
            blocked_value = float(blocked_tx[amount_col].sum())
            missed_value = 0.0
            fp_count = 0
            investigation_cost = 0.0
            total_financial_loss = 0.0
            cost_per_detected_fraud = 0.0

        return {
            "total_transactions": total_tx,
            "total_volume_inr": round(total_volume, 2),
            "total_fraud_transactions": total_fraud_tx,
            "total_fraud_exposure_inr": round(total_fraud_volume, 2),
            "action_breakdown": {
                "blocked_count": num_blocked,
                "reviewed_count": num_reviewed,
                "allowed_count": num_allowed,
                "block_rate_pct": round(num_blocked / total_tx * 100, 2),
                "review_rate_pct": round(num_reviewed / total_tx * 100, 2)
            },
            "financial_kpis": {
                "fraud_capture_rate_pct": round(fraud_capture_rate * 100, 2),
                "blocked_fraud_value_inr": round(blocked_value, 2),
                "missed_fraud_loss_inr": round(missed_value, 2),
                "investigation_overhead_cost_inr": round(investigation_cost, 2),
                "total_expected_financial_loss_inr": round(total_financial_loss, 2),
                "cost_per_detected_fraud_inr": round(cost_per_detected_fraud, 2)
            }
        }

if __name__ == "__main__":
    df_sample = pd.DataFrame({
        'amount': [5000, 84500, 1250, 45000, 300],
        'is_fraud': [0, 1, 0, 1, 0],
        'action': ['ALLOW', 'BLOCK', 'ALLOW', 'REVIEW', 'ALLOW']
    })
    tracker = BusinessKPITracker()
    kpis = tracker.calculate_business_kpis(df_sample)
    print("Sample Business KPIs:")
    import json
    print(json.dumps(kpis, indent=2))
