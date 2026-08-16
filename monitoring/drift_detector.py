"""
Drift Detection Module for Digital Payment Fraud Intelligence System.
Computes Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test statistics
to detect distribution shifts between baseline (training) data and incoming inference data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy.stats import ks_2samp

def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) between baseline and current distributions.
    
    Interpretation:
        - PSI < 0.10: No significant distribution shift (Stable).
        - 0.10 <= PSI < 0.25: Moderate shift; monitor feature closely.
        - PSI >= 0.25: Significant shift; trigger model retraining alert.
    """
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]
    
    if len(baseline) == 0 or len(current) == 0:
        return 0.0
    
    # Create bin boundaries based on baseline quantiles
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(baseline, percentiles)
    bin_edges = np.unique(bin_edges)  # Remove duplicates for constant features
    
    if len(bin_edges) < 2:
        return 0.0
    
    # Compute counts per bin
    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)
    
    # Convert to proportions with smoothing to avoid zero division
    eps = 1e-4
    baseline_pct = (baseline_counts + eps) / (len(baseline) + eps * len(baseline_counts))
    current_pct = (current_counts + eps) / (len(current) + eps * len(current_counts))
    
    # PSI formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)

class DriftDetector:
    def __init__(self, baseline_df: pd.DataFrame = None, feature_cols: List[str] = None):
        self.baseline_df = baseline_df
        self.feature_cols = feature_cols if feature_cols else list(baseline_df.columns) if baseline_df is not None else []
        self.baseline_stats = {}
        if baseline_df is not None:
            self._compute_baseline_stats()

    def _compute_baseline_stats(self):
        """Pre-compute baseline statistics per feature."""
        for col in self.feature_cols:
            if col in self.baseline_df.columns:
                vals = self.baseline_df[col].dropna().values
                self.baseline_stats[col] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "median": float(np.median(vals)),
                    "raw_values": vals
                }

    def detect_feature_drift(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate drift for all feature columns between baseline and current data.
        Returns detailed per-feature drift status and overall alert level.
        """
        results = {}
        max_psi = 0.0
        drifted_features = []
        
        for col in self.feature_cols:
            if col not in current_df.columns or col not in self.baseline_stats:
                continue
            
            base_vals = self.baseline_stats[col]["raw_values"]
            curr_vals = current_df[col].dropna().values
            
            psi = calculate_psi(base_vals, curr_vals)
            ks_stat, p_val = ks_2samp(base_vals, curr_vals) if len(curr_vals) > 0 else (0.0, 1.0)
            
            if psi >= 0.25:
                status = "SIGNIFICANT_DRIFT"
                drifted_features.append(col)
            elif psi >= 0.10:
                status = "MODERATE_DRIFT"
            else:
                status = "STABLE"
                
            max_psi = max(max_psi, psi)
            results[col] = {
                "psi": round(psi, 4),
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_val), 6),
                "status": status,
                "current_mean": round(float(np.mean(curr_vals)), 4) if len(curr_vals) > 0 else 0.0,
                "baseline_mean": self.baseline_stats[col]["mean"]
            }
            
        overall_status = "ALERT" if len(drifted_features) > 0 else "WARNING" if max_psi >= 0.10 else "HEALTHY"
        
        return {
            "overall_status": overall_status,
            "max_psi": round(max_psi, 4),
            "drifted_features_count": len(drifted_features),
            "drifted_features": drifted_features,
            "feature_details": results
        }

if __name__ == "__main__":
    np.random.seed(42)
    base_data = pd.DataFrame({"amount": np.random.normal(5000, 1000, 1000), "velocity": np.random.poisson(3, 1000)})
    shifted_data = pd.DataFrame({"amount": np.random.normal(15000, 5000, 500), "velocity": np.random.poisson(8, 500)})
    
    detector = DriftDetector(base_data)
    drift_report = detector.detect_feature_drift(shifted_data)
    print("Drift Report Summary:")
    print(f"  Overall Status: {drift_report['overall_status']}")
    print(f"  Drifted Features: {drift_report['drifted_features']}")
    print(f"  Amount PSI: {drift_report['feature_details']['amount']['psi']}")
