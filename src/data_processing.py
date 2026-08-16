"""
Data Processing Module for Digital Payment Fraud Intelligence System.
Handles dataset loading, data validation, temporal splitting, and leakage mitigation.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "PS_20174392719_1491204439457_log.csv"
)
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_raw_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Load raw PaySim dataset from CSV path."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found at: {data_path}")
    print(f"Loading raw data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate key statistics and fraud distribution summary."""
    total_records = len(df)
    total_fraud = int(df["isFraud"].sum())
    fraud_rate = float(df["isFraud"].mean() * 100)
    fraud_by_type = (
        df.groupby("type")["isFraud"].agg(["count", "sum", "mean"]).reset_index()
    )
    fraud_by_type.columns = ["type", "total_transactions", "fraud_count", "fraud_rate"]
    fraud_by_type["fraud_rate_pct"] = fraud_by_type["fraud_rate"] * 100

    summary = {
        "total_records": total_records,
        "total_fraud": total_fraud,
        "fraud_rate_pct": round(fraud_rate, 4),
        "fraud_by_type": fraud_by_type.to_dict(orient="records"),
        "null_counts": df.isnull().sum().to_dict(),
        "step_range": (int(df["step"].min()), int(df["step"].max())),
    }
    return summary


def temporal_train_val_test_split(
    df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform strict temporal splitting based on the 'step' column.
    Step represents hour index (1 to 743).
    Ensures zero temporal data leakage (Train on past -> Test on future).
    """
    df_sorted = df.sort_values("step").reset_index(drop=True)
    max_step = df_sorted["step"].max()
    min_step = df_sorted["step"].min()

    train_step_cutoff = min_step + (max_step - min_step) * train_ratio
    val_step_cutoff = min_step + (max_step - min_step) * (train_ratio + val_ratio)

    train_df = df_sorted[df_sorted["step"] <= train_step_cutoff].copy()
    val_df = df_sorted[
        (df_sorted["step"] > train_step_cutoff) & (df_sorted["step"] <= val_step_cutoff)
    ].copy()
    test_df = df_sorted[df_sorted["step"] > val_step_cutoff].copy()

    print(f"Temporal Split Results:")
    print(
        f"  Train Set: {len(train_df):,} rows (Steps {train_df['step'].min()}-{train_df['step'].max()}) | Fraud: {train_df['isFraud'].sum():,} ({train_df['isFraud'].mean()*100:.3f}%)"
    )
    print(
        f"  Val Set:   {len(val_df):,} rows (Steps {val_df['step'].min()}-{val_df['step'].max()}) | Fraud: {val_df['isFraud'].sum():,} ({val_df['isFraud'].mean()*100:.3f}%)"
    )
    print(
        f"  Test Set:  {len(test_df):,} rows (Steps {test_df['step'].min()}-{test_df['step'].max()}) | Fraud: {test_df['isFraud'].sum():,} ({test_df['isFraud'].mean()*100:.3f}%)"
    )

    return train_df, val_df, test_df


if __name__ == "__main__":
    df = load_raw_data()
    summary = get_data_summary(df)
    print("\n--- DATASET SUMMARY ---")
    print(f"Total Transactions: {summary['total_records']:,}")
    print(f"Total Fraud: {summary['total_fraud']:,}")
    print(f"Fraud Rate: {summary['fraud_rate_pct']:.4f}%")
    print("\nFraud Rate by Transaction Type:")
    for row in summary["fraud_by_type"]:
        print(
            f"  {row['type']:10s}: {row['total_transactions']:10,} txs | Fraud: {row['fraud_count']:6,} | Rate: {row['fraud_rate_pct']:.4f}%"
        )
