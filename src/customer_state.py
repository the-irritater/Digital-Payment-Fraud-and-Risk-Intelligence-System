"""
Customer State Store for Real-Time Payment Risk Intelligence.
Maintains stateful historical baselines (prior amounts, timestamps, known beneficiaries)
per customer account so single-transaction inference computes genuine velocity and behavioral features.
"""

from typing import Dict, Any, List, Set, Tuple
import pandas as pd
import numpy as np


class CustomerStateStore:
    def __init__(self):
        # Maps customer_id -> dict of state attributes
        self._state: Dict[str, Dict[str, Any]] = {}

    def get_or_create_customer(self, customer_id: str) -> Dict[str, Any]:
        if customer_id not in self._state:
            self._state[customer_id] = {
                "tx_history": [],  # List of (step, amount)
                "known_beneficiaries": set(),  # Set of destination account IDs
                "total_amount": 0.0,
                "count": 0,
            }
        return self._state[customer_id]

    def seed_from_dataframe(self, df: pd.DataFrame):
        """Seed state store from historical transaction log."""
        print(
            f"[CustomerStateStore] Seeding customer state from {len(df):,} historical transactions..."
        )
        df_sorted = df.sort_values("step")
        for _, row in df_sorted.iterrows():
            cid = str(row["nameOrig"])
            dest = str(row["nameDest"])
            step = float(row["step"])
            amt = float(row["amount"])

            state = self.get_or_create_customer(cid)
            state["tx_history"].append((step, amt))
            state["known_beneficiaries"].add(dest)
            state["total_amount"] += amt
            state["count"] += 1

    def compute_realtime_features(self, tx_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Given a single raw transaction payload, fetch the customer's state
        and compute genuine historical velocity, prior mean ratio, and beneficiary flags.
        """
        cid = str(tx_dict.get("nameOrig", ""))
        dest = str(tx_dict.get("nameDest", ""))
        step = float(tx_dict.get("step", 1))
        amt = float(tx_dict.get("amount", 0.0))

        state = self.get_or_create_customer(cid)
        history = state["tx_history"]

        # 1. Prior expanding mean ratio
        if state["count"] > 0:
            prior_mean = state["total_amount"] / state["count"]
            ratio = amt / prior_mean if prior_mean > 0 else 1.0
        else:
            ratio = 1.0
        ratio = float(np.clip(ratio, 0.0, 100.0))

        # 2. Velocity counts in prior 1h, 6h, 24h
        steps_history = [s for s, a in history if s <= step]
        amounts_history = [a for s, a in history if s <= step]

        count_1h = sum(1 for s in steps_history if (step - 1) <= s <= step) + 1
        count_6h = sum(1 for s in steps_history if (step - 6) <= s <= step) + 1
        count_24h = sum(1 for s in steps_history if (step - 24) <= s <= step) + 1

        amt_vel_6h = (
            sum(
                a
                for s, a in zip(steps_history, amounts_history)
                if (step - 6) <= s <= step
            )
            + amt
        )

        # 3. Is new beneficiary
        is_new_ben = 1 if dest not in state["known_beneficiaries"] else 0

        return {
            "amount_to_orig_prior_mean_ratio": ratio,
            "transactions_last_1h": count_1h,
            "transactions_last_6h": count_6h,
            "transactions_last_24h": count_24h,
            "amount_velocity_6h": amt_vel_6h,
            "is_new_beneficiary": is_new_ben,
        }

    def update_state(self, tx_dict: Dict[str, Any]):
        """Update customer state after evaluating transaction."""
        cid = str(tx_dict.get("nameOrig", ""))
        dest = str(tx_dict.get("nameDest", ""))
        step = float(tx_dict.get("step", 1))
        amt = float(tx_dict.get("amount", 0.0))

        state = self.get_or_create_customer(cid)
        state["tx_history"].append((step, amt))
        state["known_beneficiaries"].add(dest)
        state["total_amount"] += amt
        state["count"] += 1


# Singleton state store instance
_global_state_store = CustomerStateStore()


def get_state_store() -> CustomerStateStore:
    return _global_state_store
