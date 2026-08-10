"""
Unit tests for Risk Engine.
Validates risk tier classification, action directives, and score normalization.
"""

import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from risk_engine import RiskEngine

def _make_feature_row(**overrides):
    """Create a feature row Series with defaults."""
    defaults = {
        'log_amount': 10.0,
        'hour_of_day': 14,
        'day_of_week': 2,
        'is_night_time': 0,
        'type_TRANSFER': 1,
        'type_CASH_OUT': 0,
        'type_PAYMENT': 0,
        'type_CASH_IN': 0,
        'type_DEBIT': 0,
        'is_orig_customer': 1,
        'is_dest_customer': 1,
        'is_dest_merchant': 0,
        'orig_balance_err': 0.0,
        'dest_balance_err': 0.0,
        'is_zero_orig_balance': 0,
        'is_zero_dest_balance': 0,
        'amount_to_orig_prior_mean_ratio': 1.0,
        'transactions_last_1h': 1,
        'transactions_last_6h': 2,
        'transactions_last_24h': 3,
        'is_new_beneficiary': 0,
        'amount_velocity_6h': 5000.0
    }
    defaults.update(overrides)
    return pd.Series(defaults)

def _make_tx(**overrides):
    """Create a transaction dict with defaults."""
    defaults = {
        'amount': 5000.0,
        'type': 'TRANSFER',
        'oldbalanceOrg': 50000.0,
        'newbalanceOrig': 45000.0
    }
    defaults.update(overrides)
    return defaults

class TestRiskTierClassification:
    """Verify risk score → tier → action mapping."""

    def test_critical_tier_blocks(self):
        """Very high ML probability + multiple rule triggers should produce CRITICAL tier."""
        engine = RiskEngine()
        # ml_component = 0.99 * 100 * 0.60 = 59.4
        # anomaly_component = 0.20 * 100 * 0.20 = 4.0 (unfitted default)
        # rule triggers: velocity(35) + new_ben_high_val(30) + night(25) + drain(20) = 110 → capped at 100
        # rules_component = 100 * 0.20 = 20.0
        # total = 59.4 + 4.0 + 20.0 = 83.4 → CRITICAL
        result = engine.calculate_risk(
            ml_prob=0.99,
            feature_row=_make_feature_row(
                is_new_beneficiary=1, transactions_last_1h=6,
                transactions_last_6h=12, hour_of_day=2, is_night_time=1,
                amount_to_orig_prior_mean_ratio=15.0
            ),
            tx_dict=_make_tx(amount=84500.0, type='TRANSFER', oldbalanceOrg=84500.0, newbalanceOrig=0.0)
        )
        assert result['risk_tier'] == "CRITICAL"
        assert result['action'] == "BLOCK"

    def test_low_tier_allows(self):
        """Low ML probability with no rule triggers should produce LOW tier with ALLOW."""
        engine = RiskEngine()
        result = engine.calculate_risk(
            ml_prob=0.02,
            feature_row=_make_feature_row(),
            tx_dict=_make_tx(amount=500.0)
        )
        assert result['risk_tier'] == "LOW"
        assert result['action'] == "ALLOW"

    def test_medium_tier_reviews(self):
        """Moderate ML probability with some rule triggers should produce MEDIUM or HIGH."""
        engine = RiskEngine()
        # ml_component = 0.55 * 100 * 0.60 = 33.0
        # anomaly = 4.0 (default)
        # rules: no triggers at amount=5000 with defaults → 0
        # total ≈ 37.0 → MEDIUM
        result = engine.calculate_risk(
            ml_prob=0.55,
            feature_row=_make_feature_row(),
            tx_dict=_make_tx(amount=5000.0)
        )
        assert result['risk_tier'] in ("MEDIUM", "HIGH")
        assert result['action'] == "REVIEW"

class TestRiskScoreNormalization:
    """Verify composite score stays within [0, 100]."""

    def test_score_bounds_extreme_high(self):
        engine = RiskEngine()
        result = engine.calculate_risk(
            ml_prob=1.0,
            feature_row=_make_feature_row(
                is_new_beneficiary=1, transactions_last_1h=10,
                transactions_last_6h=20, amount_to_orig_prior_mean_ratio=50.0,
                hour_of_day=2, is_night_time=1
            ),
            tx_dict=_make_tx(amount=100000.0, oldbalanceOrg=100000.0, newbalanceOrig=0.0)
        )
        assert 0.0 <= result['risk_score'] <= 100.0

    def test_score_bounds_extreme_low(self):
        engine = RiskEngine()
        result = engine.calculate_risk(
            ml_prob=0.0,
            feature_row=_make_feature_row(),
            tx_dict=_make_tx(amount=100.0)
        )
        assert 0.0 <= result['risk_score'] <= 100.0

class TestActionWording:
    """Verify action badges use correct wording (no account freeze, no OTP hardcoding)."""

    def test_critical_no_account_freeze(self):
        engine = RiskEngine()
        # Force CRITICAL score with max signals
        result = engine.calculate_risk(
            ml_prob=0.99,
            feature_row=_make_feature_row(
                is_new_beneficiary=1, transactions_last_1h=8,
                transactions_last_6h=15, hour_of_day=2, is_night_time=1,
                amount_to_orig_prior_mean_ratio=20.0
            ),
            tx_dict=_make_tx(amount=90000.0, type='TRANSFER', oldbalanceOrg=90000.0, newbalanceOrig=0.0)
        )
        assert "freeze" not in result['action_badge'].lower(), \
            "Action badge should not claim 'account freeze'"
        assert "ESCALATE" in result['action_badge'] or "INVESTIGATION" in result['action_badge'], \
            "CRITICAL action should mention escalation or investigation"

    def test_medium_no_otp_hardcode(self):
        engine = RiskEngine()
        result = engine.calculate_risk(
            ml_prob=0.40,
            feature_row=_make_feature_row(),
            tx_dict=_make_tx(amount=5000.0)
        )
        assert "OTP" not in result['action_badge'], \
            "Action badge should not hardcode OTP as the specific auth method"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
