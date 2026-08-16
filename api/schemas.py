"""
Pydantic request/response schemas for the Fraud Intelligence API.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any
from enum import Enum


class TransactionType(str, Enum):
    TRANSFER = "TRANSFER"
    CASH_OUT = "CASH_OUT"
    PAYMENT = "PAYMENT"
    CASH_IN = "CASH_IN"
    DEBIT = "DEBIT"


class TransactionRequest(BaseModel):
    """Single transaction payload for fraud scoring."""

    step: int = Field(default=1, ge=0, description="Simulation step / time index")
    type: TransactionType = Field(description="Transaction type")
    amount: float = Field(gt=0, description="Transaction amount in INR")
    nameOrig: str = Field(min_length=1, description="Originator account ID")
    oldbalanceOrg: float = Field(
        ge=0, description="Originator balance before transaction"
    )
    newbalanceOrig: float = Field(
        ge=0, description="Originator balance after transaction"
    )
    nameDest: str = Field(min_length=1, description="Beneficiary account ID")
    oldbalanceDest: float = Field(
        ge=0, default=0.0, description="Beneficiary balance before transaction"
    )
    newbalanceDest: float = Field(
        ge=0, default=0.0, description="Beneficiary balance after transaction"
    )


class PredictionResponse(BaseModel):
    """ML fraud prediction output."""

    fraud_probability: float
    threshold: float
    is_flagged_fraud: bool
    prediction_metadata: Dict[str, Any]


class RiskComponent(BaseModel):
    """Individual risk score component."""

    ml_probability: float
    ml_contribution: float
    normalized_anomaly_score: float
    anomaly_contribution: float
    anomaly_model_active: bool
    rule_score: float
    rule_contribution: float


class TriggeredRule(BaseModel):
    """A triggered business rule."""

    rule_id: str
    severity: str
    description: str


class RiskScoreResponse(BaseModel):
    """Full risk engine evaluation output."""

    risk_score: float
    risk_tier: str
    action: str
    action_badge: str
    components: RiskComponent
    triggered_rules: List[TriggeredRule]
    engine_config: Dict[str, float]
    prediction: PredictionResponse


class ModelInfoResponse(BaseModel):
    """Model metadata and version information."""

    model_version: str
    feature_schema_version: str
    optimal_threshold: float
    feature_count: int
    calibration_method: str
    risk_engine_weights: Dict[str, float]
    software_versions: Dict[str, str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    anomaly_model_active: bool
    database_connected: bool
