"""
FastAPI Application: Digital Payment Fraud Intelligence API.
Provides RESTful endpoints for fraud prediction, risk scoring, and model introspection.
"""

import os
import sys

# Add src and sql to path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sql"))
)

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predict import FraudPredictor
from risk_engine import RiskEngine
from db_manager import get_db_manager
from api.schemas import (
    TransactionRequest,
    PredictionResponse,
    RiskScoreResponse,
    ModelInfoResponse,
    HealthResponse,
)

# Global service instances
predictor = None
risk_engine = None
db_mgr = None
meta = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services at startup."""
    global predictor, risk_engine, db_mgr, meta

    predictor = FraudPredictor()
    risk_engine = RiskEngine()
    db_mgr = get_db_manager()

    meta_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "model_metadata.json"
    )
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    print("[API] Services initialized successfully.")
    yield
    print("[API] Shutting down.")


app = FastAPI(
    title="Digital Payment Fraud Intelligence API",
    description="RESTful API for fraud prediction, risk scoring, and model introspection.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None and predictor.model is not None,
        anomaly_model_active=risk_engine.is_anomaly_active if risk_engine else False,
        database_connected=db_mgr is not None,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Return model metadata, version, and configuration."""
    return ModelInfoResponse(
        model_version=meta.get("model_version", "unknown"),
        feature_schema_version=meta.get("feature_schema_version", "unknown"),
        optimal_threshold=meta.get("optimal_threshold", 0.5),
        feature_count=len(meta.get("feature_cols", [])),
        calibration_method=meta.get("calibration", {}).get("method", "unknown"),
        risk_engine_weights=meta.get(
            "risk_engine_weights", {"w_ml": 0.6, "w_anomaly": 0.2, "w_rules": 0.2}
        ),
        software_versions=meta.get("software_versions", {}),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_fraud(tx: TransactionRequest):
    """Score a single transaction for fraud probability."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    tx_dict = tx.model_dump()
    tx_dict["type"] = tx_dict["type"].value  # Convert enum to string

    result = predictor.predict_single(tx_dict, update_state=True)
    return PredictionResponse(
        fraud_probability=result["fraud_probability"],
        threshold=result["threshold"],
        is_flagged_fraud=result["is_flagged_fraud"],
        prediction_metadata=result["prediction_metadata"],
    )


@app.post("/risk-score", response_model=RiskScoreResponse, tags=["Inference"])
async def evaluate_risk(tx: TransactionRequest):
    """Full risk engine evaluation: ML + Anomaly + Rules → composite score."""
    if predictor is None or risk_engine is None:
        raise HTTPException(status_code=503, detail="Services not loaded")

    import pandas as pd

    tx_dict = tx.model_dump()
    tx_dict["type"] = tx_dict["type"].value

    pred_result = predictor.predict_single(tx_dict, update_state=True)
    ml_prob = pred_result["fraud_probability"]
    feature_row = pd.Series(pred_result["raw_features"])

    risk_result = risk_engine.calculate_risk(ml_prob, feature_row, tx_dict)

    # Log to database
    db_mgr.log_evaluation(tx_dict, risk_result)

    return RiskScoreResponse(
        risk_score=risk_result["risk_score"],
        risk_tier=risk_result["risk_tier"],
        action=risk_result["action"],
        action_badge=risk_result["action_badge"],
        components=risk_result["components"],
        triggered_rules=risk_result["triggered_rules"],
        engine_config=risk_result["engine_config"],
        prediction=PredictionResponse(
            fraud_probability=pred_result["fraud_probability"],
            threshold=pred_result["threshold"],
            is_flagged_fraud=pred_result["is_flagged_fraud"],
            prediction_metadata=pred_result["prediction_metadata"],
        ),
    )


@app.get("/metrics", tags=["Monitoring"])
async def get_prediction_metrics():
    """Return current model performance metrics from metadata."""
    return {
        "test_metrics": meta.get("test_metrics", {}),
        "optimal_threshold": meta.get("optimal_threshold", 0.5),
        "calibration": meta.get("calibration", {}),
        "risk_engine_weights": meta.get("risk_engine_weights", {}),
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
