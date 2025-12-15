"""FastAPI service exposing model predictions."""

from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import PredictionResponse, Transaction
from src.data_processing import engineer_features

# IMPORTANT: absolute path inside Docker
MODEL_PATH = Path("/app/artifacts/best_model.pkl")

app = FastAPI(
    title="Credit Risk Scoring API",
    version="1.0.0",
)


@app.on_event("startup")
def load_model() -> None:
    """
    Load trained model at application startup.
    """
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Train the model first and ensure artifacts are mounted."
        )

    app.state.model = joblib.load(MODEL_PATH)


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(tx: Transaction):
    """
    Predict credit risk probability for a single transaction.
    """
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Convert request → DataFrame
    df = pd.DataFrame([tx.dict()])

    # Feature engineering (same logic as training)
    X, _ = engineer_features(df)

    model = app.state.model
    risk_prob = float(model.predict_proba(X)[:, 1][0])

    return PredictionResponse(risk_probability=risk_prob)
