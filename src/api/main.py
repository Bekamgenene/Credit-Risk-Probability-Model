"""FastAPI service exposing model predictions (loads model from MLflow Model Registry).

This module attempts to load the best model from the MLflow Model Registry at startup
using environment variables. If registry loading fails the service falls back to a
local pickled artifact (useful for local development).
"""

from __future__ import annotations

from pathlib import Path
import os
import logging
import joblib

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import PredictionResponse, Transaction
from src.data_processing import engineer_features

logger = logging.getLogger(__name__)

# Local fallback model path (inside Docker this is /app/artifacts/...)
MODEL_PATH = Path(os.getenv("LOCAL_MODEL_PATH", "/app/artifacts/best_model.pkl"))

# MLflow / Model Registry configuration (set these in your environment / docker-compose)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "credit-risk-best")
MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

app = FastAPI(
    title="Credit Risk Scoring API",
    version="1.0.0",
)


@app.on_event("startup")
def load_model() -> None:
    """
    Load trained model at application startup.

    Logic:
    1. If MLFLOW_TRACKING_URI is provided, attempt to load the model from the MLflow
       Model Registry using the URI: models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}
       (for example: models:/credit-risk-best/Production).
    2. If registry loading fails or MLFLOW_TRACKING_URI is not set, fall back to a
       local pickled model at LOCAL_MODEL_PATH (default: /app/artifacts/best_model.pkl).

    The code prefers the registry model (so you can deploy model updates centrally via MLflow)
    but remains usable for local/dev workflows where a local artifact is available.
    """
    # configure mlflow tracking uri if provided
    if MLFLOW_TRACKING_URI:
        logger.info("Setting MLflow tracking URI to %s", MLFLOW_TRACKING_URI)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # try loading from MLflow Model Registry first
    if MLFLOW_TRACKING_URI:
        model_uri = f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"
        try:
            logger.info("Attempting to load model from registry: %s", model_uri)
            # load as sklearn model to preserve predict_proba when available
            app.state.model = mlflow.sklearn.load_model(model_uri)
            app.state.model_source = model_uri
            logger.info("Loaded model from MLflow Model Registry: %s", model_uri)
            return
        except Exception as exc:  # fallback to local file
            logger.warning(
                "Failed to load model from MLflow registry (%s): %s", model_uri, exc
            )

    # fallback: load a local artifact
    if MODEL_PATH.exists():
        logger.info("Loading local model from %s", MODEL_PATH)
        app.state.model = joblib.load(MODEL_PATH)
        app.state.model_source = str(MODEL_PATH)
        logger.info("Loaded local model from %s", MODEL_PATH)
        return

    # If we reach here we couldn't load a model
    raise RuntimeError(
        "Model not found in MLflow Model Registry or at local path. "
        "Set MLFLOW_TRACKING_URI and ensure the model is registered, or mount a local "
        "model artifact at the LOCAL_MODEL_PATH (default: /app/artifacts/best_model.pkl)."
    )


@app.get("/model-info")
def model_info():
    """Return metadata about the currently loaded model (useful for debugging)."""
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=500, detail="Model not loaded")

    return {"model_source": getattr(app.state, "model_source", "unknown")}


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

    # Prefer predict_proba if available (common for sklearn classifiers).
    try:
        probs = model.predict_proba(X)
        risk_prob = float(probs[:, 1][0])
    except Exception:
        # Some mlflow pyfunc models expose a `predict` that returns probabilities.
        preds = model.predict(X)
        # preds may be (n_samples, 2) or (n_samples,) depending on model flavor
        try:
            if hasattr(preds, "ndim") and preds.ndim == 2:
                risk_prob = float(preds[0, 1])
            else:
                risk_prob = float(preds[0])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {exc}")

    # clamp to [0,1]
    risk_prob = max(0.0, min(1.0, risk_prob))

    return PredictionResponse(risk_probability=risk_prob)
