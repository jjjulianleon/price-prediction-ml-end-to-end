"""
Servicio web rápido (FastAPI) para servir el modelo empaquetado.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from src.models.predict_model import load_model, predict
except Exception:
    load_model = None
    predict = None

app = FastAPI(title="API - Predicción de Precios ML", version="1.0")

MODEL: Any = None
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pkl")
USE_MOCK = os.getenv("MOCK_PREDICTION", "1") == "1"


class TripInput(BaseModel):
    """Contrato de entrada sin leakage (no tip_amount, tolls_amount, total_amount)."""

    pickup_datetime: str = Field(..., example="2024-03-10 14:35:00")
    pickup_longitude: float = Field(..., example=-73.9857)
    pickup_latitude: float = Field(..., example=40.7484)
    dropoff_longitude: float = Field(..., example=-73.7769)
    dropoff_latitude: float = Field(..., example=40.6413)
    passenger_count: int = Field(..., ge=1, le=8, example=2)
    trip_distance: float = Field(..., ge=0.0, example=7.8)
    ratecodeid: int = Field(..., ge=1, le=6, example=1)
    payment_type: int = Field(..., ge=1, le=6, example=1)
    vendorid: int = Field(..., ge=1, le=2, example=1)
    store_and_fwd_flag: str = Field(..., example="N")

    class Config:
        schema_extra = {
            "example": {
                "pickup_datetime": "2024-03-10 14:35:00",
                "pickup_longitude": -73.9857,
                "pickup_latitude": 40.7484,
                "dropoff_longitude": -73.7769,
                "dropoff_latitude": 40.6413,
                "passenger_count": 2,
                "trip_distance": 7.8,
                "ratecodeid": 1,
                "payment_type": 1,
                "vendorid": 1,
                "store_and_fwd_flag": "N",
            }
        }


@app.on_event("startup")
def load_artifacts() -> None:
    """Ejecutado al iniciar el servidor FastAPI. Usado para dejar en caché el modelo."""
    global MODEL
    if load_model is None:
        return
    try:
        MODEL = load_model(MODEL_PATH)
    except Exception:
        MODEL = None


@app.get("/health")
def health() -> Dict[str, Any]:
    """Healthcheck básico y estado del modelo."""
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict")
def predict_price(trip: TripInput) -> Dict[str, Any]:
    """Endpoint para predecir total_amount del viaje entrante."""
    if MODEL is None and not USE_MOCK:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    payload = trip.dict()
    input_df = pd.DataFrame([payload])

    if MODEL is None and USE_MOCK:
        estimate = 2.5 + 2.1 * payload["trip_distance"] + 0.5 * payload["passenger_count"]
        return {"estimated_total_amount": round(float(estimate), 2), "model": "mock"}

    if predict is None:
        raise HTTPException(status_code=500, detail="Predict function unavailable.")

    prediction = predict(MODEL, input_df)
    return {"estimated_total_amount": float(prediction[0]), "model": "trained"}
