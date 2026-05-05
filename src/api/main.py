"""FastAPI service for NYC Taxi fare prediction."""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict_model import load_model, predict

app = FastAPI(title="NYC Taxi Fare Prediction API", version="2.0")

MODEL: Any = None
MODEL_NAME: str = "unknown"


def _find_latest_model() -> Path | None:
    model_dir = Path(os.getenv("MODEL_DIR", "data/models"))
    candidates = list(model_dir.glob("*.joblib"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


class TripInput(BaseModel):
    """Datos de entrada del viaje — sin leakage, sin coordenadas."""

    pickup_datetime: str = Field(..., example="2025-01-15 14:35:00")
    pickup_location_id: int = Field(..., ge=1, le=265, example=237)
    dropoff_location_id: int = Field(..., ge=1, le=265, example=141)
    passenger_count: int = Field(..., ge=1, le=8, example=2)
    trip_distance: float = Field(..., ge=0.0, example=7.8)
    vendor_id: int = Field(..., ge=1, le=2, example=1)
    ratecode_id: int = Field(..., ge=1, le=6, example=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "pickup_datetime": "2025-01-15 14:35:00",
                "pickup_location_id": 237,
                "dropoff_location_id": 141,
                "passenger_count": 2,
                "trip_distance": 7.8,
                "vendor_id": 1,
                "ratecode_id": 1,
            }
        }
    }


@app.on_event("startup")
def load_artifacts() -> None:
    global MODEL, MODEL_NAME
    model_path = _find_latest_model()
    if model_path is None:
        return
    try:
        MODEL = load_model(model_path)
        MODEL_NAME = MODEL.get("model_name", "unknown") if isinstance(MODEL, dict) else "unknown"
    except Exception:
        MODEL = None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "model_loaded": MODEL is not None, "model_name": MODEL_NAME}


@app.post("/predict")
def predict_price(trip: TripInput) -> Dict[str, Any]:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    payload = trip.model_dump()
    input_df = pd.DataFrame([payload])

    prediction = predict(MODEL, input_df)
    return {
        "estimated_fare_amount": round(float(prediction[0]), 2),
        "model": MODEL_NAME,
    }
