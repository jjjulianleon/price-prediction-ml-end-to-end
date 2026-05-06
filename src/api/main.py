"""FastAPI service for NYC Taxi fare prediction."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict_model import load_model, predict
from src.utils.config import get_settings

app = FastAPI(title="NYC Taxi Fare Prediction API", version="2.0")

MODEL: Any = None
MODEL_NAME: str = "unknown"


def _find_production_model() -> Path | None:
    settings = get_settings(validate=False)
    artifact_path = settings.production_artifact_path
    if artifact_path.exists():
        return artifact_path
    return None


class TripInput(BaseModel):
    """Datos de entrada del viaje — sin leakage, sin coordenadas."""

    trip_type: Literal["yellow", "green"] = Field(..., example="yellow")
    pickup_datetime: str = Field(..., example="2025-01-15 14:35:00")
    pickup_location_id: int = Field(..., ge=1, le=265, example=237)
    dropoff_location_id: int = Field(..., ge=1, le=265, example=141)
    passenger_count: int = Field(..., ge=1, le=8, example=2)
    estimated_distance: float = Field(..., ge=0.0, example=7.8)
    vendor_id: int = Field(..., ge=1, le=2, example=1)
    ratecode_id: int = Field(..., ge=1, le=6, example=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "trip_type": "yellow",
                "pickup_datetime": "2025-01-15 14:35:00",
                "pickup_location_id": 237,
                "dropoff_location_id": 141,
                "passenger_count": 2,
                "estimated_distance": 7.8,
                "vendor_id": 1,
                "ratecode_id": 1,
            }
        }
    }


@app.on_event("startup")
def load_artifacts() -> None:
    global MODEL, MODEL_NAME
    model_path = _find_production_model()
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
    payload["trip_type"] = str(payload["trip_type"]).lower()
    input_df = pd.DataFrame([payload])

    prediction = predict(MODEL, input_df)
    return {
        "estimated_fare_amount": round(float(prediction[0]), 2),
        "model": MODEL_NAME,
    }
