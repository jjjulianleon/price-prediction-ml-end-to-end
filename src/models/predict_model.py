"""Inference helpers for saved model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.features.build_features import DISTANCE_COLUMN, LEGACY_DISTANCE_COLUMN, select_raw_feature_columns


def load_model(model_path: str | Path) -> Any:
    return joblib.load(model_path)


def predict(model_artifact: Any, input_data: pd.DataFrame) -> list[float]:
    if DISTANCE_COLUMN not in input_data.columns and LEGACY_DISTANCE_COLUMN in input_data.columns:
        input_data = input_data.rename(columns={LEGACY_DISTANCE_COLUMN: DISTANCE_COLUMN})

    if isinstance(model_artifact, dict) and {"model", "preprocessor"}.issubset(model_artifact):
        X = select_raw_feature_columns(input_data)
        transformed = model_artifact["preprocessor"].transform(X)
        predictions = model_artifact["model"].predict(transformed)
        return [float(value) for value in predictions]

    if hasattr(model_artifact, "predict"):
        predictions = model_artifact.predict(input_data)
        return [float(value) for value in predictions]

    raise TypeError("Unsupported model artifact format.")
