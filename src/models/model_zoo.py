"""Model catalog for NYC Taxi fare comparison pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from xgboost import XGBRegressor


@dataclass
class ModelEntry:
    name: str
    estimator: Any
    needs_incremental: bool = False  # True = uses partial_fit (SGD only)
    sample_only: bool = True         # True = train on sample, not out-of-core


MODEL_ZOO: list[ModelEntry] = [
    ModelEntry(
        name="dummy_regressor",
        estimator=DummyRegressor(strategy="mean"),
        needs_incremental=False,
        sample_only=True,
    ),
    ModelEntry(
        name="sgd_regressor",
        estimator=SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=0.0001,
            learning_rate="invscaling",
            eta0=0.01,
            random_state=42,
        ),
        needs_incremental=True,
        sample_only=False,
    ),
    ModelEntry(
        name="random_forest",
        estimator=RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=42,
        ),
        needs_incremental=False,
        sample_only=True,
    ),
    ModelEntry(
        name="xgboost",
        estimator=XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            tree_method="hist",
        ),
        needs_incremental=False,
        sample_only=True,
    ),
    ModelEntry(
        name="lightgbm",
        estimator=LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        ),
        needs_incremental=False,
        sample_only=True,
    ),
    ModelEntry(
        name="hist_gradient_boosting",
        estimator=HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=150,
            min_samples_leaf=50,
            random_state=42,
        ),
        needs_incremental=False,
        sample_only=True,
    ),
]
