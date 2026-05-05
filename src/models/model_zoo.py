"""Model catalog for NYC Taxi fare comparison pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor

try:  # pragma: no cover - depends on optional dependency
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - handled through model availability metadata
    XGBRegressor = None

try:  # pragma: no cover - depends on optional dependency
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover - handled through model availability metadata
    LGBMRegressor = None

try:  # pragma: no cover - depends on optional dependency
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover - handled through model availability metadata
    CatBoostRegressor = None


@dataclass(frozen=True)
class ModelEntry:
    name: str
    build_estimator: Callable[[], Any]
    training_strategy: str
    description: str
    required: bool = False
    dependency: str | None = None
    rubric_status: str = "optional"


def _dummy_regressor():
    return DummyRegressor(strategy="mean")


def _sgd_regressor():
    return SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=0.0001,
        learning_rate="invscaling",
        eta0=0.01,
        random_state=42,
    )


def _ridge_regressor():
    return Ridge(alpha=1.0, random_state=42)


def _random_forest():
    return RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=42,
    )


def _adaboost():
    return AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4, random_state=42),
        n_estimators=150,
        learning_rate=0.05,
        random_state=42,
    )


def _gradient_boosting():
    return GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )


def _hist_gradient_boosting():
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=200,
        min_samples_leaf=50,
        random_state=42,
    )


def _bagging():
    return BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=8, random_state=42),
        n_estimators=60,
        max_samples=0.7,
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
    )


def _pasting():
    return BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=8, random_state=42),
        n_estimators=60,
        max_samples=0.7,
        bootstrap=False,
        n_jobs=-1,
        random_state=42,
    )


def _voting():
    return VotingRegressor(
        estimators=[
            ("rf", _random_forest()),
            ("gbr", _gradient_boosting()),
            ("hgb", _hist_gradient_boosting()),
        ]
    )


def _xgboost():
    if XGBRegressor is None:  # pragma: no cover - protected by availability checks
        raise ImportError("xgboost is not installed.")
    return XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
        objective="reg:squarederror",
    )


def _lightgbm():
    if LGBMRegressor is None:  # pragma: no cover - protected by availability checks
        raise ImportError("lightgbm is not installed.")
    return LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )


def _catboost():
    if CatBoostRegressor is None:  # pragma: no cover - protected by availability checks
        raise ImportError("catboost is not installed.")
    return CatBoostRegressor(
        iterations=400,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
    )


MODEL_ZOO: list[ModelEntry] = [
    ModelEntry(
        name="dummy_regressor",
        build_estimator=_dummy_regressor,
        training_strategy="sample",
        description="Baseline mean regressor.",
        rubric_status="baseline",
    ),
    ModelEntry(
        name="ridge_regressor",
        build_estimator=_ridge_regressor,
        training_strategy="sample",
        description="Regularized linear baseline over bounded sample.",
        rubric_status="baseline",
    ),
    ModelEntry(
        name="sgd_regressor",
        build_estimator=_sgd_regressor,
        training_strategy="incremental",
        description="True partial_fit baseline over Snowflake batches.",
        rubric_status="baseline",
    ),
    ModelEntry(
        name="random_forest",
        build_estimator=_random_forest,
        training_strategy="sample",
        description="Reference bagging-style tree ensemble on bounded sample.",
        rubric_status="recommended",
    ),
    ModelEntry(
        name="adaboost",
        build_estimator=_adaboost,
        training_strategy="sample",
        description="Required boosting baseline on bounded sample.",
        required=True,
        rubric_status="required",
    ),
    ModelEntry(
        name="gradient_boosting",
        build_estimator=_gradient_boosting,
        training_strategy="sample",
        description="Required boosting baseline on bounded sample.",
        required=True,
        rubric_status="required",
    ),
    ModelEntry(
        name="hist_gradient_boosting",
        build_estimator=_hist_gradient_boosting,
        training_strategy="sample",
        description="Histogram boosting on bounded sample.",
        rubric_status="recommended",
    ),
    ModelEntry(
        name="bagging",
        build_estimator=_bagging,
        training_strategy="sample",
        description="Bagging ensemble with bootstrap sampling.",
        rubric_status="recommended",
    ),
    ModelEntry(
        name="pasting",
        build_estimator=_pasting,
        training_strategy="sample",
        description="Pasting ensemble without bootstrap resampling.",
        rubric_status="recommended",
    ),
    ModelEntry(
        name="voting",
        build_estimator=_voting,
        training_strategy="sample",
        description="Voting ensemble combining heterogeneous regressors.",
        rubric_status="recommended",
    ),
    ModelEntry(
        name="xgboost",
        build_estimator=_xgboost,
        training_strategy="sample",
        description="Required gradient boosting implementation with hist backend.",
        required=True,
        dependency="xgboost",
        rubric_status="required",
    ),
    ModelEntry(
        name="lightgbm",
        build_estimator=_lightgbm,
        training_strategy="sample",
        description="Required LightGBM implementation on bounded sample.",
        required=True,
        dependency="lightgbm",
        rubric_status="required",
    ),
    ModelEntry(
        name="catboost",
        build_estimator=_catboost,
        training_strategy="sample",
        description="Required CatBoost implementation on bounded sample.",
        required=True,
        dependency="catboost",
        rubric_status="required",
    ),
]


def available_model_entries() -> list[ModelEntry]:
    available: list[ModelEntry] = []
    for entry in MODEL_ZOO:
        if entry.dependency == "xgboost" and XGBRegressor is None:
            continue
        if entry.dependency == "lightgbm" and LGBMRegressor is None:
            continue
        if entry.dependency == "catboost" and CatBoostRegressor is None:
            continue
        available.append(entry)
    return available


def unavailable_required_models() -> list[str]:
    missing: list[str] = []
    for entry in MODEL_ZOO:
        if not entry.required:
            continue
        if entry.dependency == "xgboost" and XGBRegressor is None:
            missing.append(entry.name)
        if entry.dependency == "lightgbm" and LGBMRegressor is None:
            missing.append(entry.name)
        if entry.dependency == "catboost" and CatBoostRegressor is None:
            missing.append(entry.name)
    return missing
