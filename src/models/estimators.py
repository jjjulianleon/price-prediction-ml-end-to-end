"""Estimator builders shared by experiment and production catalogs."""

from __future__ import annotations

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


def build_dummy_regressor():
    return DummyRegressor(strategy="mean")


def build_sgd_regressor():
    return SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=0.0001,
        learning_rate="invscaling",
        eta0=0.01,
        random_state=42,
    )


def build_ridge_regressor():
    return Ridge(alpha=1.0, random_state=42)


def build_random_forest():
    return RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=20,
        n_jobs=-1,
        random_state=42,
    )


def build_adaboost():
    return AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4, random_state=42),
        n_estimators=150,
        learning_rate=0.05,
        random_state=42,
    )


def build_gradient_boosting():
    return GradientBoostingRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )


def build_hist_gradient_boosting():
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=200,
        min_samples_leaf=50,
        random_state=42,
    )


def build_bagging():
    return BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=8, random_state=42),
        n_estimators=60,
        max_samples=0.7,
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
    )


def build_pasting():
    return BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=8, random_state=42),
        n_estimators=60,
        max_samples=0.7,
        bootstrap=False,
        n_jobs=-1,
        random_state=42,
    )


def build_voting():
    return VotingRegressor(
        estimators=[
            ("rf", build_random_forest()),
            ("gbr", build_gradient_boosting()),
            ("hgb", build_hist_gradient_boosting()),
        ]
    )


def build_xgboost():
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


def build_lightgbm():
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


def build_catboost():
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
