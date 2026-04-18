"""
Model training, evaluation, and persistence for the Food Delivery Time
Prediction project.

Three regression models are supported out-of-the-box:

* ``linear``  – ``LinearRegression`` (baseline)
* ``random_forest`` – ``RandomForestRegressor``
* ``xgboost`` – ``XGBRegressor``

The public API is intentionally slim:

* :func:`get_model` – construct a model by name.
* :func:`train_model` – fit a model on feature matrix *X* and target *y*.
* :func:`evaluate_model` – compute RMSE, MAE, and R² on a held-out set.
* :func:`save_model` / :func:`load_model` – persist models with ``joblib``.
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor as _XGBRegressor

    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGBOOST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, Any] = {
    "linear": lambda: LinearRegression(),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    ),
}

if _XGBOOST_AVAILABLE:
    _MODEL_REGISTRY["xgboost"] = lambda: _XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def get_model(name: str = "random_forest") -> Any:
    """Return a fresh (unfitted) model instance by name.

    Parameters
    ----------
    name:
        One of ``"linear"``, ``"random_forest"``, or ``"xgboost"``.

    Returns
    -------
    scikit-learn compatible estimator.

    Raises
    ------
    ValueError
        If *name* is not recognised.
    """
    name = name.lower()
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _MODEL_REGISTRY[name]()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_model(
    model: Any,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
) -> Any:
    """Fit *model* on training data and return it.

    Parameters
    ----------
    model:
        Unfitted scikit-learn compatible estimator.
    X_train:
        Feature matrix.
    y_train:
        Target values.

    Returns
    -------
    Fitted estimator.
    """
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Evaluate *model* on held-out data and return a metrics dict.

    Computed metrics
    ----------------
    * ``rmse`` – Root Mean Squared Error
    * ``mae``  – Mean Absolute Error
    * ``r2``   – Coefficient of determination

    Parameters
    ----------
    model:
        Fitted estimator.
    X_test:
        Feature matrix for the test set.
    y_test:
        True target values.

    Returns
    -------
    dict[str, float]
        ``{"rmse": …, "mae": …, "r2": …}``
    """
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, path: str) -> None:
    """Serialise *model* to *path* using joblib.

    Parameters
    ----------
    model:
        Fitted estimator to save.
    path:
        Destination file path (e.g. ``"models/rf_model.joblib"``).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(model, path)
    print(f"[Model] Saved to {path}")


def load_model(path: str) -> Any:
    """Load and return a model from *path*.

    Parameters
    ----------
    path:
        Path to a joblib-serialised model file.

    Returns
    -------
    Fitted estimator.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
