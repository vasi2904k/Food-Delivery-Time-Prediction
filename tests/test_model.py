"""
Tests for src/model.py
"""

import numpy as np
import pandas as pd
import pytest

from src.model import evaluate_model, get_model, train_model


def _dummy_regression_data(n=100, n_features=5, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_features))
    coef = rng.random(n_features)
    y = X @ coef + rng.normal(0, 0.1, n)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    y_series = pd.Series(y, name="target")
    return X_df, y_series


@pytest.mark.parametrize("name", ["linear", "random_forest"])
def test_get_model_returns_estimator(name):
    model = get_model(name)
    assert hasattr(model, "fit") and hasattr(model, "predict")


def test_get_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("nonexistent_model")


@pytest.mark.parametrize("name", ["linear", "random_forest"])
def test_train_and_predict(name):
    X, y = _dummy_regression_data()
    model = get_model(name)
    fitted = train_model(model, X, y)
    preds = fitted.predict(X)
    assert len(preds) == len(y)


@pytest.mark.parametrize("name", ["linear", "random_forest"])
def test_evaluate_model_returns_expected_keys(name):
    X, y = _dummy_regression_data()
    model = train_model(get_model(name), X, y)
    metrics = evaluate_model(model, X, y)
    assert set(metrics.keys()) == {"rmse", "mae", "r2"}


@pytest.mark.parametrize("name", ["linear", "random_forest"])
def test_evaluate_model_values_are_finite(name):
    X, y = _dummy_regression_data()
    model = train_model(get_model(name), X, y)
    metrics = evaluate_model(model, X, y)
    for key, val in metrics.items():
        assert np.isfinite(val), f"{key} is not finite: {val}"


def test_evaluate_rmse_non_negative():
    X, y = _dummy_regression_data()
    model = train_model(get_model("linear"), X, y)
    metrics = evaluate_model(model, X, y)
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
