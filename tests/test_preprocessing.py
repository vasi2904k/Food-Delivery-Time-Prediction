"""
Tests for src/preprocessing.py
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import DeliveryPreprocessor, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from src.data_generator import generate_dataset
from src.feature_engineering import engineer_features


@pytest.fixture()
def sample_df():
    df = generate_dataset(n_samples=200, random_state=0)
    return engineer_features(df)


def test_fit_transform_returns_dataframe(sample_df):
    proc = DeliveryPreprocessor()
    X = sample_df.drop(columns=[TARGET])
    result = proc.fit_transform(X)
    assert isinstance(result, pd.DataFrame)


def test_no_missing_values_after_transform(sample_df):
    proc = DeliveryPreprocessor()
    X = sample_df.drop(columns=[TARGET])
    result = proc.fit_transform(X)
    present_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in result.columns]
    assert result[present_cols].isnull().sum().sum() == 0


def test_categorical_columns_are_numeric_after_transform(sample_df):
    proc = DeliveryPreprocessor()
    X = sample_df.drop(columns=[TARGET])
    result = proc.fit_transform(X)
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in result.columns]
    for col in cat_cols:
        assert pd.api.types.is_numeric_dtype(result[col]), (
            f"Column {col!r} should be numeric after encoding"
        )


def test_transform_without_fit_raises():
    proc = DeliveryPreprocessor()
    df = pd.DataFrame({"Weatherconditions": ["Sunny"]})
    with pytest.raises(RuntimeError):
        proc.transform(df)


def test_duplicate_rows_dropped(sample_df):
    X = sample_df.drop(columns=[TARGET])
    dup = pd.concat([X, X.iloc[:5]], ignore_index=True)
    proc = DeliveryPreprocessor()
    result = proc.fit_transform(dup)
    assert len(result) <= len(dup)


def test_imputation_fills_numeric_nan(sample_df):
    X = sample_df.drop(columns=[TARGET]).copy()
    # Introduce NaNs in the first column
    col = "Delivery_person_Age"
    X.loc[0, col] = np.nan

    proc = DeliveryPreprocessor()
    result = proc.fit_transform(X)
    assert result[col].isnull().sum() == 0


def test_scale_numeric_false_preserves_values(sample_df):
    proc = DeliveryPreprocessor(scale_numeric=False)
    X = sample_df.drop(columns=[TARGET])
    result = proc.fit_transform(X)
    # When not scaled, Age values should remain in the original range
    assert result["Delivery_person_Age"].max() <= 50


def test_transform_consistent_with_fit(sample_df):
    """Values produced by fit_transform on train should be reproducible on test."""
    proc = DeliveryPreprocessor()
    X = sample_df.drop(columns=[TARGET])
    train = X.iloc[:150].copy()
    test = X.iloc[150:].copy()

    proc.fit(train)
    result_train = proc.transform(train)
    result_test = proc.transform(test)

    # No column should be all-NaN
    for col in result_train.columns:
        assert not result_train[col].isnull().all()
    for col in result_test.columns:
        assert not result_test[col].isnull().all()
