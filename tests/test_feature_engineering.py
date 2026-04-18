"""
Tests for src/feature_engineering.py
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_distance_feature,
    add_time_features,
    engineer_features,
    haversine_km,
)


def _base_df(n=10, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Restaurant_latitude": 12.97 + rng.uniform(-0.1, 0.1, n),
        "Restaurant_longitude": 77.59 + rng.uniform(-0.1, 0.1, n),
        "Delivery_location_latitude": 12.97 + rng.uniform(-0.1, 0.1, n),
        "Delivery_location_longitude": 77.59 + rng.uniform(-0.1, 0.1, n),
    })


def test_haversine_same_point_is_zero():
    lat = pd.Series([12.97])
    lon = pd.Series([77.59])
    dist = haversine_km(lat, lon, lat, lon)
    assert dist.iloc[0] == pytest.approx(0.0, abs=1e-6)


def test_haversine_known_distance():
    # Approximately 111 km per degree latitude
    lat1 = pd.Series([0.0])
    lon1 = pd.Series([0.0])
    lat2 = pd.Series([1.0])
    lon2 = pd.Series([0.0])
    dist = haversine_km(lat1, lon1, lat2, lon2)
    assert dist.iloc[0] == pytest.approx(111.195, abs=0.5)


def test_add_distance_feature_adds_column():
    df = _base_df()
    result = add_distance_feature(df)
    assert "distance_km" in result.columns


def test_add_distance_feature_non_negative():
    df = _base_df(n=50)
    result = add_distance_feature(df)
    assert (result["distance_km"] >= 0).all()


def test_add_distance_feature_missing_cols_raises():
    df = pd.DataFrame({"Restaurant_latitude": [12.97]})
    with pytest.raises(ValueError, match="Missing coordinate columns"):
        add_distance_feature(df)


def test_add_distance_feature_does_not_mutate_input():
    df = _base_df()
    original_cols = set(df.columns)
    _ = add_distance_feature(df)
    assert set(df.columns) == original_cols


def test_add_time_features_no_date_col_unchanged():
    df = _base_df()
    result = add_time_features(df, date_col="Order_Date")
    pd.testing.assert_frame_equal(df, result)


def test_add_time_features_extracts_hour_and_weekend():
    df = pd.DataFrame({
        "Order_Date": ["2024-01-06 14:30:00", "2024-01-07 09:00:00"],  # Sat, Sun
        "value": [1, 2],
    })
    result = add_time_features(df, date_col="Order_Date")
    assert "order_hour" in result.columns
    assert "is_weekend" in result.columns
    assert "Order_Date" not in result.columns
    assert list(result["order_hour"]) == [14, 9]
    assert list(result["is_weekend"]) == [1, 1]  # Both weekend


def test_engineer_features_end_to_end():
    from src.data_generator import generate_dataset
    df = generate_dataset(n_samples=50, random_state=1)
    result = engineer_features(df)
    assert "distance_km" in result.columns
    assert len(result) == 50
