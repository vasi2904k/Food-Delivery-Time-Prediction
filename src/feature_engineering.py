"""
Feature engineering for the Food Delivery Time Prediction project.

This module adds derived features to the raw DataFrame before the
preprocessing step encodes / scales the data:

* ``distance_km`` – Haversine great-circle distance between the restaurant
  and the delivery location.
* ``order_hour`` – Hour-of-day extracted from an optional ``Order_Date``
  column (dropped afterwards).
* ``is_weekend`` – Binary flag derived from ``Order_Date`` when available.

If the date column is absent the date-related features are simply not added,
so the module works with or without timestamps.
"""

import numpy as np
import pandas as pd


def haversine_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> pd.Series:
    """Return the Haversine great-circle distance in kilometres.

    Parameters
    ----------
    lat1, lon1:
        Origin latitude / longitude (restaurant).
    lat2, lon2:
        Destination latitude / longitude (customer).

    Returns
    -------
    pd.Series
        Distance in kilometres for each row.
    """
    r = 6371.0
    phi1 = np.radians(lat1.values)
    phi2 = np.radians(lat2.values)
    dphi = np.radians(lat2.values - lat1.values)
    dlambda = np.radians(lon2.values - lon1.values)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return pd.Series(2 * r * np.arcsin(np.sqrt(a)), index=lat1.index)


def add_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``distance_km`` column computed from restaurant / delivery coords.

    Parameters
    ----------
    df:
        DataFrame that must contain the four coordinate columns:
        ``Restaurant_latitude``, ``Restaurant_longitude``,
        ``Delivery_location_latitude``, ``Delivery_location_longitude``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the new ``distance_km`` column appended.
    """
    required = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing coordinate columns: {missing}")

    df = df.copy()
    df["distance_km"] = haversine_km(
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"],
    ).clip(lower=0.1)
    return df


def add_time_features(df: pd.DataFrame, date_col: str = "Order_Date") -> pd.DataFrame:
    """Add ``order_hour`` and ``is_weekend`` from an optional date column.

    If *date_col* is not present in *df* the function returns *df* unchanged.

    Parameters
    ----------
    df:
        Input DataFrame.
    date_col:
        Name of the datetime column to derive time features from.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with optional time-based columns added and *date_col*
        dropped.
    """
    if date_col not in df.columns:
        return df

    df = df.copy()
    dt = pd.to_datetime(df[date_col], errors="coerce")
    df["order_hour"] = dt.dt.hour
    df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    df.drop(columns=[date_col], inplace=True)
    return df


def engineer_features(df: pd.DataFrame, date_col: str = "Order_Date") -> pd.DataFrame:
    """Apply all feature-engineering steps in one call.

    Parameters
    ----------
    df:
        Raw input DataFrame.
    date_col:
        Name of the optional datetime column.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with all derived features.
    """
    df = add_distance_feature(df)
    df = add_time_features(df, date_col=date_col)
    return df
