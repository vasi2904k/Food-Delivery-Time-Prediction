"""
Data preprocessing pipeline for the Food Delivery Time Prediction project.

Responsibilities
----------------
* Drop duplicate rows.
* Handle missing values (median for numeric, mode for categorical).
* Encode categorical features with ``OrdinalEncoder`` (preserving a
  consistent mapping across train and inference calls).
* Scale numeric features with ``StandardScaler``.

The :class:`DeliveryPreprocessor` is designed to be fitted on training data
and then applied to unseen data, following the standard scikit-learn
``fit`` / ``transform`` pattern.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
]

NUMERIC_FEATURES = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Vehicle_condition",
    "multiple_deliveries",
    "distance_km",  # added by feature engineering
]

TARGET = "Time_taken(min)"


class DeliveryPreprocessor:
    """Fit-transform preprocessing for the delivery-time dataset.

    Parameters
    ----------
    scale_numeric:
        Whether to apply ``StandardScaler`` to numeric columns.
        Defaults to ``True``.
    """

    def __init__(self, scale_numeric: bool = True) -> None:
        self.scale_numeric = scale_numeric
        self._cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self._scaler = StandardScaler() if scale_numeric else None
        self._cat_medians: dict = {}
        self._num_medians: dict = {}
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"[Preprocessor] Dropped {dropped} duplicate rows.")
        return df

    def _impute(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()
        if fit:
            self._num_medians = {
                col: df[col].median()
                for col in NUMERIC_FEATURES
                if col in df.columns
            }
            self._cat_medians = {
                col: df[col].mode().iloc[0]
                for col in CATEGORICAL_FEATURES
                if col in df.columns
            }
        for col, val in self._num_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        for col, val in self._cat_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "DeliveryPreprocessor":
        """Fit the preprocessor on training data (excluding the target).

        Parameters
        ----------
        df:
            Training DataFrame.  The target column (``Time_taken(min)``)
            may be present but is ignored.

        Returns
        -------
        self
        """
        df = self._drop_duplicates(df)
        df = self._impute(df, fit=True)

        cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        if cat_cols:
            self._cat_encoder.fit(df[cat_cols])

        num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        if self._scaler and num_cols:
            self._scaler.fit(df[num_cols])

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transformations to *df*.

        Parameters
        ----------
        df:
            DataFrame to transform.  Must contain the same feature columns
            as the training data.  The target column is passed through
            unchanged if present.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame ready for model input.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        df = self._drop_duplicates(df)
        df = self._impute(df, fit=False)

        cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        if cat_cols:
            df[cat_cols] = self._cat_encoder.transform(df[cat_cols])

        num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        if self._scaler and num_cols:
            df[num_cols] = self._scaler.transform(df[num_cols])

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in a single call."""
        return self.fit(df).transform(df)
