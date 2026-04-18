"""
End-to-end training pipeline for the Food Delivery Time Prediction project.

Usage
-----
    # Train the default model (Random Forest) on synthetic data
    python train.py

    # Train a specific model
    python train.py --model xgboost

    # Train on a custom CSV file
    python train.py --data data/my_dataset.csv --model random_forest

The fitted preprocessor and model are saved under the ``models/`` directory
so they can be loaded by ``predict.py`` for inference.
"""

import argparse
import os
import sys

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.dirname(__file__))

from src.data_generator import generate_dataset
from src.feature_engineering import engineer_features
from src.model import evaluate_model, get_model, save_model, train_model
from src.preprocessing import DeliveryPreprocessor, TARGET

_DEFAULT_DATA_PATH = os.path.join("data", "food_delivery_data.csv")
_MODEL_DIR = "models"


def _load_or_generate_data(data_path: str) -> pd.DataFrame:
    if os.path.exists(data_path):
        print(f"[Train] Loading data from {data_path}")
        return pd.read_csv(data_path)
    print(f"[Train] {data_path} not found – generating synthetic dataset …")
    df = generate_dataset()
    os.makedirs(os.path.dirname(os.path.abspath(data_path)), exist_ok=True)
    df.to_csv(data_path, index=False)
    print(f"[Train] Synthetic data saved to {data_path}")
    return df


def run_training(
    data_path: str = _DEFAULT_DATA_PATH,
    model_name: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Execute the full training pipeline.

    Parameters
    ----------
    data_path:
        Path to the CSV dataset.  If not found, synthetic data is generated.
    model_name:
        One of ``"linear"``, ``"random_forest"``, or ``"xgboost"``.
    test_size:
        Fraction of data held out for evaluation.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{"rmse": …, "mae": …, "r2": …}`` on the test split.
    """
    # 1. Load data
    df = _load_or_generate_data(data_path)

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Split into features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 4. Preprocessing (fit only on train)
    preprocessor = DeliveryPreprocessor(scale_numeric=True)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Keep only the feature columns the preprocessor knows about
    from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    feature_cols = [
        c for c in CATEGORICAL_FEATURES + NUMERIC_FEATURES
        if c in X_train_proc.columns
    ]
    X_train_proc = X_train_proc[feature_cols]
    X_test_proc = X_test_proc[feature_cols]

    # 5. Train model
    print(f"[Train] Training {model_name} model …")
    model = get_model(model_name)
    model = train_model(model, X_train_proc, y_train)

    # 6. Evaluate
    metrics = evaluate_model(model, X_test_proc, y_test)
    print(
        f"[Train] Results on test set:\n"
        f"        RMSE = {metrics['rmse']:.3f} min\n"
        f"        MAE  = {metrics['mae']:.3f} min\n"
        f"        R²   = {metrics['r2']:.4f}"
    )

    # 7. Save artefacts
    os.makedirs(_MODEL_DIR, exist_ok=True)
    save_model(model, os.path.join(_MODEL_DIR, f"{model_name}_model.joblib"))
    preprocessor_path = os.path.join(_MODEL_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"[Train] Preprocessor saved to {preprocessor_path}")

    # Save feature column order for inference
    feature_cols_path = os.path.join(_MODEL_DIR, "feature_cols.joblib")
    joblib.dump(feature_cols, feature_cols_path)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a food delivery time prediction model."
    )
    parser.add_argument(
        "--data",
        default=_DEFAULT_DATA_PATH,
        help="Path to the CSV dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default="random_forest",
        choices=["linear", "random_forest", "xgboost"],
        help="Model to train (default: %(default)s).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for evaluation (default: %(default)s).",
    )
    args = parser.parse_args()
    run_training(
        data_path=args.data,
        model_name=args.model,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
