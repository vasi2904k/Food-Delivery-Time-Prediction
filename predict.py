"""
CLI prediction interface for the Food Delivery Time Prediction project.

Usage
-----
    python predict.py \\
        --restaurant-lat 12.97 --restaurant-lon 77.59 \\
        --delivery-lat 12.99  --delivery-lon 77.62 \\
        --weather Sunny --traffic High \\
        --vehicle motorcycle --order Meal \\
        --age 28 --rating 4.5 \\
        --condition 2 --multiple-deliveries 1 \\
        --festival No --city Metropolitan

Alternatively, supply a CSV file for batch prediction:

    python predict.py --input records.csv --output predictions.csv
"""

import argparse
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.feature_engineering import engineer_features
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

_MODEL_DIR = "models"
_DEFAULT_MODEL = "random_forest"


def _load_artefacts(model_name: str = _DEFAULT_MODEL):
    model_path = os.path.join(_MODEL_DIR, f"{model_name}_model.joblib")
    preprocessor_path = os.path.join(_MODEL_DIR, "preprocessor.joblib")
    feature_cols_path = os.path.join(_MODEL_DIR, "feature_cols.joblib")

    for p in (model_path, preprocessor_path, feature_cols_path):
        if not os.path.exists(p):
            sys.exit(
                f"[Predict] Artefact not found: {p}\n"
                "Run `python train.py` first to train and save a model."
            )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    feature_cols = joblib.load(feature_cols_path)
    return model, preprocessor, feature_cols


def predict_from_dataframe(
    df: pd.DataFrame,
    model_name: str = _DEFAULT_MODEL,
) -> pd.Series:
    """Run inference on *df* and return predicted delivery times (minutes).

    Parameters
    ----------
    df:
        DataFrame with raw feature columns (same schema as training data).
    model_name:
        Which saved model to load.

    Returns
    -------
    pd.Series
        Predicted delivery times in minutes.
    """
    model, preprocessor, feature_cols = _load_artefacts(model_name)

    df = engineer_features(df)
    df_proc = preprocessor.transform(df)

    # Align to training feature columns
    missing = [c for c in feature_cols if c not in df_proc.columns]
    if missing:
        raise ValueError(f"Missing feature columns after preprocessing: {missing}")

    X = df_proc[feature_cols]
    predictions = model.predict(X)
    return pd.Series(predictions, name="Predicted_Time_taken(min)")


def _single_prediction_df(args: argparse.Namespace) -> pd.DataFrame:
    return pd.DataFrame([{
        "Delivery_person_Age": args.age,
        "Delivery_person_Ratings": args.rating,
        "Restaurant_latitude": args.restaurant_lat,
        "Restaurant_longitude": args.restaurant_lon,
        "Delivery_location_latitude": args.delivery_lat,
        "Delivery_location_longitude": args.delivery_lon,
        "Weatherconditions": args.weather,
        "Road_traffic_density": args.traffic,
        "Vehicle_condition": args.condition,
        "Type_of_order": args.order,
        "Type_of_vehicle": args.vehicle,
        "multiple_deliveries": args.multiple_deliveries,
        "Festival": args.festival,
        "City": args.city,
    }])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict food delivery time using a trained model."
    )
    # Batch mode
    parser.add_argument("--input", help="CSV file with raw records for batch prediction.")
    parser.add_argument("--output", help="Output CSV path for batch predictions.")
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        choices=["linear", "random_forest", "xgboost"],
        help="Which saved model to use (default: %(default)s).",
    )

    # Single-record mode
    parser.add_argument("--restaurant-lat", type=float, default=12.970)
    parser.add_argument("--restaurant-lon", type=float, default=77.590)
    parser.add_argument("--delivery-lat", type=float, default=12.985)
    parser.add_argument("--delivery-lon", type=float, default=77.612)
    parser.add_argument(
        "--weather",
        default="Sunny",
        choices=["Sunny", "Cloudy", "Fog", "Windy", "Sandstorms", "Stormy"],
    )
    parser.add_argument(
        "--traffic",
        default="Medium",
        choices=["Low", "Medium", "High", "Jam"],
    )
    parser.add_argument(
        "--vehicle",
        default="motorcycle",
        choices=["motorcycle", "scooter", "electric_scooter", "bicycle"],
    )
    parser.add_argument(
        "--order",
        default="Meal",
        choices=["Snack", "Meal", "Drinks", "Buffet"],
    )
    parser.add_argument("--age", type=int, default=28)
    parser.add_argument("--rating", type=float, default=4.5)
    parser.add_argument("--condition", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--multiple-deliveries", type=int, default=0)
    parser.add_argument("--festival", default="No", choices=["No", "Yes"])
    parser.add_argument(
        "--city",
        default="Metropolitan",
        choices=["Metropolitan", "Urban", "Semi-Urban"],
    )

    args = parser.parse_args()

    if args.input:
        df = pd.read_csv(args.input)
        preds = predict_from_dataframe(df, model_name=args.model)
        df["Predicted_Time_taken(min)"] = preds.values
        out_path = args.output or args.input.replace(".csv", "_predictions.csv")
        df.to_csv(out_path, index=False)
        print(f"[Predict] Predictions saved to {out_path}")
    else:
        df = _single_prediction_df(args)
        preds = predict_from_dataframe(df, model_name=args.model)
        print(f"[Predict] Estimated delivery time: {preds.iloc[0]:.1f} minutes")


if __name__ == "__main__":
    main()
