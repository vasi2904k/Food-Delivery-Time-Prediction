"""
Synthetic data generator for the Food Delivery Time Prediction dataset.

The generated dataset mimics real-world delivery records with features such as
delivery-person attributes, restaurant/customer GPS coordinates, weather,
traffic, order type, and vehicle type.  The target variable
``Time_taken(min)`` is derived deterministically from these features so that
a trained model can learn meaningful relationships.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEATHER_CONDITIONS = ["Sunny", "Cloudy", "Fog", "Windy", "Sandstorms", "Stormy"]
TRAFFIC_DENSITIES = ["Low", "Medium", "High", "Jam"]
ORDER_TYPES = ["Snack", "Meal", "Drinks", "Buffet"]
VEHICLE_TYPES = ["motorcycle", "scooter", "electric_scooter", "bicycle"]
CITIES = ["Metropolitan", "Urban", "Semi-Urban"]
FESTIVALS = ["No", "Yes"]

# Base coordinates: Bangalore, India
_BASE_LAT = 12.97
_BASE_LON = 77.59
_COORD_SPREAD = 0.15  # ~16 km radius

# Additive delivery-time contributions (minutes)
_TRAFFIC_DELAY = {"Low": 0, "Medium": 5, "High": 12, "Jam": 22}
_WEATHER_DELAY = {
    "Sunny": 0,
    "Cloudy": 2,
    "Fog": 6,
    "Windy": 4,
    "Sandstorms": 10,
    "Stormy": 15,
}
_VEHICLE_SPEED = {
    "motorcycle": 30,
    "scooter": 25,
    "electric_scooter": 20,
    "bicycle": 12,
}  # km/h


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                  lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorised Haversine distance in kilometres."""
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def generate_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic food-delivery dataset.

    Parameters
    ----------
    n_samples:
        Number of rows to generate.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all features and the target column
        ``Time_taken(min)``.
    """
    rng = np.random.default_rng(random_state)

    # --- Delivery-person attributes ---
    delivery_person_age = rng.integers(20, 46, n_samples)
    delivery_person_ratings = np.round(rng.uniform(2.5, 5.0, n_samples), 1)

    # --- Locations ---
    restaurant_lat = _BASE_LAT + rng.uniform(-_COORD_SPREAD, _COORD_SPREAD, n_samples)
    restaurant_lon = _BASE_LON + rng.uniform(-_COORD_SPREAD, _COORD_SPREAD, n_samples)
    delivery_lat = _BASE_LAT + rng.uniform(-_COORD_SPREAD, _COORD_SPREAD, n_samples)
    delivery_lon = _BASE_LON + rng.uniform(-_COORD_SPREAD, _COORD_SPREAD, n_samples)

    # --- Categorical features ---
    weather = rng.choice(WEATHER_CONDITIONS, n_samples)
    traffic = rng.choice(TRAFFIC_DENSITIES, n_samples)
    order_type = rng.choice(ORDER_TYPES, n_samples)
    vehicle_type = rng.choice(VEHICLE_TYPES, n_samples)
    city = rng.choice(CITIES, n_samples)
    festival = rng.choice(FESTIVALS, n_samples, p=[0.9, 0.1])

    # --- Numeric operational features ---
    vehicle_condition = rng.integers(0, 4, n_samples)  # 0=poor … 3=excellent
    multiple_deliveries = rng.integers(0, 4, n_samples)

    # --- Derived features ---
    distance_km = _haversine_km(restaurant_lat, restaurant_lon,
                                delivery_lat, delivery_lon)
    # Ensure minimum distance of 0.5 km
    distance_km = np.maximum(distance_km, 0.5)

    # --- Target: Time_taken(min) ---
    speed = np.array([_VEHICLE_SPEED[v] for v in vehicle_type])
    travel_time = (distance_km / speed) * 60  # minutes

    traffic_delay = np.array([_TRAFFIC_DELAY[t] for t in traffic])
    weather_delay = np.array([_WEATHER_DELAY[w] for w in weather])

    # Poor vehicle condition adds delay
    condition_delay = (3 - vehicle_condition) * 1.5

    # Each extra delivery adds ~8 minutes
    multi_delay = multiple_deliveries * 8

    # Festival adds congestion
    festival_delay = np.where(np.array(festival) == "Yes", 10, 0)

    # Low rating → slower (higher ratings → faster, more motivated delivery)
    rating_factor = 1.0 + (4.0 - delivery_person_ratings) * 0.05

    # Base restaurant preparation time (5–15 min)
    prep_time = rng.uniform(5, 15, n_samples)

    time_taken = (
        prep_time
        + travel_time * rating_factor
        + traffic_delay
        + weather_delay
        + condition_delay
        + multi_delay
        + festival_delay
        + rng.normal(0, 2, n_samples)  # small random noise
    ).clip(5, 120)

    time_taken = np.round(time_taken).astype(int)

    df = pd.DataFrame({
        "Delivery_person_Age": delivery_person_age,
        "Delivery_person_Ratings": delivery_person_ratings,
        "Restaurant_latitude": restaurant_lat.round(6),
        "Restaurant_longitude": restaurant_lon.round(6),
        "Delivery_location_latitude": delivery_lat.round(6),
        "Delivery_location_longitude": delivery_lon.round(6),
        "Weatherconditions": weather,
        "Road_traffic_density": traffic,
        "Vehicle_condition": vehicle_condition,
        "Type_of_order": order_type,
        "Type_of_vehicle": vehicle_type,
        "multiple_deliveries": multiple_deliveries,
        "Festival": festival,
        "City": city,
        "Time_taken(min)": time_taken,
    })

    return df


if __name__ == "__main__":
    import os

    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", "food_delivery_data.csv"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = generate_dataset()
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path} ({len(df)} rows).")
