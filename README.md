# Food Delivery Time Prediction

A machine-learning project that predicts food delivery times based on customer
and restaurant GPS coordinates, weather conditions, road-traffic density,
delivery-person attributes, and order details.

---

## Project structure

```
Food-Delivery-Time-Prediction/
├── src/
│   ├── data_generator.py       # Synthetic dataset generation
│   ├── feature_engineering.py  # Distance (Haversine) and time features
│   ├── preprocessing.py        # Cleaning, encoding, and scaling pipeline
│   └── model.py                # Model registry, training, and evaluation
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_model.py
│   └── test_preprocessing.py
├── train.py                    # End-to-end training script
├── predict.py                  # CLI inference interface
└── requirements.txt
```

---

## Features used

| Feature | Description |
|---|---|
| `Delivery_person_Age` | Age of the delivery person |
| `Delivery_person_Ratings` | Rider rating (2.5–5.0) |
| `Restaurant_latitude/longitude` | Restaurant GPS coordinates |
| `Delivery_location_latitude/longitude` | Customer GPS coordinates |
| `distance_km` | Haversine distance (computed automatically) |
| `Weatherconditions` | Sunny / Cloudy / Fog / Windy / Sandstorms / Stormy |
| `Road_traffic_density` | Low / Medium / High / Jam |
| `Vehicle_condition` | 0 (poor) – 3 (excellent) |
| `Type_of_order` | Snack / Meal / Drinks / Buffet |
| `Type_of_vehicle` | motorcycle / scooter / electric_scooter / bicycle |
| `multiple_deliveries` | Number of concurrent deliveries |
| `Festival` | Yes / No |
| `City` | Metropolitan / Urban / Semi-Urban |

**Target:** `Time_taken(min)` – estimated delivery time in minutes.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a model

```bash
# Random Forest (default) – generates synthetic data if CSV not found
python train.py

# XGBoost
python train.py --model xgboost

# Use your own CSV dataset
python train.py --data path/to/your_data.csv --model random_forest
```

Trained artefacts are saved to `models/`.

### 3. Predict delivery time

**Single record:**

```bash
python predict.py \
    --restaurant-lat 12.97 --restaurant-lon 77.59 \
    --delivery-lat   12.99 --delivery-lon   77.62 \
    --weather Sunny --traffic High \
    --vehicle motorcycle --order Meal \
    --age 28 --rating 4.5 \
    --condition 2 --multiple-deliveries 1 \
    --festival No --city Metropolitan
```

**Batch prediction from CSV:**

```bash
python predict.py --input records.csv --output predictions.csv
```

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Models supported

| Name | Class |
|---|---|
| `linear` | `LinearRegression` (baseline) |
| `random_forest` | `RandomForestRegressor` |
| `xgboost` | `XGBRegressor` |

---

## Evaluation metrics

* **RMSE** – Root Mean Squared Error (minutes)
* **MAE** – Mean Absolute Error (minutes)
* **R²** – Coefficient of determination
