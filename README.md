# Food Delivery Time Prediction

End-to-end machine learning project to estimate food delivery time using order, traffic, weather, and customer/restaurant features.

This repository contains the full workflow:
1. Data preprocessing and feature engineering
2. Exploratory data analysis (EDA) and visualization
3. Model training, tuning, and evaluation

## Problem Statement

Accurate delivery-time estimation helps food delivery platforms:
- Improve customer trust with better ETAs
- Plan delivery operations more effectively
- Identify key factors that delay deliveries

The objective is to build ML models that predict delivery time and compare their performance.

## Dataset

- Raw data: data/raw/Food_Delivery_Time_Prediction.csv
- Processed data: data/processed/processed_data.csv

The dataset includes features such as:
- Delivery distance
- Weather and traffic conditions
- Delivery person experience
- Order priority and order time
- Vehicle type
- Restaurant/customer rating
- Order cost and tip amount

## Project Structure

```text
Food Delivery Time Prediction/
|-- data/
|   |-- raw/
|   |   `-- Food_Delivery_Time_Prediction.csv
|   `-- processed/
|       `-- processed_data.csv
|-- notebook/
|   |-- 01_data_preprocessing.ipynb
|   |-- 02_EDA_and_visualization.ipynb
|   `-- 03_modelliing.ipynb
|-- outputs/
|   `-- plots/
|-- requirements.txt
`-- README.md
```

## Workflow

### 1) Data Preprocessing

Notebook: notebook/01_data_preprocessing.ipynb

What is done:
- Data loading and basic quality checks (nulls, duplicates, shape)
- Feature selection and transformations
- Categorical encoding (manual mapping + one-hot encoding)
- Feature engineering (for example, rush-hour indicator)
- Scaling/normalization
- Export of cleaned data to the processed folder

### 2) EDA and Visualization

Notebook: notebook/02_EDA_and_visualization.ipynb

What is done:
- Pair plots for relationship exploration
- Boxplots to inspect potential outliers
- Bar/count plots for categorical distributions
- Plot artifacts saved under outputs/plots/

### 3) Modelling and Evaluation

Notebook: notebook/03_modelliing.ipynb

Models included:
- Linear Regression
- Logistic Regression (delivery status classification derived from median delivery time)
- Random Forest Regressor (with GridSearchCV)

Evaluation metrics used:
- Regression: R2, MAE, MSE, RMSE
- Classification: Accuracy, Precision, Recall, F1-score, Confusion Matrix

## Model Score Comparison (Notebook 3)

The scores below are taken from the actual outputs of Notebook 3.

### Regression Models



| Model                   | R2      | MAE     | MSE      | RMSE    |
|-------------------------|--------:|--------:|---------:|--------:|
| Linear Regression       | -0.0323 | 26.5619 | 954.7647 | 30.8993 |
| Random Forest Regressor | 0.7341  | 13.5517 | 245.9528 | 15.6829 |

### Classification Model

| Model               | Accuracy | Precision | Recall | F1     |
|---------------------|---------:|----------:|-------:|-------:|
| Logistic Regression | 0.5000   | 0.5238    | 0.5238 | 0.5238 |

Best logistic hyperparameters (GridSearchCV):
- C: 0.01
- penalty: l2
- solver: lbfgs

Best random forest hyperparameters (GridSearchCV):
- n_estimators: 200
- max_depth: 10
- min_samples_split: 5
- min_samples_leaf: 1

### Linear Regression Details

- Training score: 0.0613
- Testing score: -0.0323

### Logistic Regression Feature Importance

Top features from the logistic regression model:
- distance_calculated
- Weather_Conditions_Rainy
- Delivery_Person_Experience
- Distance
- Vehicle_Type_Bicycle
- Weather_Conditions_Cloudy
- Traffic_Conditions
- Order_Time_Night
- Tip_Amount
- is_peak_hour

### Random Forest Feature Importance

Top features from the random forest regressor:
- Delivery_Status
- Tip_Amount
- Distance
- Restaurant_Rating
- Order_Cost
- distance_calculated
- Customer_Rating
- Delivery_Person_Experience
- Traffic_Conditions
- Order_Priority

### Quick Interpretation

- Random Forest Regressor performed best among the regression models, with a strong R2 of 0.7341 and much lower error values than Linear Regression.
- Linear Regression performed poorly on the test set, with a negative R2 and a very small training score.
- Logistic Regression stayed near chance-level performance at 50% accuracy, which suggests the derived binary label is not easy to separate with a linear classifier.

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook

## Installation

1. Clone the repository
```bash
git clone https://github.com/vasi2904k/Food-Delivery-Time-Prediction.git
cd Food-Delivery-Time-Prediction
```

2. Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## How to Run

Run notebooks in this order:
1. notebook/01_data_preprocessing.ipynb
2. notebook/02_EDA_and_visualization.ipynb
3. notebook/03_modelliing.ipynb

## Key Learnings

- Real-world ETA prediction needs both operational and contextual features
- Proper preprocessing and encoding significantly impact model performance
- Visual EDA helps identify feature behavior before modeling
- Comparing multiple algorithms gives better confidence than using a single model

## Future Improvements

- Add XGBoost/RandomForest Classifier/LightGBM for stronger benchmarking
- Build a simple Streamlit app for interactive prediction

If you are viewing this on GitHub, feel free to explore the notebooks and plots, and share feedback.



