# Bike-Sharing-Demand

This project predicts the hourly **bike rental demand** based on weather conditions, time, and calendar data.

📊 **Objective:**  
Predict the number of bike rentals (`count`) using features like temperature, humidity, hour of the day, and whether it's a holiday or working day.

🧪 **Evaluation Metric:**  
Root Mean Squared Log Error (**RMSLE**) – the same metric used in the [Kaggle competition](https://www.kaggle.com/competitions/bike-sharing-demand).

---

## 📁 Dataset

- Source: Kaggle – [Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand)
- Files:
  - `train.csv`: Training data with features and target `count`
  - `test.csv`: Test data without target
  - `sampleSubmission.csv`: Required format for submission

---

## 🧠 What I Did

### 1. 🧼 Exploratory Data Analysis (EDA)
- Parsed `datetime` into `hour`, `weekday`, `month`, `year`
- Visualized demand patterns by time, season, and weather
- Identified **data leakage** via `casual` and `registered` → excluded from model

### 2. 🧱 Feature Engineering
Added powerful features to improve model performance:
- `hour_workingday`: hour × working day interaction
- `temp_bin`: binned temperature categories
- `hour_sin`, `hour_cos`: cyclical encoding of hour
- `is_rush_hour`, `is_night`, `is_weekend`: boolean flags

### 3. 🤖 Model Training
Compared multiple models:
| Model                    | RMSLE   |
|-------------------------|---------|
| Linear Regression        | 1.0912  |
| Random Forest (Tuned)    | 0.7260  |
| XGBoost                  | 0.4368  |
| **XGBoost + log1p** 🎯   | **0.2878** |

> ✅ Final model: XGBoost + log-transformed target using `TransformedTargetRegressor`

---

## 🧪 Final Model (XGBoost + log1p)

```python
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
import numpy as np

model = TransformedTargetRegressor(
    regressor=XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),
    func=np.log1p,
    inverse_func=np.expm1
)
