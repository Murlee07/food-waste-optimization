
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# 1. Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "food_sales_cleaned.csv"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load historical sales data
df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df = df.dropna(subset=["date", "food_item", "units_sold"])
df = df.sort_values("date").reset_index(drop=True)

# 3. Create features from the date
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["day_of_year"] = df["date"].dt.dayofyear
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# These features are known before the day begins.
feature_columns = [
    "food_item",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "is_holiday",
]

target_column = "units_sold"

X = df[feature_columns]
y = df[target_column]

# 4. Split by date, not randomly.
# The latest 20% of unique dates are held out for testing.
unique_dates = sorted(df["date"].unique())
split_position = int(len(unique_dates) * 0.8)
split_position = max(1, min(split_position, len(unique_dates) - 1))
test_start_date = unique_dates[split_position]

train_mask = df["date"] < test_start_date
test_mask = df["date"] >= test_start_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]
y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("========== DATA SPLIT ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Test period begins:", pd.Timestamp(test_start_date).date())

# 5. Prepare categorical and numerical features
categorical_features = ["food_item"]
numerical_features = [
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "is_holiday",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        ("numerical", "passthrough", numerical_features),
    ]
)

# 6. Train and compare two models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
}

results = []
trained_pipelines = {}

for name, estimator in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(np.mean((y_test - predictions) ** 2)))

    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
    })

    trained_pipelines[name] = pipeline

# 7. Compare model performance
results_df = pd.DataFrame(results).sort_values("MAE")

print("\n========== MODEL COMPARISON ==========")
print(results_df.round(3).to_string(index=False))

# 8. Select the model with the lowest MAE
best_name = results_df.iloc[0]["model"]
best_model = trained_pipelines[best_name]

model_path = MODEL_DIR / "demand_model.joblib"
joblib.dump(best_model, model_path)

print("\n========== TRAINING COMPLETE ==========")
print("Selected model:", best_name)
print("Saved model to:", model_path)
print("Remember: evaluation uses synthetic practice data.")
