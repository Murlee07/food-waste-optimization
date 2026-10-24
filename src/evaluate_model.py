
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Project paths
ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "food_sales_cleaned.csv"
MODEL_FILE = ROOT / "models" / "demand_model.joblib"
OUTPUT_FILE = ROOT / "data" / "processed" / "model_predictions.csv"

# Features used when training the model
FEATURES = [
    "food_item",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "is_holiday",
]

TARGET = "units_sold"


def main():
    # 1. Check required files
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run your data generation and analysis scripts first."
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}\n"
            "Run: python src/train_model.py"
        )

    # 2. Load the dataset
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 3. Create date features required by the model
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    # 4. Validate required columns
    required_columns = FEATURES + [TARGET, "date"]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    # 5. Use the latest 20% of dates as the test set
    unique_dates = np.sort(df["date"].dropna().unique())

    if len(unique_dates) < 2:
        raise ValueError(
            "At least two different dates are required for evaluation."
        )

    split_index = int(len(unique_dates) * 0.8)
    split_index = max(1, min(split_index, len(unique_dates) - 1))

    test_dates = unique_dates[split_index:]
    test_df = df[df["date"].isin(test_dates)].copy()

    if test_df.empty:
        raise ValueError("No test records were selected.")

    # 6. Load the trained model
    model = joblib.load(MODEL_FILE)

    # 7. Predict demand
    X_test = test_df[FEATURES]
    actual = test_df[TARGET].astype(float)

    predicted = model.predict(X_test)
    predicted = np.maximum(predicted, 0)

    # 8. Calculate evaluation metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    # 9. Display results
    print("\n" + "=" * 45)
    print("       FOOD DEMAND MODEL EVALUATION")
    print("=" * 45)

    print(f"Total dataset records: {len(df)}")
    print(f"Test records:          {len(test_df)}")
    print(
        f"Test period:           "
        f"{test_df['date'].min().date()} to "
        f"{test_df['date'].max().date()}"
    )

    print("-" * 45)
    print(f"MAE:  {mae:.2f} units")
    print(f"RMSE: {rmse:.2f} units")
    print(f"R²:   {r2:.3f}")
    print("=" * 45)

    # 10. Save predictions for further analysis
    results = test_df[
        ["date", "food_item", TARGET]
    ].copy()

    results["predicted_units_sold"] = predicted
    results["prediction_error"] = (
        results[TARGET] - results["predicted_units_sold"]
    )
    results["absolute_error"] = results["prediction_error"].abs()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"\nPredictions saved successfully to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
