
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "model_predictions.csv"
FIGURE_DIR = ROOT / "reports" / "figures"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {DATA_FILE}\n"
            "Run: python src/evaluate_model.py"
        )

    df = pd.read_csv(DATA_FILE, parse_dates=["date"])

    required_columns = {
        "date",
        "food_item",
        "units_sold",
        "predicted_units_sold",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in prediction file: {missing}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Chart 1: Actual vs predicted demand over time
    daily = (
        df.groupby("date", as_index=False)
        .agg(
            actual_demand=("units_sold", "sum"),
            predicted_demand=("predicted_units_sold", "sum"),
        )
        .sort_values("date")
    )

    plt.figure(figsize=(12, 6))
    plt.plot(
        daily["date"],
        daily["actual_demand"],
        label="Actual demand",
    )
    plt.plot(
        daily["date"],
        daily["predicted_demand"],
        label="Predicted demand",
    )
    plt.title("Actual vs Predicted Daily Food Demand")
    plt.xlabel("Date")
    plt.ylabel("Units sold")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    chart1 = FIGURE_DIR / "actual_vs_predicted.png"
    plt.savefig(chart1, dpi=150)
    plt.close()

    # Chart 2: Prediction error by food item
    df["prediction_error"] = (
        df["units_sold"] - df["predicted_units_sold"]
    )

    error_summary = (
        df.groupby("food_item")["prediction_error"]
        .apply(lambda errors: errors.abs().mean())
        .sort_values()
    )

    plt.figure(figsize=(10, 6))
    error_summary.plot(kind="bar")
    plt.title("Mean Absolute Prediction Error by Food Item")
    plt.xlabel("Food item")
    plt.ylabel("Mean absolute error (units)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    chart2 = FIGURE_DIR / "prediction_error_by_item.png"
    plt.savefig(chart2, dpi=150)
    plt.close()

    print("Visualization completed successfully!")
    print(f"Chart 1 saved to: {chart1}")
    print(f"Chart 2 saved to: {chart2}")


if __name__ == "__main__":
    main()
