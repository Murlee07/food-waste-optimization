
from datetime import date
from pathlib import Path
from statistics import NormalDist

import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "food_sales_cleaned.csv"
MODEL_FILE = ROOT / "models" / "demand_model.joblib"


# --------------------------------------------------
# 2. Food preparation optimization
# --------------------------------------------------

def recommend_preparation(
    predicted_demand,
    demand_std,
    waste_cost,
    shortage_cost,
):
    """
    Recommend preparation quantity using a normal
    approximation and the newsvendor cost principle.

    waste_cost:
        Cost of one unused prepared portion.

    shortage_cost:
        Cost of one unmet customer portion.
    """

    if waste_cost <= 0 or shortage_cost <= 0:
        raise ValueError("Both costs must be greater than zero.")

    if predicted_demand < 0 or demand_std < 0:
        raise ValueError("Demand and standard deviation cannot be negative.")

    # Critical fractile: balance shortage and waste costs.
    critical_fractile = shortage_cost / (shortage_cost + waste_cost)

    # Avoid infinite quantiles at the distribution boundaries.
    critical_fractile = min(max(critical_fractile, 0.001), 0.999)

    z_score = NormalDist().inv_cdf(critical_fractile)

    # Approximate the demand quantile to prepare.
    recommended = max(
        0,
        int(np.ceil(predicted_demand + z_score * demand_std)),
    )

    return recommended


# --------------------------------------------------
# 3. Load the trained model and historical data
# --------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATA_FILE}\n"
        "Run python src/analyze_data.py first."
    )

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_FILE}\n"
        "Run python src/train_model.py first."
    )

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "food_item", "units_sold"])

model = joblib.load(MODEL_FILE)


# --------------------------------------------------
# 4. Choose a food item and forecast date
# --------------------------------------------------

food_items = sorted(df["food_item"].unique())

print("\n===== FOOD WASTE OPTIMIZATION =====")
print("\nAvailable food items:")

for index, item in enumerate(food_items, start=1):
    print(f"{index}. {item}")

while True:
    try:
        choice = int(input("\nChoose a food item number: "))
        if 1 <= choice <= len(food_items):
            food_item = food_items[choice - 1]
            break
        print("Choose a number from the displayed list.")
    except ValueError:
        print("Enter a valid whole number.")

forecast_date = date.today()

print(f"\nForecast date: {forecast_date}")

holiday_input = input("Is this a holiday? (y/n): ").strip().lower()

if holiday_input not in {"y", "n"}:
    raise ValueError("Enter y or n for the holiday question.")

is_holiday = int(holiday_input == "y")

day_of_week = forecast_date.weekday()
is_weekend = int(day_of_week >= 5)


# --------------------------------------------------
# 5. Predict demand
# --------------------------------------------------

features = pd.DataFrame([{
    "food_item": food_item,
    "day_of_week": day_of_week,
    "month": forecast_date.month,
    "day_of_year": forecast_date.timetuple().tm_yday,
    "is_weekend": is_weekend,
    "is_holiday": is_holiday,
}])

predicted_demand = max(0.0, float(model.predict(features)[0]))

# Estimate demand variation from historical observations.
item_sales = df.loc[
    df["food_item"] == food_item, "units_sold"
]

demand_std = float(item_sales.std())
if not np.isfinite(demand_std):
    demand_std = 0.0


# --------------------------------------------------
# 6. Configure waste and shortage costs
# --------------------------------------------------

print("\nEnter costs in your local currency.")

waste_cost = float(
    input("Cost per wasted portion (e.g. 20): ")
)

shortage_cost = float(
    input("Cost per unmet portion (e.g. 100): ")
)

recommended = recommend_preparation(
    predicted_demand=predicted_demand,
    demand_std=demand_std,
    waste_cost=waste_cost,
    shortage_cost=shortage_cost,
)


# --------------------------------------------------
# 7. Estimate outcomes and display recommendation
# --------------------------------------------------

# These estimates use a normal demand approximation.
# They are planning estimates, not guaranteed outcomes.
normal = NormalDist(mu=predicted_demand, sigma=demand_std)

if demand_std > 0:
    expected_waste = (
        (recommended - predicted_demand)
        * normal.cdf((recommended - predicted_demand) / demand_std)
        + demand_std
        * np.exp(
            -0.5 * ((recommended - predicted_demand) / demand_std) ** 2
        )
        / np.sqrt(2 * np.pi)
    )

    expected_shortage = (
        (predicted_demand - recommended)
        * (1 - normal.cdf((recommended - predicted_demand) / demand_std))
        + demand_std
        * np.exp(
            -0.5 * ((recommended - predicted_demand) / demand_std) ** 2
        )
        / np.sqrt(2 * np.pi)
    )

else:
    expected_waste = max(recommended - predicted_demand, 0)
    expected_shortage = max(predicted_demand - recommended, 0)

print("\n===== RECOMMENDATION =====")
print("Food item:", food_item)
print(f"Predicted demand: {predicted_demand:.1f} portions")
print(f"Estimated demand variation: {demand_std:.1f}")
print(f"Recommended preparation: {recommended} portions")
print(f"Expected waste: {expected_waste:.1f} portions")
print(f"Expected shortage: {expected_shortage:.1f} portions")
print(
    "Estimated waste and shortage cost:",
    f"{expected_waste * waste_cost + expected_shortage * shortage_cost:.2f}",
)

print("\nRecommendation generated successfully.")
print("Note: this prototype does not yet account for leftover inventory.")
