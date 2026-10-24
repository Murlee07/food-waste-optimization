
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. Set up file paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "raw" / "food_sales.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("\n========== DATASET OVERVIEW ==========")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))
print("\nFirst five records:")
print(df.head())

print("\nColumn data types:")
print(df.dtypes)

# --------------------------------------------------
# 3. Check data quality
# --------------------------------------------------

print("\n========== DATA QUALITY ==========")
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Convert date strings into actual dates
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Remove duplicate rows and rows with missing essential values
df = df.drop_duplicates()

required_columns = [
    "date",
    "food_item",
    "units_sold",
    "quantity_prepared",
    "waste_quantity",
]

df = df.dropna(subset=required_columns)

# Validate quantities
quantity_columns = [
    "units_sold",
    "quantity_prepared",
    "waste_quantity",
]

for column in quantity_columns:
    df = df[df[column] >= 0]

# Save cleaned dataset
cleaned_file = PROCESSED_DIR / "food_sales_cleaned.csv"
df.to_csv(cleaned_file, index=False)

print("\nCleaned records:", len(df))
print("Saved cleaned dataset to:", cleaned_file)

# --------------------------------------------------
# 4. Calculate food-item statistics
# --------------------------------------------------

summary = (
    df.groupby("food_item")
    .agg(
        total_sales=("units_sold", "sum"),
        total_prepared=("quantity_prepared", "sum"),
        total_waste=("waste_quantity", "sum"),
        average_daily_sales=("units_sold", "mean"),
    )
    .reset_index()
)

summary["waste_percentage"] = (
    summary["total_waste"] / summary["total_prepared"] * 100
)

summary = summary.sort_values("total_waste", ascending=False)

print("\n========== FOOD ITEM SUMMARY ==========")
print(summary.round(2).to_string(index=False))

summary.to_csv(
    PROCESSED_DIR / "food_item_summary.csv",
    index=False,
)

# --------------------------------------------------
# 5. Compare weekday and weekend demand
# --------------------------------------------------

df["day_type"] = df["is_weekend"].map({
    0: "Weekday",
    1: "Weekend",
})

day_summary = (
    df.groupby("day_type")["units_sold"]
    .mean()
    .reindex(["Weekday", "Weekend"])
)

print("\n========== WEEKDAY VS WEEKEND ==========")
print(day_summary.round(2))

# --------------------------------------------------
# 6. Create a food waste chart
# --------------------------------------------------

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))

sns.barplot(
    data=summary,
    x="food_item",
    y="total_waste",
)

plt.title("Total Food Waste by Food Item")
plt.xlabel("Food Item")
plt.ylabel("Total Waste (Units)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

waste_chart = FIGURES_DIR / "food_waste_by_item.png"
plt.savefig(waste_chart, dpi=150)
plt.close()

# --------------------------------------------------
# 7. Create a daily sales trend chart
# --------------------------------------------------

daily_sales = (
    df.groupby("date")["units_sold"]
    .sum()
    .sort_index()
)

plt.figure(figsize=(12, 6))

plt.plot(
    daily_sales.index,
    daily_sales.values,
    label="Daily Sales",
)

plt.title("Daily Restaurant Sales Trend")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.tight_layout()

sales_chart = FIGURES_DIR / "daily_sales_trend.png"
plt.savefig(sales_chart, dpi=150)
plt.close()

# --------------------------------------------------
# 8. Finish
# --------------------------------------------------

print("\n========== ANALYSIS COMPLETE ==========")
print("Summary:", PROCESSED_DIR / "food_item_summary.csv")
print("Waste chart:", waste_chart)
print("Sales chart:", sales_chart)
