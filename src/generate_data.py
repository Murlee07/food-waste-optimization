
import csv
import random
from datetime import date, timedelta
from pathlib import Path

# Make the generated data reproducible
random.seed(42)

# Find the main project folder automatically
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "food_sales.csv"

food_items = {
    "Vegetable Rice": 40,
    "Chicken Biryani": 60,
    "Paneer Curry": 30,
    "Masala Dosa": 45,
    "Veg Burger": 35,
}

start_date = date(2025, 1, 1)
number_of_days = 365

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([
        "date",
        "food_item",
        "day_of_week",
        "is_weekend",
        "is_holiday",
        "units_sold",
        "quantity_prepared",
        "waste_quantity",
    ])

    for day_number in range(number_of_days):
        current_date = start_date + timedelta(days=day_number)
        day_of_week = current_date.weekday()
        is_weekend = int(day_of_week >= 5)

        # Illustrative holiday dates for our sample dataset
        is_holiday = int(
            (current_date.month, current_date.day)
            in {(1, 26), (8, 15), (10, 2), (12, 25)}
        )

        for food_item, base_sales in food_items.items():
            demand = base_sales

            # Slightly higher demand on weekends and holidays
            if is_weekend:
                demand *= 1.20
            if is_holiday:
                demand *= 1.25

            # Add random variation in daily sales
            units_sold = max(
                0, round(random.gauss(demand, base_sales * 0.15))
            )

            # Simulate a kitchen that prepares slightly more than demand
            extra_food = random.randint(2, 12)
            quantity_prepared = units_sold + extra_food
            waste_quantity = quantity_prepared - units_sold

            writer.writerow([
                current_date.isoformat(),
                food_item,
                day_of_week,
                is_weekend,
                is_holiday,
                units_sold,
                quantity_prepared,
                waste_quantity,
            ])

print(f"Dataset created successfully: {OUTPUT_FILE}")
print(f"Number of records: {number_of_days * len(food_items):,}")
