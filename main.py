
# AI-Powered Food Waste Prediction & Optimization

project_name = "Food Waste Prediction & Optimization"

food_items = [
    "Vegetable Rice",
    "Chicken Biryani",
    "Paneer Curry"
]

historical_sales = [40, 55, 35]

print("=" * 45)
print(project_name)
print("=" * 45)

for i in range(len(food_items)):
    print(food_items[i], ":", historical_sales[i], "units sold")

total_sales = sum(historical_sales)
print("Total sales:", total_sales)

average_sales = total_sales / len(historical_sales)
print("Average sales per food item:", round(average_sales, 2))
