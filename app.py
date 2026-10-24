
from datetime import date
from pathlib import Path
from statistics import NormalDist

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_FILE = ROOT / "data" / "processed" / "food_sales_cleaned.csv"
MODEL_FILE = ROOT / "models" / "demand_model.joblib"

st.set_page_config(
    page_title="Food Waste Prediction & Optimization",
    page_icon="🍽️",
    layout="wide",
)

st.title("🍽️ AI-Powered Food Waste Prediction & Optimization")
st.write(
    "Forecast restaurant demand, estimate potential waste, "
    "and plan food preparation."
)


# --------------------------------------------------
# 2. Load data and model
# --------------------------------------------------

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(
        subset=["date", "food_item", "units_sold"]
    )
    return data


@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not DATA_FILE.exists():
    st.error(
        f"Dataset not found: {DATA_FILE}. "
        "Run python src/analyze_data.py first."
    )
    st.stop()

if not MODEL_FILE.exists():
    st.error(
        f"Trained model not found: {MODEL_FILE}. "
        "Run python src/train_model.py first."
    )
    st.stop()

df = load_data(str(DATA_FILE))
model = load_model(str(MODEL_FILE))

if df.empty:
    st.error("The dataset contains no usable records.")
    st.stop()


# --------------------------------------------------
# 3. Historical overview
# --------------------------------------------------

st.header("1. Historical Restaurant Overview")

total_sales = int(df["units_sold"].sum())
total_prepared = int(df["quantity_prepared"].sum())
total_waste = int(df["waste_quantity"].sum())

waste_rate = (
    total_waste / total_prepared * 100
    if total_prepared > 0
    else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total units sold", f"{total_sales:,}")
col2.metric("Total units prepared", f"{total_prepared:,}")
col3.metric("Total food waste", f"{total_waste:,}")
col4.metric("Historical waste rate", f"{waste_rate:.1f}%")

st.caption(
    "These statistics describe the supplied dataset. "
    "The current dataset is synthetic practice data."
)


# --------------------------------------------------
# 4. Historical charts
# --------------------------------------------------

st.subheader("Daily Sales Trend")

daily_sales = (
    df.groupby("date")["units_sold"]
    .sum()
    .sort_index()
)

st.line_chart(daily_sales)

st.subheader("Waste by Food Item")

waste_summary = (
    df.groupby("food_item")["waste_quantity"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(waste_summary)


# --------------------------------------------------
# 5. Demand prediction and preparation planning
# --------------------------------------------------

st.header("2. Demand Forecast & Preparation Planner")

food_items = sorted(df["food_item"].unique())

with st.form("forecast_form"):
    selected_food = st.selectbox(
        "Select a food item",
        food_items,
    )

    forecast_date = st.date_input(
        "Forecast date",
        value=date.today(),
        min_value=date(2020, 1, 1),
        max_value=date(2035, 12, 31),
    )

    is_holiday = st.checkbox(
        "Is this date a holiday?",
        value=False,
    )

    waste_cost = st.number_input(
        "Cost per wasted portion",
        min_value=0.01,
        value=20.0,
        step=5.0,
    )

    shortage_cost = st.number_input(
        "Cost per unmet portion",
        min_value=0.01,
        value=100.0,
        step=5.0,
    )

    available_leftovers = st.number_input(
        "Usable leftover portions available",
        min_value=0,
        value=0,
        step=1,
    )

    submitted = st.form_submit_button("Predict & Optimize")


if submitted:
    day_of_week = forecast_date.weekday()

    features = pd.DataFrame([{
        "food_item": selected_food,
        "day_of_week": day_of_week,
        "month": forecast_date.month,
        "day_of_year": forecast_date.timetuple().tm_yday,
        "is_weekend": int(day_of_week >= 5),
        "is_holiday": int(is_holiday),
    }])

    predicted_demand = max(
        0.0,
        float(model.predict(features)[0]),
    )

    historical_item = df.loc[
        df["food_item"] == selected_food,
        "units_sold",
    ]

    demand_std = float(historical_item.std())

    if not np.isfinite(demand_std):
        demand_std = 0.0

    critical_fractile = (
        shortage_cost / (shortage_cost + waste_cost)
    )

    z_score = NormalDist().inv_cdf(critical_fractile)

    # Recommended total stock, including usable leftovers.
    target_stock = max(
        0,
        int(np.ceil(predicted_demand + z_score * demand_std)),
    )

    # Only prepare the quantity not already available.
    recommended_preparation = max(
        0,
        target_stock - int(available_leftovers),
    )

    expected_waste = 0.0
    expected_shortage = 0.0

    if demand_std > 0:
        z = (
            target_stock - predicted_demand
        ) / demand_std

        phi = (
            np.exp(-0.5 * z**2)
            / np.sqrt(2 * np.pi)
        )

        cdf = NormalDist().cdf(z)

        expected_waste = max(
            0.0,
            (target_stock - predicted_demand) * cdf
            + demand_std * phi,
        )

        expected_shortage = max(
            0.0,
            (predicted_demand - target_stock) * (1 - cdf)
            + demand_std * phi,
        )
    else:
        expected_waste = max(
            0.0,
            target_stock - predicted_demand,
        )

        expected_shortage = max(
            0.0,
            predicted_demand - target_stock,
        )

    expected_cost = (
        expected_waste * waste_cost
        + expected_shortage * shortage_cost
    )

    st.subheader(f"Forecast for {selected_food}")
    st.caption(f"Date: {forecast_date.isoformat()}")

    a, b, c = st.columns(3)

    a.metric(
        "Predicted demand",
        f"{predicted_demand:.1f} portions",
    )

    b.metric(
        "Total stock target",
        f"{target_stock} portions",
    )

    c.metric(
        "Prepare now",
        f"{recommended_preparation} portions",
    )

    d, e, f = st.columns(3)

    d.metric(
        "Expected waste",
        f"{expected_waste:.1f} portions",
    )

    e.metric(
        "Expected shortage",
        f"{expected_shortage:.1f} portions",
    )

    f.metric(
        "Expected waste + shortage cost",
        f"{expected_cost:.2f}",
    )

    st.info(
        "Planning estimate only. Demand variation currently uses "
        "historical sales variation, not calibrated forecast errors. "
        "Replace this assumption with validation-based uncertainty "
        "before using the recommendations operationally."
    )


# --------------------------------------------------
# 6. Historical food item details
# --------------------------------------------------

st.header("3. Food Item Details")

selected_history_item = st.selectbox(
    "Explore historical performance",
    food_items,
    key="history_item",
)

item_history = df.loc[
    df["food_item"] == selected_history_item
].sort_values("date")

st.dataframe(
    item_history[
        [
            "date",
            "units_sold",
            "quantity_prepared",
            "waste_quantity",
        ]
    ].tail(30),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "AI-generated estimates are not guaranteed outcomes. "
    "Validate this prototype against real restaurant data "
    "before making purchasing or preparation decisions."
)
