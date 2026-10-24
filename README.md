# AI-Powered Food Waste Prediction & Optimization

## Project Overview

This project uses machine learning to predict food demand and recommend preparation quantities to help reduce food waste while minimizing shortages.

The project demonstrates a complete workflow, from synthetic data generation and exploratory data analysis to model training, evaluation, and an interactive Streamlit dashboard.

## Features

- Generate sample food sales and waste data.
- Analyze food sales and waste patterns.
- Train and compare Linear Regression and Random Forest models.
- Evaluate demand predictions using MAE, RMSE, and R².
- Recommend preparation quantities using demand predictions, waste costs, shortage costs, and available leftovers.
- Visualize actual versus predicted demand.
- Explore results through an interactive Streamlit dashboard.

## Technology Stack

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit

## Project Structure

```text
food-waste-optimization/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── generate_data.py
│   ├── analyze_data.py
│   ├── train_model.py
│   ├── optimize_food.py
│   ├── evaluate_model.py
│   └── visualize_predictions.py
├── app.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd food-waste-optimization
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run the Project

Generate sample data:

```bash
python src/generate_data.py
```

Analyze the data:

```bash
python src/analyze_data.py
```

Train the model:

```bash
python src/train_model.py
```

Evaluate predictions:

```bash
python src/evaluate_model.py
```

Generate prediction visualizations:

```bash
python src/visualize_predictions.py
```

Launch the dashboard:

```bash
python -m streamlit run app.py
```

## Evaluation Metrics

- **MAE:** Mean Absolute Error.
- **RMSE:** Root Mean Squared Error.
- **R²:** Coefficient of determination.

Record your actual evaluation results after running the evaluation script.

## Data Disclaimer

The current project uses synthetically generated data for educational and demonstration purposes. Results do not establish real-world food waste reduction or production accuracy. Real operational data is required for practical validation.

## Future Improvements

- Train and evaluate models using real sales and inventory data.
- Include ingredient inventory, expiry dates, and leftover tracking.
- Improve model validation and compare additional algorithms.
- Add historical dashboards and downloadable reports.
- Deploy the dashboard for online access.

## License

Choose and add an appropriate open-source license before distributing the project.
