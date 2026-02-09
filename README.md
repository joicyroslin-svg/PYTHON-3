# House Price Prediction (Linear Regression, INR)

Beginner-friendly project that builds a Linear Regression model to predict house
prices from a small, sample dataset.

## What this project includes
- Sample dataset creation inside the script (Indian Rupees, INR)
- Train/test split with evaluation metrics (MAE, R²)
- Model training
- Price prediction for a new example
- Matplotlib visualization with two charts (Actual vs Predicted, Residuals)

## Requirements
Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Run

```bash
python House-Price-Prediction.py
```

## Project summary
This project builds a simple Linear Regression model to estimate house prices
based on practical features like size, bedrooms, bathrooms, age, distance to
city center, and parking. It uses a clean train/test split, reports evaluation
metrics, and visualizes performance with both an actual-vs-predicted plot and a
residuals plot. The goal is to demonstrate a complete, beginner-friendly ML
workflow suitable for internship submission.

## Notes
- Prices are in lakhs (1 lakh = 100,000 INR) in the dataset.
- Example prediction prints a price in INR lakhs and shows a chart.
- Features used: size, bedrooms, bathrooms, age, distance to city center, parking.
