"""
Beginner-friendly house price prediction using Linear Regression.
This script creates a small sample dataset, trains a model, evaluates it,
makes a prediction, and visualizes results.

Notes:
- Currency is Indian Rupees (INR).
- Prices are in lakhs (1 lakh = 100,000 INR) to keep numbers readable.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def create_sample_data() -> tuple[np.ndarray, np.ndarray]:
    # Features:
    # [size_sqft, bedrooms, bathrooms, age_years, distance_km, parking]
    # Prices are in lakhs (INR).
    data = np.array(
        [
            [850, 2, 1, 28, 18, 0],
            [900, 2, 1, 20, 15, 0],
            [1100, 2, 2, 15, 14, 0],
            [1200, 3, 2, 12, 12, 1],
            [1350, 3, 2, 10, 10, 1],
            [1500, 3, 2, 8, 9, 1],
            [1650, 3, 2, 6, 8, 1],
            [1800, 3, 3, 5, 7, 1],
            [2000, 4, 3, 5, 6, 1],
            [2200, 4, 3, 4, 5, 1],
            [2400, 4, 3, 4, 4, 1],
            [2600, 4, 4, 3, 4, 1],
            [2800, 4, 4, 3, 3, 1],
            [3000, 5, 4, 2, 3, 1],
            [3200, 5, 4, 2, 2, 1],
            [3400, 5, 5, 1, 2, 1],
            [3600, 5, 5, 1, 1, 1],
            [3800, 5, 5, 1, 1, 1],
        ],
        dtype=float,
    )
    prices = np.array(
        [45, 48, 55, 62, 70, 78, 85, 95, 110, 125, 135, 150, 165, 180, 195, 210, 230, 250],
        dtype=float,
    )

    return data, prices


def train_model(x: np.ndarray, y: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(x, y)
    return model


def predict_price(
    model: LinearRegression,
    size_sqft: float,
    bedrooms: int,
    bathrooms: int,
    age_years: float,
    distance_km: float,
    parking: int,
) -> float:
    features = np.array(
        [[size_sqft, bedrooms, bathrooms, age_years, distance_km, parking]],
        dtype=float,
    )
    predicted_lakhs = model.predict(features)[0]
    return float(predicted_lakhs)


def plot_results(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("House Price Prediction (INR Lakhs)", fontsize=12, fontweight="bold")

    ax.scatter(y_true, y_pred, color="#64dfdf", edgecolor="#0b0f14", s=70, label="Predictions")
    ax.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        color="#f4a261",
        linestyle="--",
        linewidth=2,
        label="Perfect Fit",
    )
    ax.set_title("Actual vs Predicted", fontsize=10)
    ax.set_xlabel("Actual Price (INR Lakhs)")
    ax.set_ylabel("Predicted Price (INR Lakhs)")
    ax.grid(color="#2c333a", alpha=0.6, linewidth=0.6)
    ax.legend()

    plt.tight_layout()

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "house_price_plots.png")
    plt.savefig(output_path, dpi=200)
    plt.show()
    return output_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    x, y = create_sample_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )
    model = train_model(x_train, y_train)

    # Evaluate on test data
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model evaluation (test set):")
    print(f"  MAE: {mae:.2f} lakhs")
    print(f"  R^2: {r2:.3f}")

    # Example prediction
    example_features = {
        "size_sqft": 2100,
        "bedrooms": 4,
        "bathrooms": 3,
        "age_years": 6,
        "distance_km": 6,
        "parking": 1,
    }
    predicted_lakhs = predict_price(
        model,
        example_features["size_sqft"],
        example_features["bedrooms"],
        example_features["bathrooms"],
        example_features["age_years"],
        example_features["distance_km"],
        example_features["parking"],
    )

    print("\nExample house:")
    for key, value in example_features.items():
        print(f"  {key}: {value}")
    print(f"Predicted price: ₹{predicted_lakhs:,.2f} lakhs")

    output_path = plot_results(y_test, y_pred)
    print(f"\nSaved plots to: {output_path}")


if __name__ == "__main__":
    main()
