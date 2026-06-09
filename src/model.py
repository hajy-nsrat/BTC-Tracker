# Step 4: Train a model and generate price predictions

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_and_predict(df, forecast_days=30):
    """
    Trains a Random Forest model on historical BTC features
    and predicts the next `forecast_days` of prices.

    Args:
        df (pd.DataFrame): Feature-engineered DataFrame.
        forecast_days (int): How many days into the future to predict.

    Returns:
        dict: Contains 'actual', 'test_predictions', 'future_predictions',
              'mae', 'rmse', and 'model'.
    """
    print("\nTraining Random Forest model...")

    # Features used for training
    feature_cols = ["ma_7", "ma_30", "pct_change", "volatility_7", "lag_1", "lag_7", "lag_30"]
    target_col = "price"

    X = df[feature_cols]
    y = df[target_col]

    # Split into train/test (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # shuffle=False keeps time order
    )

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on test set
    test_preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    print(f"Model trained.")
    print(f"  MAE:  ${mae:,.2f}  (on average, predictions are off by this amount)")
    print(f"  RMSE: ${rmse:,.2f}")

    # --- Future predictions ---
    # We roll forward day by day, using the last known data as input
    print(f"\nGenerating {forecast_days}-day future forecast...")

    last_row = df.iloc[-1].copy()
    future_prices = []
    future_dates  = []

    last_date = df.index[-1]

    for i in range(1, forecast_days + 1):
        features = pd.DataFrame([{
            "ma_7":         last_row["ma_7"],
            "ma_30":        last_row["ma_30"],
            "pct_change":   last_row["pct_change"],
            "volatility_7": last_row["volatility_7"],
            "lag_1":        last_row["price"],
            "lag_7":        last_row["lag_7"],
            "lag_30":       last_row["lag_30"],
        }])

        predicted_price = model.predict(features)[0]
        next_date = last_date + pd.Timedelta(days=i)

        future_prices.append(predicted_price)
        future_dates.append(next_date)

        # Update last_row so next iteration uses this prediction as lag_1
        last_row["lag_30"]      = last_row["lag_7"]
        last_row["lag_7"]       = last_row["lag_1"]
        last_row["lag_1"]       = last_row["price"]
        last_row["price"]       = predicted_price
        last_row["pct_change"]  = ((predicted_price - last_row["lag_1"]) / last_row["lag_1"]) * 100

    future_df = pd.DataFrame({
        "date":  future_dates,
        "predicted_price": future_prices
    }).set_index("date")

    test_df = pd.DataFrame({
        "actual":    y_test.values,
        "predicted": test_preds
    }, index=y_test.index)

    return {
        "model":               model,
        "test_results":        test_df,
        "future_predictions":  future_df,
        "mae":                 mae,
        "rmse":                rmse,
    }


def save_predictions(future_df, filepath="outputs/predictions.csv"):
    future_df.to_csv(filepath)
    print(f"Predictions saved to {filepath}")
