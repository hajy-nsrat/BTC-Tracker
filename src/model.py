# Step 4: Train a model and generate price predictions

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.features import FEATURE_COLS, compute_features_from_prices

logger = logging.getLogger(__name__)


def _return_to_next_price(price_t: float, predicted_return_pct: float) -> float:
    return price_t * (1 + predicted_return_pct / 100)


def _evaluate_price_mae(actual_prices, predicted_prices) -> float:
    return mean_absolute_error(actual_prices, predicted_prices)


def _cross_validate_returns(X_train, y_train, prices_train, n_splits=5):
    """Walk-forward CV on the training set; reports mean price MAE."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_maes = []

    for train_ix, val_ix in tscv.split(X_train):
        if len(val_ix) == 0:
            continue

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train.iloc[train_ix], y_train.iloc[train_ix])

        val_returns = model.predict(X_train.iloc[val_ix])
        actual_next = prices_train.shift(-1).iloc[val_ix]
        price_t = prices_train.iloc[val_ix]
        predicted_next = [
            _return_to_next_price(p, r) for p, r in zip(price_t, val_returns)
        ]

        valid = actual_next.notna()
        if valid.sum() == 0:
            continue
        fold_maes.append(
            _evaluate_price_mae(actual_next[valid], np.array(predicted_next)[valid.values])
        )

    return float(np.mean(fold_maes)) if fold_maes else None


def train_and_predict(df, forecast_days=30, test_size=0.2, n_cv_splits=5):
    """
    Trains a Random Forest on next-day returns and forecasts future prices.

    Features at day t use only data through t. The model predicts the return
    into day t+1, which is converted back to a price for evaluation and plotting.
    """
    logger.info("Training Random Forest model (next-day return target)...")

    X = df[FEATURE_COLS]
    y = df["next_return"]
    prices = df["price"]

    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    prices_train = prices.iloc[:split_idx]

    cv_mae = _cross_validate_returns(
        X_train, y_train, prices_train, n_splits=n_cv_splits
    )
    if cv_mae is not None:
        logger.info("  Walk-forward CV price MAE: $%s", f"{cv_mae:,.2f}")

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    test_pred_returns = model.predict(X_test)

    test_dates = []
    actual_prices = []
    predicted_prices = []
    baseline_prices = []

    for idx, pred_return in zip(X_test.index, test_pred_returns):
        price_t = prices.loc[idx]
        actual_next = prices.shift(-1).loc[idx]
        if pd.isna(actual_next):
            continue

        test_dates.append(idx + pd.Timedelta(days=1))
        actual_prices.append(actual_next)
        predicted_prices.append(_return_to_next_price(price_t, pred_return))
        baseline_prices.append(price_t)  # naive: 0% return → price stays flat

    mae = _evaluate_price_mae(actual_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    baseline_mae = _evaluate_price_mae(actual_prices, baseline_prices)

    logger.info("Test set (next-day price):")
    logger.info("  Model MAE     : $%s", f"{mae:,.2f}")
    logger.info("  Model RMSE    : $%s", f"{rmse:,.2f}")
    logger.info("  Baseline MAE  : $%s (naive: tomorrow = today)", f"{baseline_mae:,.2f}")
    if mae < baseline_mae:
        logger.info("  Model beats baseline by $%s", f"{baseline_mae - mae:,.2f}")
    else:
        logger.info("  Baseline beats model by $%s", f"{mae - baseline_mae:,.2f}")

    logger.info("Generating %d-day future forecast...", forecast_days)

    price_history = prices.copy()
    last_date = df.index[-1]
    future_prices = []
    future_dates = []

    for i in range(1, forecast_days + 1):
        features = pd.DataFrame([compute_features_from_prices(price_history)])
        pred_return = model.predict(features)[0]
        next_price = _return_to_next_price(price_history.iloc[-1], pred_return)
        next_date = last_date + pd.Timedelta(days=i)

        future_prices.append(next_price)
        future_dates.append(next_date)

        price_history = pd.concat(
            [price_history, pd.Series([next_price], index=[next_date])]
        )

    future_df = pd.DataFrame(
        {"predicted_price": future_prices},
        index=pd.DatetimeIndex(future_dates, name="date"),
    )

    test_df = pd.DataFrame(
        {
            "actual": actual_prices,
            "predicted": predicted_prices,
            "baseline": baseline_prices,
        },
        index=pd.DatetimeIndex(test_dates, name="date"),
    )

    return {
        "model": model,
        "test_results": test_df,
        "future_predictions": future_df,
        "mae": mae,
        "rmse": rmse,
        "baseline_mae": baseline_mae,
        "cv_mae": cv_mae,
    }


def save_predictions(future_df, filepath="outputs/predictions.csv"):
    future_df.to_csv(filepath)
    logger.info("Predictions saved to %s", filepath)
