# Step 3: Feature engineering — add useful columns for the model

import logging

import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "ma_7",
    "ma_30",
    "pct_change",
    "volatility_7",
    "price_current",
    "price_lag_7",
    "price_lag_30",
]


def compute_features_from_prices(prices: pd.Series) -> dict:
    """
    Compute model features from a price series ending on the most recent day.

    All values use only data through the last row (no future leakage).
    Requires at least 31 observations for the 30-day lag feature.
    """
    if len(prices) < 31:
        raise ValueError(f"Need at least 31 price observations, got {len(prices)}")

    tail_7 = prices.iloc[-7:]
    tail_30 = prices.iloc[-30:]

    current = prices.iloc[-1]
    previous = prices.iloc[-2]

    return {
        "ma_7": tail_7.mean(),
        "ma_30": tail_30.mean(),
        "pct_change": (current - previous) / previous * 100,
        "volatility_7": tail_7.std(),
        "price_current": current,
        "price_lag_7": prices.iloc[-8],
        "price_lag_30": prices.iloc[-31],
    }


def add_features(df):
    """
    Adds technical indicator columns for next-day return prediction.

    Each row at date t uses prices through day t to predict the return into day t+1.
  """
    logger.info("Engineering features...")

    df = df.copy()

    df["ma_7"] = df["price"].rolling(window=7).mean()
    df["ma_30"] = df["price"].rolling(window=30).mean()
    df["pct_change"] = df["price"].pct_change() * 100
    df["volatility_7"] = df["price"].rolling(window=7).std()

    df["price_current"] = df["price"]
    df["price_lag_7"] = df["price"].shift(7)
    df["price_lag_30"] = df["price"].shift(30)

    # Target: percentage return from day t to day t+1
    df["next_return"] = df["price"].pct_change().shift(-1) * 100

    before = len(df)
    df = df.dropna()
    logger.info(
        "Dropped %d rows with NaN. Features ready: %d rows.",
        before - len(df),
        len(df),
    )

    return df
