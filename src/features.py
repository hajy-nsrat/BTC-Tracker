# Step 3: Feature engineering — add useful columns for the model

import pandas as pd


def add_features(df):
    """
    Adds technical indicator columns to the DataFrame:
    - 7-day and 30-day rolling averages
    - Daily % change
    - 7-day rolling volatility (std dev)
    - Lag features (price 1, 7, 30 days ago)

    Args:
        df (pd.DataFrame): Cleaned DataFrame with 'price' column and date index.

    Returns:
        pd.DataFrame: DataFrame with additional feature columns, NaN rows dropped.
    """
    print("Engineering features...")

    df = df.copy()

    # Rolling averages (trend indicators)
    df["ma_7"]  = df["price"].rolling(window=7).mean()
    df["ma_30"] = df["price"].rolling(window=30).mean()

    # Daily percentage change
    df["pct_change"] = df["price"].pct_change() * 100

    # Volatility: rolling 7-day standard deviation
    df["volatility_7"] = df["price"].rolling(window=7).std()

    # Lag features: price N days ago
    df["lag_1"]  = df["price"].shift(1)
    df["lag_7"]  = df["price"].shift(7)
    df["lag_30"] = df["price"].shift(30)

    # Drop rows with NaN (caused by rolling/shift operations)
    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df)} rows with NaN from rolling calculations.")
    print(f"Features added. Final shape: {df.shape}")

    return df
