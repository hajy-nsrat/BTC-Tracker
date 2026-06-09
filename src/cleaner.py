# Step 2: Clean and prepare the BTC price data

import pandas as pd


def load_data(filepath="data/btc_prices.csv"):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, parse_dates=["date"])
    print(f"Loaded {len(df)} rows.")
    return df


def inspect_data(df):
    print("\n--- Shape ---")
    print(f"{df.shape[0]} rows, {df.shape[1]} columns")
    print("\n--- First 5 rows ---")
    print(df.head())
    print("\n--- Missing values ---")
    print(df.isnull().sum())
    print("\n--- Price statistics ---")
    print(df["price"].describe())


def clean_data(df):
    print("\nCleaning data...")

    before = len(df)
    df = df.drop_duplicates(subset="date", keep="first")
    after = len(df)
    if before != after:
        print(f"Removed {before - after} duplicate rows.")

    df["date"] = pd.to_datetime(df["date"])
    df["price"] = df["price"].astype(float)
    df = df.sort_values("date").reset_index(drop=True)

    missing_before = df["price"].isnull().sum()
    df["price"] = df["price"].ffill()
    if missing_before > 0:
        print(f"Filled {missing_before} missing price values.")

    df = df.set_index("date")
    print(f"Cleaning complete. Final shape: {df.shape}")
    return df


def save_clean_data(df, filepath="data/btc_prices_clean.csv"):
    df.to_csv(filepath)
    print(f"Clean data saved to {filepath}")
