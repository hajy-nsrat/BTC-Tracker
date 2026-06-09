# Step 1: Fetch historical BTC price data from CoinGecko API

import requests
import pandas as pd


def fetch_btc_prices(days=365, currency="usd"):
    """
    Fetches historical BTC prices from CoinGecko API.

    Args:
        days (int): Number of days of historical data to fetch. Default is 365.
        currency (str): The currency to fetch prices in. Default is 'usd'.

    Returns:
        pd.DataFrame: A DataFrame with two columns: 'date' and 'price'.
                      Returns None if the request fails.
    """
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

    params = {
        "vs_currency": currency,
        "days": days,
        "interval": "daily"
    }

    print(f"Fetching BTC price data for the last {days} days...")

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Request failed. Status code: {response.status_code}")
        return None

    data = response.json()
    prices = data["prices"]

    df = pd.DataFrame(prices, columns=["date", "price"])
    df["date"] = pd.to_datetime(df["date"], unit="ms")
    df["date"] = df["date"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    print(f"Successfully fetched {len(df)} rows of data.")
    return df


def save_to_csv(df, filepath="data/btc_prices.csv"):
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
