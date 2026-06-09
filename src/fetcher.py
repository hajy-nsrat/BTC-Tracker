# Step 1: Fetch historical BTC price data from CoinGecko API

import logging
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30


def fetch_btc_prices(days=365, currency="usd"):
    """
    Fetches historical BTC prices from CoinGecko API.

    Returns a DataFrame with 'date' and 'price' columns, or None on failure.
    """
    params = {
        "vs_currency": currency,
        "days": days,
        "interval": "daily",
    }

    logger.info("Fetching BTC price data for the last %d days...", days)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                COINGECKO_URL, params=params, timeout=REQUEST_TIMEOUT
            )
        except requests.RequestException as exc:
            logger.warning("Request error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                return None
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 429:
            wait = 2 ** attempt
            logger.warning("Rate limited (429). Retrying in %ds...", wait)
            time.sleep(wait)
            continue

        if response.status_code != 200:
            logger.error("Request failed. Status code: %d", response.status_code)
            return None

        data = response.json()
        prices = data["prices"]

        df = pd.DataFrame(prices, columns=["date", "price"])
        df["date"] = pd.to_datetime(df["date"], unit="ms").dt.normalize()

        logger.info("Successfully fetched %d rows of data.", len(df))
        return df

    return None


def save_to_csv(df, filepath="data/btc_prices.csv"):
    df.to_csv(filepath, index=False)
    logger.info("Data saved to %s", filepath)
