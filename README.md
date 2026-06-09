# BTC Price Tracker & Predictor

A Python project that fetches historical Bitcoin price data, cleans it,
engineers features, trains a prediction model, and visualizes the results.

## Project Structure

```
btc_tracker/
├── data/                   # Raw and cleaned CSV files
├── outputs/                # Charts and prediction CSV
├── notebooks/              # Jupyter notebooks for exploration
├── src/
│   ├── fetcher.py          # Fetch data from CoinGecko API
│   ├── cleaner.py          # Clean and prepare data
│   ├── features.py         # Add technical indicator features
│   ├── model.py            # Train model and predict
│   └── visualizer.py       # Generate charts
├── main.py                 # Run the full pipeline
└── requirements.txt        # Dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Outputs

- `outputs/price_history.png`  — historical price with moving averages
- `outputs/predictions.png`    — actual vs predicted + 30-day forecast
- `outputs/volatility.png`     — 7-day rolling volatility
- `outputs/predictions.csv`    — 30-day forecast as a CSV

## Notes

Predictions are for learning purposes only. Bitcoin is highly volatile
and real price forecasting is significantly more complex.
