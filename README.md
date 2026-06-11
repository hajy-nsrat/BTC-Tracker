# BTC Price Tracker & Predictor

A Python project that fetches historical Bitcoin price data, cleans it,
engineers features, trains a next-day return model, and visualizes the results.

## Project Structure

```
BTC-Tracker/
├── data/                   # Raw and cleaned CSV files
├── outputs/                # Charts and prediction CSV
├── src/
│   ├── fetcher.py          # Fetch data from CoinGecko API
│   ├── cleaner.py          # Clean and prepare data
│   ├── features.py         # Add technical indicator features
│   ├── model.py            # Train model and predict
│   ├── prediction_tracker.py  # Log forecasts & score vs actuals
│   └── visualizer.py       # Generate charts
├── main.py                 # Run the full pipeline
└── requirements.txt        # Dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

Full pipeline (fetch + train + forecast):

```bash
python main.py
```

Skip the API fetch and reuse saved data:

```bash
python main.py --skip-fetch
```

Other options:

```bash
python main.py --days 730 --forecast-days 14 --verbose
```

| Flag | Default | Description |
|------|---------|-------------|
| `--days` | 365 | Days of history to fetch |
| `--forecast-days` | 30 | Days to forecast ahead |
| `--skip-fetch` | off | Use existing `data/btc_prices.csv` |
| `--data-path` | `data/btc_prices.csv` | Raw CSV path |
| `--output-dir` | `outputs` | Output directory |
| `--verbose` | off | Debug logging |

## Outputs

- `outputs/price_history.png`  — historical price with moving averages
- `outputs/predictions.png`    — actual vs predicted vs baseline + forecast
- `outputs/volatility.png`     — 7-day rolling volatility
- `outputs/predictions.csv`    — latest forecast as a CSV
- `outputs/prediction_log.csv` — append-only log of every run's forecasts
- `outputs/prediction_scores.csv` — matured predictions vs actual prices

## Prediction tracking

Each run logs its forward forecast. On the next run, once a target date has
passed, the pipeline scores those predictions against real prices and prints
a report (e.g. "yesterday we predicted $X, actual was $Y").

Run daily for meaningful tracking:

```bash
python main.py
```

## Model notes

- The model predicts **next-day percentage return**, not raw price levels.
- Features at day `t` use only data through day `t` (no same-day leakage).
- Multi-day forecasts roll forward day-by-day, recomputing rolling features each step.
- Metrics include a **naive baseline** (tomorrow's price = today's price) and
  walk-forward cross-validation on the training set.

Predictions are for learning purposes only. Bitcoin is highly volatile
and real price forecasting is significantly more complex.
