# main.py — runs the full BTC tracker pipeline

import argparse
import logging
import os
import sys

from src.cleaner import clean_data, inspect_data, load_data, save_clean_data
from src.features import add_features
from src.fetcher import fetch_btc_prices, save_to_csv
from src.model import save_predictions, train_and_predict
from src.visualizer import plot_predictions, plot_price_history, plot_volatility


def parse_args():
    parser = argparse.ArgumentParser(description="BTC Price Tracker & Predictor")
    parser.add_argument("--days", type=int, default=365, help="Days of history to fetch")
    parser.add_argument(
        "--forecast-days", type=int, default=30, help="Days to forecast ahead"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip API fetch and use existing data/btc_prices.csv",
    )
    parser.add_argument(
        "--data-path",
        default="data/btc_prices.csv",
        help="Path to raw price CSV",
    )
    parser.add_argument(
        "--clean-path",
        default="data/btc_prices_clean.csv",
        help="Path to cleaned price CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for charts and predictions",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging"
    )
    return parser.parse_args()


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stdout,
    )


def main():
    args = parse_args()
    setup_logging(args.verbose)

    os.makedirs("data", exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 45)
    print("        BTC Price Tracker & Predictor")
    print("=" * 45)

    if not args.skip_fetch:
        print("\n[Step 1] Fetching BTC price data...")
        df_raw = fetch_btc_prices(days=args.days)
        if df_raw is None:
            print("Failed to fetch data. Exiting.")
            return 1
        save_to_csv(df_raw, filepath=args.data_path)
    else:
        print(f"\n[Step 1] Skipping fetch — using {args.data_path}")

    print("\n[Step 2] Cleaning data...")
    df = load_data(filepath=args.data_path)
    inspect_data(df)
    df_clean = clean_data(df)
    save_clean_data(df_clean, filepath=args.clean_path)

    print("\n[Step 3] Engineering features...")
    df_features = add_features(df_clean)

    print("\n[Step 4] Training model and generating predictions...")
    results = train_and_predict(df_features, forecast_days=args.forecast_days)
    predictions_path = os.path.join(args.output_dir, "predictions.csv")
    save_predictions(results["future_predictions"], filepath=predictions_path)

    print("\n[Step 5] Generating charts...")
    plot_price_history(
        df_features, save_path=os.path.join(args.output_dir, "price_history.png")
    )
    plot_predictions(
        df_features, results, save_path=os.path.join(args.output_dir, "predictions.png")
    )
    plot_volatility(
        df_features, save_path=os.path.join(args.output_dir, "volatility.png")
    )

    print("\n" + "=" * 45)
    print("Pipeline complete!")
    print(f"  Date range   : {df_clean.index.min().date()} → {df_clean.index.max().date()}")
    print(f"  Last price   : ${df_clean['price'].iloc[-1]:,.2f}")
    print(f"  Model MAE    : ${results['mae']:,.2f}")
    print(f"  Baseline MAE : ${results['baseline_mae']:,.2f} (tomorrow = today)")
    if results["cv_mae"] is not None:
        print(f"  CV MAE       : ${results['cv_mae']:,.2f} (walk-forward)")
    print(f"  Model RMSE   : ${results['rmse']:,.2f}")
    print(f"\n  {args.forecast_days}-day forecast:")
    future = results["future_predictions"]
    print(f"    Start : ${future['predicted_price'].iloc[0]:,.2f}")
    print(f"    End   : ${future['predicted_price'].iloc[-1]:,.2f}")
    print("=" * 45)
    print(f"\nOutputs saved to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
