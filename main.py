# main.py — runs the full BTC tracker pipeline

from src.fetcher import fetch_btc_prices, save_to_csv
from src.cleaner import load_data, inspect_data, clean_data, save_clean_data
from src.features import add_features
from src.model import train_and_predict, save_predictions
from src.visualizer import plot_price_history, plot_predictions, plot_volatility


def main():
    print("=" * 45)
    print("        BTC Price Tracker & Predictor")
    print("=" * 45)

    # --- Step 1: Fetch ---
    print("\n[Step 1] Fetching BTC price data...")
    df_raw = fetch_btc_prices(days=365)
    if df_raw is None:
        print("Failed to fetch data. Exiting.")
        return
    save_to_csv(df_raw, filepath="data/btc_prices.csv")

    # --- Step 2: Clean ---
    print("\n[Step 2] Cleaning data...")
    df = load_data(filepath="data/btc_prices.csv")
    inspect_data(df)
    df_clean = clean_data(df)
    save_clean_data(df_clean, filepath="data/btc_prices_clean.csv")

    # --- Step 3: Feature Engineering ---
    print("\n[Step 3] Engineering features...")
    df_features = add_features(df_clean)

    # --- Step 4: Train & Predict ---
    print("\n[Step 4] Training model and generating predictions...")
    results = train_and_predict(df_features, forecast_days=30)
    save_predictions(results["future_predictions"], filepath="outputs/predictions.csv")

    # --- Step 5: Visualize ---
    print("\n[Step 5] Generating charts...")
    plot_price_history(df_features, save_path="outputs/price_history.png")
    plot_predictions(df_features, results, save_path="outputs/predictions.png")
    plot_volatility(df_features, save_path="outputs/volatility.png")

    # --- Summary ---
    print("\n" + "=" * 45)
    print("Pipeline complete!")
    print(f"  Date range : {df_clean.index.min().date()} → {df_clean.index.max().date()}")
    print(f"  Last price : ${df_clean['price'].iloc[-1]:,.2f}")
    print(f"  Model MAE  : ${results['mae']:,.2f}")
    print(f"  Model RMSE : ${results['rmse']:,.2f}")
    print(f"\n  30-day forecast:")
    future = results["future_predictions"]
    print(f"    Start : ${future['predicted_price'].iloc[0]:,.2f}")
    print(f"    End   : ${future['predicted_price'].iloc[-1]:,.2f}")
    print("=" * 45)
    print("\nOutputs saved to outputs/")


if __name__ == "__main__":
    main()
