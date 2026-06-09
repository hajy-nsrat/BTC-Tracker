# Step 5: Visualize price history and predictions

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_price_history(df, save_path="outputs/price_history.png"):
    """
    Plots the full historical BTC price with moving averages.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df.index, df["price"], label="BTC Price", color="#F7931A", linewidth=1.5)

    if "ma_7" in df.columns:
        ax.plot(df.index, df["ma_7"],  label="7-day MA",  color="#3498db", linewidth=1, linestyle="--")
    if "ma_30" in df.columns:
        ax.plot(df.index, df["ma_30"], label="30-day MA", color="#9b59b6", linewidth=1, linestyle="--")

    ax.set_title("Bitcoin Price History", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Price history chart saved to {save_path}")


def plot_predictions(df_history, results, save_path="outputs/predictions.png"):
    """
    Plots actual price history alongside test predictions and future forecast.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Historical prices (last 90 days for clarity)
    history_tail = df_history["price"].tail(90)
    ax.plot(history_tail.index, history_tail.values,
            label="Historical Price", color="#F7931A", linewidth=1.5)

    # Test set: actual vs predicted
    test_df = results["test_results"]
    ax.plot(test_df.index, test_df["actual"],
            label="Actual (test period)", color="#2ecc71", linewidth=1.5)
    ax.plot(test_df.index, test_df["predicted"],
            label="Predicted (test period)", color="#e74c3c", linewidth=1.5, linestyle="--")

    if "baseline" in test_df.columns:
        ax.plot(test_df.index, test_df["baseline"],
                label="Baseline (test period)", color="#95a5a6", linewidth=1, linestyle=":")

    # Future predictions
    future_df = results["future_predictions"]
    ax.plot(future_df.index, future_df["predicted_price"],
            label="Future Forecast", color="#9b59b6", linewidth=2, linestyle="--")

    # Shade the forecast zone
    ax.axvspan(future_df.index[0], future_df.index[-1], alpha=0.05, color="#9b59b6")

    ax.set_title("BTC Price Prediction", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Predictions chart saved to {save_path}")


def plot_volatility(df, save_path="outputs/volatility.png"):
    """
    Plots the 7-day rolling volatility over time.
    """
    if "volatility_7" not in df.columns:
        print("No volatility column found, skipping.")
        return

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(df.index, df["volatility_7"], alpha=0.4, color="#e74c3c")
    ax.plot(df.index, df["volatility_7"], color="#e74c3c", linewidth=1)
    ax.set_title("BTC 7-Day Rolling Volatility", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Std Dev (USD)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Volatility chart saved to {save_path}")
