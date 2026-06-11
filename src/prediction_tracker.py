# Track forward forecasts across runs and score them against actual prices

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LOG_COLUMNS = ["run_date", "target_date", "horizon_days", "predicted_price"]
SCORE_COLUMNS = [
    "run_date",
    "target_date",
    "horizon_days",
    "predicted_price",
    "actual_price",
    "error",
    "error_pct",
    "scored_at",
]


def _normalize_date(value) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def append_prediction_log(
    future_df: pd.DataFrame,
    run_date,
    log_path: str = "outputs/prediction_log.csv",
) -> None:
    """Append this run's forward forecast to the prediction log."""
    run_ts = _normalize_date(run_date)

    rows = []
    for target_date, row in future_df.iterrows():
        target_ts = _normalize_date(target_date)
        horizon = (target_ts - run_ts).days
        if horizon < 1:
            continue
        rows.append(
            {
                "run_date": run_ts,
                "target_date": target_ts,
                "horizon_days": horizon,
                "predicted_price": row["predicted_price"],
            }
        )

    if not rows:
        return

    new_batch = pd.DataFrame(rows)

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path, parse_dates=["run_date", "target_date"])
        existing = existing[existing["run_date"] != run_ts]
        log = pd.concat([existing, new_batch], ignore_index=True)
    else:
        log = new_batch

    log = log.sort_values(["run_date", "horizon_days"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log.to_csv(log_path, index=False)
    logger.info("Logged %d forecasts for run_date %s → %s", len(new_batch), run_ts.date(), log_path)


def score_predictions(
    actual_prices: pd.Series,
    log_path: str = "outputs/prediction_log.csv",
    scores_path: str = "outputs/prediction_scores.csv",
) -> dict:
    """
    Score matured log entries against actual prices.

    Returns a summary dict with overall stats and recent next-day (horizon=1) results.
    """
    if not os.path.exists(log_path):
        logger.info("No prediction log found at %s — skipping score step.", log_path)
        return {"scored_count": 0, "new_scores": pd.DataFrame(), "horizon_1": pd.DataFrame()}

    log = pd.read_csv(log_path, parse_dates=["run_date", "target_date"])
    actual = actual_prices.copy()
    actual.index = pd.DatetimeIndex(actual.index).normalize()

    last_actual_date = actual.index.max()
    matured = log[log["target_date"] <= last_actual_date].copy()

    if matured.empty:
        logger.info("No matured predictions to score yet.")
        return {"scored_count": 0, "new_scores": pd.DataFrame(), "horizon_1": pd.DataFrame()}

    if os.path.exists(scores_path):
        scored = pd.read_csv(scores_path, parse_dates=["run_date", "target_date", "scored_at"])
        scored_keys = set(
            zip(scored["run_date"], scored["target_date"], scored["horizon_days"])
        )
    else:
        scored = pd.DataFrame(columns=SCORE_COLUMNS)
        scored_keys = set()

    to_score = matured[
        ~matured.apply(
            lambda r: (r["run_date"], r["target_date"], r["horizon_days"]) in scored_keys,
            axis=1,
        )
    ]

    new_scores = pd.DataFrame()

    if to_score.empty:
        logger.info("All matured predictions already scored.")
        all_scores = scored
    else:
        new_rows = []
        scored_at = pd.Timestamp.now()

        for _, row in to_score.iterrows():
            target = _normalize_date(row["target_date"])
            if target not in actual.index:
                continue

            actual_price = float(actual.loc[target])
            predicted = float(row["predicted_price"])
            error = predicted - actual_price
            error_pct = (error / actual_price) * 100 if actual_price else np.nan

            new_rows.append(
                {
                    "run_date": row["run_date"],
                    "target_date": target,
                    "horizon_days": int(row["horizon_days"]),
                    "predicted_price": predicted,
                    "actual_price": actual_price,
                    "error": error,
                    "error_pct": error_pct,
                    "scored_at": scored_at,
                }
            )

        new_scores = pd.DataFrame(new_rows)
        if not new_scores.empty:
            all_scores = pd.concat([scored, new_scores], ignore_index=True)
            all_scores = all_scores.sort_values(
                ["target_date", "run_date", "horizon_days"]
            ).reset_index(drop=True)

            os.makedirs(os.path.dirname(scores_path) or ".", exist_ok=True)
            all_scores.to_csv(scores_path, index=False)
            logger.info("Scored %d new prediction(s) → %s", len(new_scores), scores_path)
        else:
            all_scores = scored

    horizon_1 = all_scores[all_scores["horizon_days"] == 1].copy()
    if not horizon_1.empty:
        horizon_1 = horizon_1.sort_values("target_date")

    summary = {
        "scored_count": len(new_scores),
        "new_scores": new_scores,
        "horizon_1": horizon_1,
        "all_scores": all_scores,
    }

    if not horizon_1.empty:
        summary["horizon_1_mae"] = horizon_1["error"].abs().mean()
        summary["horizon_1_count"] = len(horizon_1)

    return summary


def print_score_report(summary: dict) -> None:
    """Print a human-readable report of prediction vs actual results."""
    horizon_1 = summary.get("horizon_1", pd.DataFrame())

    if horizon_1 is None or horizon_1.empty:
        print("\n[Prediction tracker] No scored next-day forecasts yet.")
        print("  Run daily and scores will appear once target dates pass.")
        return

    print("\n[Prediction tracker] Next-day forecast accuracy (horizon = 1 day)")

    if "horizon_1_mae" in summary:
        print(f"  Tracked forecasts : {summary['horizon_1_count']}")
        print(f"  Mean abs error    : ${summary['horizon_1_mae']:,.2f}")

    recent = horizon_1.tail(7)
    print("\n  Recent results:")
    for _, row in recent.iterrows():
        run_d = pd.Timestamp(row["run_date"]).date()
        target_d = pd.Timestamp(row["target_date"]).date()
        print(
            f"    {run_d} → {target_d}: "
            f"predicted ${row['predicted_price']:,.2f}, "
            f"actual ${row['actual_price']:,.2f}, "
            f"error ${row['error']:+,.2f} ({row['error_pct']:+.2f}%)"
        )

    latest = horizon_1.iloc[-1]
    print(
        f"\n  Latest: on {pd.Timestamp(latest['run_date']).date()} "
        f"predicted {pd.Timestamp(latest['target_date']).date()} "
        f"at ${latest['predicted_price']:,.2f} — "
        f"actual ${latest['actual_price']:,.2f} "
        f"(${latest['error']:+,.2f})"
    )
