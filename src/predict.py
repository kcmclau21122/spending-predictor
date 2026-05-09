"""
Walk-forward prediction module.

Generates multi-step forecasts for per-category LightGBM models and a
global Prophet model, then ranks categories by projected spend.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LAG_MONTHS = [1, 2, 3, 6, 12]
_ROLLING_WINDOWS = [3, 6, 12]


def build_prediction_row(
    cat_df: pd.DataFrame,
    feature_cols: list[str],
    step: int,
) -> np.ndarray:
    """
    Construct the next-step feature vector from the extended history in cat_df.

    cat_df contains both original rows and any already-appended prediction rows.
    step is 0-indexed and indicates which prediction we are building.
    Returns a 1-D numpy array aligned with feature_cols.
    """
    history = cat_df["total"].tolist()
    last_date = cat_df["year_month"].iloc[-1]
    next_date = last_date + pd.DateOffset(months=1)

    month = next_date.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    quarter = float((month - 1) // 3 + 1)
    year = float(next_date.year)
    months_elapsed = float(cat_df["months_elapsed"].iloc[-1] + 1)

    lag_vals: dict[str, float] = {}
    for lag in _LAG_MONTHS:
        idx = len(history) - lag
        lag_vals[f"lag_{lag}m"] = float(history[idx]) if idx >= 0 else np.nan

    rolling_vals: dict[str, float] = {}
    for window in _ROLLING_WINDOWS:
        window_data = history[-window:]
        if len(window_data) == window:
            rolling_vals[f"rolling_mean_{window}m"] = float(np.mean(window_data))
            std = np.std(window_data, ddof=1) if len(window_data) > 1 else 0.0
            rolling_vals[f"rolling_std_{window}m"] = float(std)
        else:
            rolling_vals[f"rolling_mean_{window}m"] = np.nan
            rolling_vals[f"rolling_std_{window}m"] = np.nan

    if len(history) >= 13:
        yoy_base = history[-12]
        yoy_change = float(history[-1] - yoy_base)
        yoy_pct_change = float(yoy_change / (yoy_base + 1e-9))
    else:
        yoy_change = np.nan
        yoy_pct_change = np.nan

    feature_dict: dict[str, float] = {
        "month_sin": month_sin,
        "month_cos": month_cos,
        "quarter": quarter,
        "year": year,
        "months_elapsed": months_elapsed,
        **lag_vals,
        **rolling_vals,
        "yoy_change": yoy_change,
        "yoy_pct_change": yoy_pct_change,
    }
    return np.array([feature_dict.get(col, np.nan) for col in feature_cols], dtype=float)


def predict_next_months(
    models: dict,
    category_features: dict[str, pd.DataFrame],
    feature_cols: list[str],
    n_months: int = 12,
) -> dict[str, list[float]]:
    """
    Iteratively forecast n months ahead per category (walk-forward).

    For each step, the previous prediction is appended to the working DataFrame
    so that lag and rolling features are updated correctly before the next step.
    """
    results: dict[str, list[float]] = {}

    for cat, model in models.items():
        if cat not in category_features:
            logger.warning("No feature data for category '%s'; skipping.", cat)
            continue

        cat_df = category_features[cat].copy()
        predictions: list[float] = []

        for step in range(n_months):
            row = build_prediction_row(cat_df, feature_cols, step)
            pred = float(np.clip(model.predict(row.reshape(1, -1))[0], 0.0, None))
            predictions.append(pred)

            last_date = cat_df["year_month"].iloc[-1]
            next_date = last_date + pd.DateOffset(months=1)
            new_row = pd.DataFrame(
                [{
                    "year_month": next_date,
                    "total": pred,
                    "months_elapsed": cat_df["months_elapsed"].iloc[-1] + 1,
                }]
            )
            cat_df = pd.concat([cat_df, new_row], ignore_index=True)

        results[cat] = predictions

    return results


def prophet_forecast(prophet_model, periods: int = 12) -> pd.DataFrame:
    """Return a Prophet forecast DataFrame with ds, yhat, yhat_lower, yhat_upper."""
    future = prophet_model.make_future_dataframe(periods=periods, freq="MS")
    forecast = prophet_model.predict(future)
    return (
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        .tail(periods)
        .reset_index(drop=True)
    )


def rank_reduction_opportunities(
    forecasts: dict[str, list[float]],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Rank categories by average predicted monthly spend.
    Columns: category, avg_monthly, total_12m.
    """
    rows = [
        {
            "category": cat,
            "avg_monthly": float(np.mean(vals)),
            "total_12m": float(np.sum(vals)),
        }
        for cat, vals in forecasts.items()
    ]
    df = (
        pd.DataFrame(rows)
        .sort_values("avg_monthly", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return df
