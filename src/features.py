"""
Feature engineering for spending time-series forecasting.

Takes monthly_summary output from ingest.py and produces model-ready features.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LAG_MONTHS = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [3, 6, 12]


def _add_time_features(df: pd.DataFrame, date_col: str = "year_month") -> pd.DataFrame:
    df = df.copy()
    df["month"] = df[date_col].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["quarter"] = df[date_col].dt.quarter
    df["year"] = df[date_col].dt.year
    # Ordinal month index (trend feature)
    min_date = df[date_col].min()
    df["months_elapsed"] = (
        (df[date_col].dt.year - min_date.year) * 12
        + (df[date_col].dt.month - min_date.month)
    )
    return df


def _add_lag_features(df: pd.DataFrame, target_col: str = "total") -> pd.DataFrame:
    df = df.copy()
    for lag in LAG_MONTHS:
        df[f"lag_{lag}m"] = df[target_col].shift(lag)
    return df


def _add_rolling_features(df: pd.DataFrame, target_col: str = "total") -> pd.DataFrame:
    df = df.copy()
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}m"] = df[target_col].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}m"] = df[target_col].shift(1).rolling(window).std()
    return df


def _add_yoy_change(df: pd.DataFrame, target_col: str = "total") -> pd.DataFrame:
    df = df.copy()
    df["yoy_change"] = df[target_col] - df[target_col].shift(12)
    df["yoy_pct_change"] = df["yoy_change"] / (df[target_col].shift(12) + 1e-9)
    return df


def build_category_features(monthly_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build per-category feature DataFrames for model training.

    Returns a dict mapping category name to feature DataFrame.
    Rows with NaN lag/rolling values (due to insufficient history) are dropped.
    """
    result: dict[str, pd.DataFrame] = {}
    categories = monthly_df["category"].unique()

    # Full month range so missing months get 0-filled
    all_months = pd.date_range(
        monthly_df["year_month"].min(),
        monthly_df["year_month"].max(),
        freq="MS",
    )

    for cat in categories:
        cat_df = (
            monthly_df[monthly_df["category"] == cat]
            .set_index("year_month")[["total"]]
            .reindex(all_months, fill_value=0.0)
            .rename_axis("year_month")
            .reset_index()
        )

        cat_df = _add_time_features(cat_df)
        cat_df = _add_lag_features(cat_df)
        cat_df = _add_rolling_features(cat_df)
        cat_df = _add_yoy_change(cat_df)
        cat_df["category"] = cat

        # Drop rows where core lag features are NaN (first 12 months)
        required_lags = [f"lag_{l}m" for l in LAG_MONTHS if l <= len(cat_df)]
        cat_df = cat_df.dropna(subset=required_lags).reset_index(drop=True)

        if len(cat_df) >= 3:
            result[cat] = cat_df
        else:
            logger.warning("Skipping category '%s' — insufficient data after lag generation.", cat)

    return result


def build_total_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all categories into total monthly spending and build features.
    Used by Prophet and the retirement gap model.
    """
    total = (
        monthly_df[monthly_df["category"] != "income"]
        .groupby("year_month")["total"]
        .sum()
        .reset_index()
    )
    total = total.sort_values("year_month").reset_index(drop=True)
    total = _add_time_features(total)
    total = _add_lag_features(total)
    total = _add_rolling_features(total)
    total = _add_yoy_change(total)
    return total


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature column names used for LightGBM."""
    cols = [
        "month_sin", "month_cos", "quarter", "year", "months_elapsed",
    ]
    cols += [f"lag_{l}m" for l in LAG_MONTHS]
    cols += [f"rolling_mean_{w}m" for w in ROLLING_WINDOWS]
    cols += [f"rolling_std_{w}m" for w in ROLLING_WINDOWS]
    cols += ["yoy_change", "yoy_pct_change"]
    return cols


def build_prophet_df(total_df: pd.DataFrame) -> pd.DataFrame:
    """Convert total_df to Prophet's expected ds/y format."""
    return total_df[["year_month", "total"]].rename(
        columns={"year_month": "ds", "total": "y"}
    )
