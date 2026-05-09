"""
Unit tests for src/features.py — Phase 2.

Coverage:
  - Time feature generation (_add_time_features)
  - Lag feature generation (_add_lag_features)
  - Rolling feature generation (_add_rolling_features)
  - Category feature building with zero-fill (build_category_features)
  - Category exclusion for insufficient data (build_category_features)
  - Feature column list (get_feature_columns)
  - Prophet DataFrame conversion (build_prophet_df)
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    _add_lag_features,
    _add_rolling_features,
    _add_time_features,
    build_category_features,
    build_prophet_df,
    build_total_features,
    get_feature_columns,
    LAG_MONTHS,
    ROLLING_WINDOWS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_time_df(n_months: int, start: str = "2020-01-01") -> pd.DataFrame:
    months = pd.date_range(start, periods=n_months, freq="MS")
    return pd.DataFrame({
        "year_month": months,
        "total": [float(i + 1) * 10 for i in range(n_months)],
    })


def _make_monthly_df(n_months: int, categories: list, start: str = "2020-01-01") -> pd.DataFrame:
    months = pd.date_range(start, periods=n_months, freq="MS")
    rng = np.random.default_rng(0)
    rows = []
    for m in months:
        for cat in categories:
            rows.append({
                "year_month": m,
                "category": cat,
                "total": float(rng.uniform(50, 500)),
            })
    return pd.DataFrame(rows)


# ── _add_time_features ────────────────────────────────────────────────────────

def test_add_time_features_columns_present():
    df = _make_time_df(6)
    result = _add_time_features(df)
    for col in ["month_sin", "month_cos", "quarter", "year", "months_elapsed"]:
        assert col in result.columns, f"Expected column {col!r} in output"


def test_add_time_features_months_elapsed_starts_at_zero():
    df = _make_time_df(6)
    result = _add_time_features(df)
    assert result["months_elapsed"].iloc[0] == 0


def test_add_time_features_months_elapsed_increments_by_one():
    df = _make_time_df(6)
    result = _add_time_features(df)
    diffs = result["months_elapsed"].diff().dropna()
    assert (diffs == 1).all()


def test_add_time_features_sin_cos_in_range():
    df = _make_time_df(12)
    result = _add_time_features(df)
    assert result["month_sin"].between(-1.0, 1.0).all()
    assert result["month_cos"].between(-1.0, 1.0).all()


# ── _add_lag_features ─────────────────────────────────────────────────────────

def test_lag_features_first_row_is_nan():
    df = _make_time_df(6)
    result = _add_lag_features(df)
    assert pd.isna(result["lag_1m"].iloc[0])


def test_lag_features_second_row_equals_first_total():
    df = _make_time_df(6)
    result = _add_lag_features(df)
    assert result["lag_1m"].iloc[1] == pytest.approx(df["total"].iloc[0])


def test_lag_features_all_columns_present():
    df = _make_time_df(6)
    result = _add_lag_features(df)
    for lag in LAG_MONTHS:
        assert f"lag_{lag}m" in result.columns


# ── _add_rolling_features ─────────────────────────────────────────────────────

def test_rolling_features_columns_present():
    df = _make_time_df(10)
    result = _add_rolling_features(df)
    for w in ROLLING_WINDOWS:
        assert f"rolling_mean_{w}m" in result.columns
        assert f"rolling_std_{w}m" in result.columns


def test_rolling_mean_3m_nan_for_first_three_rows():
    """rolling_mean_3m uses shift(1).rolling(3) — needs 3 non-NaN shifted values."""
    df = _make_time_df(10)
    result = _add_rolling_features(df)
    assert result["rolling_mean_3m"].iloc[:3].isna().all()


def test_rolling_mean_3m_non_nan_at_row_three():
    """Row index 3 is the first with 3 shifted non-NaN values available."""
    df = _make_time_df(10)
    result = _add_rolling_features(df)
    assert not pd.isna(result["rolling_mean_3m"].iloc[3])


# ── build_category_features ───────────────────────────────────────────────────

def test_build_category_features_zero_fill(sample_monthly_df):
    """A category missing one month gets a zero-fill row that survives lag dropping."""
    all_months = sorted(sample_monthly_df["year_month"].unique())
    # Remove the 15th month (index 14) from groceries — it falls after the 12-row lag drop
    target_month = all_months[14]
    trimmed = sample_monthly_df[
        ~((sample_monthly_df["category"] == "groceries") & (sample_monthly_df["year_month"] == target_month))
    ].copy()

    result = build_category_features(trimmed)

    assert "groceries" in result, "Category with a gap month should still be present after zero-fill"
    groceries_df = result["groceries"]
    assert (groceries_df["total"] == 0.0).any(), "Zero-filled month must appear in output with total == 0"


def test_build_category_features_min_samples_excluded():
    """Category with <3 rows after lag generation is excluded from result."""
    # 14-month range: lag_12m NaN for first 12 rows → 2 remaining → excluded (<3)
    monthly_df = _make_monthly_df(14, ["groceries"])
    result = build_category_features(monthly_df)
    assert "groceries" not in result


def test_build_category_features_sufficient_samples_included():
    """Category with exactly 3 rows after lag generation (15-month range) is included."""
    # 15-month range: 15 - 12 = 3 rows → at the inclusion threshold
    monthly_df = _make_monthly_df(15, ["groceries"])
    result = build_category_features(monthly_df)
    assert "groceries" in result


def test_build_category_features_result_columns(sample_monthly_df):
    """All expected feature columns are present in each category DataFrame."""
    result = build_category_features(sample_monthly_df)
    assert result, "Result should not be empty"
    for cat_df in result.values():
        assert "total" in cat_df.columns
        assert "year_month" in cat_df.columns
        assert "category" in cat_df.columns
        break


# ── get_feature_columns ───────────────────────────────────────────────────────

def test_get_feature_columns_returns_list():
    cols = get_feature_columns()
    assert isinstance(cols, list)
    assert len(cols) > 0


def test_get_feature_columns_all_present_in_built_df(sample_monthly_df):
    """Every column returned by get_feature_columns() must exist in a built category DataFrame."""
    feature_cols = get_feature_columns()
    result = build_category_features(sample_monthly_df)
    assert result, "Need at least one category to validate feature columns"
    cat_df = next(iter(result.values()))
    for col in feature_cols:
        assert col in cat_df.columns, f"Feature column {col!r} missing from category DataFrame"


# ── build_prophet_df ──────────────────────────────────────────────────────────

def test_build_prophet_df_columns(sample_monthly_df):
    """Output of build_prophet_df must have exactly ds and y columns."""
    total_df = build_total_features(sample_monthly_df)
    prophet_df = build_prophet_df(total_df)
    assert list(prophet_df.columns) == ["ds", "y"]


def test_build_prophet_df_length(sample_monthly_df):
    """Output row count matches total_df row count."""
    total_df = build_total_features(sample_monthly_df)
    prophet_df = build_prophet_df(total_df)
    assert len(prophet_df) == len(total_df)


def test_build_prophet_df_ds_is_datetime(sample_monthly_df):
    """The ds column must be datetime dtype."""
    total_df = build_total_features(sample_monthly_df)
    prophet_df = build_prophet_df(total_df)
    assert pd.api.types.is_datetime64_any_dtype(prophet_df["ds"])
