"""
Unit tests for src/predict.py — Phase 4.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import lightgbm as lgb

from src.features import build_category_features, get_feature_columns
from src.predict import (
    build_prediction_row,
    predict_next_months,
    prophet_forecast,
    rank_reduction_opportunities,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_monthly_df(n_months: int = 24, start: str = "2022-01-01") -> pd.DataFrame:
    months = pd.date_range(start, periods=n_months, freq="MS")
    rng = np.random.default_rng(99)
    rows = []
    for m in months:
        for cat in ("groceries", "dining"):
            rows.append({
                "year_month": m,
                "category": cat,
                "total": float(rng.uniform(50, 500)),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def feature_cols():
    return get_feature_columns()


@pytest.fixture(scope="module")
def category_features():
    return build_category_features(_make_monthly_df())


@pytest.fixture(scope="module")
def trained_models(category_features, feature_cols):
    """Train a tiny LightGBM model per category."""
    models = {}
    for cat, df in category_features.items():
        X = df[feature_cols].values
        y = df["total"].values
        model = lgb.LGBMRegressor(n_estimators=10, verbose=-1, random_state=0)
        model.fit(X, y)
        models[cat] = model
    return models


@pytest.fixture(scope="module")
def forecasts(trained_models, category_features, feature_cols):
    return predict_next_months(trained_models, category_features, feature_cols, n_months=12)


# ── build_prediction_row ──────────────────────────────────────────────────────

def test_build_prediction_row_shape(category_features, feature_cols):
    cat_df = next(iter(category_features.values()))
    row = build_prediction_row(cat_df, feature_cols, step=0)
    assert row.shape == (len(feature_cols),)


def test_build_prediction_row_advances_month(category_features, feature_cols):
    cat_df = next(iter(category_features.values()))
    last_date = cat_df["year_month"].iloc[-1]
    row = build_prediction_row(cat_df, feature_cols, step=0)

    month_sin_idx = feature_cols.index("month_sin")
    expected_month = (last_date + pd.DateOffset(months=1)).month
    expected_sin = float(np.sin(2 * np.pi * expected_month / 12))
    assert abs(row[month_sin_idx] - expected_sin) < 1e-9


# ── predict_next_months ───────────────────────────────────────────────────────

def test_predict_next_months_length(forecasts):
    for cat, preds in forecasts.items():
        assert len(preds) == 12, f"Expected 12 predictions for '{cat}', got {len(preds)}"


def test_predictions_non_negative(forecasts):
    for cat, preds in forecasts.items():
        assert all(v >= 0 for v in preds), f"Negative prediction found for '{cat}'"


def test_predict_next_months_all_categories_returned(trained_models, forecasts):
    assert set(forecasts.keys()) == set(trained_models.keys())


def test_predict_next_months_custom_n(trained_models, category_features, feature_cols):
    result = predict_next_months(trained_models, category_features, feature_cols, n_months=3)
    for preds in result.values():
        assert len(preds) == 3


def test_predict_next_months_skips_missing_category(category_features, feature_cols):
    fake_model = lgb.LGBMRegressor(n_estimators=5, verbose=-1)
    # Train on one category's data just to have a fitted model
    cat_df = next(iter(category_features.values()))
    fake_model.fit(cat_df[feature_cols].values, cat_df["total"].values)

    result = predict_next_months(
        models={"nonexistent_cat": fake_model},
        category_features=category_features,
        feature_cols=feature_cols,
        n_months=6,
    )
    assert result == {}


# ── rank_reduction_opportunities ─────────────────────────────────────────────

def test_rank_reduction_shape(forecasts):
    df = rank_reduction_opportunities(forecasts, top_n=5)
    assert set(df.columns) == {"category", "avg_monthly", "total_12m"}
    assert len(df) <= 5


def test_rank_reduction_sorted_descending(forecasts):
    df = rank_reduction_opportunities(forecasts)
    avgs = df["avg_monthly"].tolist()
    assert avgs == sorted(avgs, reverse=True)


def test_rank_reduction_top_n_limits_rows():
    fake = {f"cat_{i}": [float(i * 10)] * 12 for i in range(20)}
    df = rank_reduction_opportunities(fake, top_n=7)
    assert len(df) == 7


def test_rank_reduction_columns_exist(forecasts):
    df = rank_reduction_opportunities(forecasts)
    for col in ("category", "avg_monthly", "total_12m"):
        assert col in df.columns


# ── prophet_forecast ──────────────────────────────────────────────────────────

def _make_mock_prophet(periods: int) -> MagicMock:
    future_dates = pd.date_range("2024-01-01", periods=periods, freq="MS")
    forecast_df = pd.DataFrame({
        "ds": future_dates,
        "yhat": [200.0] * periods,
        "yhat_lower": [150.0] * periods,
        "yhat_upper": [250.0] * periods,
        "extra_col": [0.0] * periods,
    })
    mock = MagicMock()
    mock.make_future_dataframe.return_value = pd.DataFrame({"ds": future_dates})
    mock.predict.return_value = forecast_df
    return mock


def test_prophet_forecast_columns():
    mock_model = _make_mock_prophet(12)
    result = prophet_forecast(mock_model, periods=12)
    assert set(result.columns) == {"ds", "yhat", "yhat_lower", "yhat_upper"}


def test_prophet_forecast_length():
    periods = 6
    mock_model = _make_mock_prophet(periods)
    result = prophet_forecast(mock_model, periods=periods)
    assert len(result) == periods


def test_prophet_forecast_no_extra_columns():
    mock_model = _make_mock_prophet(12)
    result = prophet_forecast(mock_model, periods=12)
    assert list(result.columns) == ["ds", "yhat", "yhat_lower", "yhat_upper"]
