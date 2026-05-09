"""
Unit/smoke tests for src/train.py — Phase 3.

Uses tiny synthetic data (24 months, 2 categories) so training completes in
seconds. MODELS_DIR is redirected to tmp_path via monkeypatch to avoid
touching the real models/ directory.
"""

import pytest
import pandas as pd
import numpy as np

import lightgbm as lgb

from src.features import build_category_features, get_feature_columns
from src.train import _train_lgbm_category, train_category_models, load_category_models


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_monthly_df(n_months: int = 24, start: str = "2022-01-01") -> pd.DataFrame:
    """Synthetic monthly expense DataFrame with groceries and dining."""
    months = pd.date_range(start, periods=n_months, freq="MS")
    rng = np.random.default_rng(7)
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
def category_features():
    monthly_df = _make_monthly_df()
    return build_category_features(monthly_df)


@pytest.fixture(scope="module")
def feature_cols():
    return get_feature_columns()


# ── _train_lgbm_category ──────────────────────────────────────────────────────

def test_train_lgbm_category_returns_model(category_features, feature_cols):
    cat = next(iter(category_features))
    model, metrics = _train_lgbm_category(cat, category_features[cat], feature_cols)
    assert isinstance(model, lgb.LGBMRegressor)
    assert isinstance(metrics, dict)


def test_metrics_keys(category_features, feature_cols):
    cat = next(iter(category_features))
    _, metrics = _train_lgbm_category(cat, category_features[cat], feature_cols)
    for key in ("category", "n_samples", "val_mae", "val_mape"):
        assert key in metrics, f"Metrics dict missing key {key!r}"


# ── train_category_models ─────────────────────────────────────────────────────

def test_train_category_models_saves_registry(monkeypatch, tmp_path, category_features, feature_cols):
    import src.train as train_mod
    monkeypatch.setattr(train_mod, "MODELS_DIR", tmp_path)

    train_mod.train_category_models(category_features, feature_cols)

    assert (tmp_path / "category_registry.joblib").exists()


# ── load_category_models ──────────────────────────────────────────────────────

def test_load_category_models_raises_if_missing(monkeypatch, tmp_path):
    import src.train as train_mod
    monkeypatch.setattr(train_mod, "MODELS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        load_category_models()


# ── train_prophet_model ───────────────────────────────────────────────────────

def test_train_prophet_model_saves(monkeypatch, tmp_path, sample_monthly_df):
    pytest.importorskip("prophet")

    from src.features import build_total_features, build_prophet_df
    from src.train import train_prophet_model
    import src.train as train_mod

    monkeypatch.setattr(train_mod, "MODELS_DIR", tmp_path)

    total_df = build_total_features(sample_monthly_df)
    prophet_df = build_prophet_df(total_df)
    train_prophet_model(prophet_df)

    assert (tmp_path / "prophet_total.joblib").exists()
