"""
Model training module.

Trains:
  1. Per-category LightGBM models (monthly spending forecast per category)
  2. Prophet model (overall spending trend + seasonality)

Models are saved to the models/ directory as joblib files.
"""

import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

import lightgbm as lgb
from src.drift import save_drift_baseline

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 5,
    "device": "cpu",  # Switch to "gpu" if lgb GPU build installed
    "verbose": -1,
    "random_state": 42,
}


def _train_lgbm_category(
    cat: str, cat_df: pd.DataFrame, feature_cols: list[str]
) -> tuple[lgb.LGBMRegressor, dict]:
    """Train a LightGBM model for a single spending category."""
    df = cat_df.dropna(subset=feature_cols).copy()
    X = df[feature_cols].values
    y = df["total"].values

    tscv = TimeSeriesSplit(n_splits=min(3, len(df) - 1))
    val_maes, val_mapes = [], []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        preds = np.clip(model.predict(X_val), 0, None)
        val_maes.append(mean_absolute_error(y_val, preds))
        if y_val.mean() > 0:
            val_mapes.append(mean_absolute_percentage_error(y_val, preds))

    # Final model on full data
    final_model = lgb.LGBMRegressor(**LGBM_PARAMS)
    final_model.fit(X, y, callbacks=[lgb.log_evaluation(-1)])

    metrics = {
        "category": cat,
        "n_samples": len(df),
        "val_mae": float(np.mean(val_maes)) if val_maes else None,
        "val_mape": float(np.mean(val_mapes)) if val_mapes else None,
    }
    return final_model, metrics


def train_category_models(
    category_features: dict[str, pd.DataFrame],
    feature_cols: list[str],
) -> dict:
    """Train LightGBM models for all categories. Returns model registry dict."""
    registry = {}
    all_metrics = []

    for cat, df in category_features.items():
        logger.info("Training LightGBM for category: %s (%d samples)", cat, len(df))
        try:
            model, metrics = _train_lgbm_category(cat, df, feature_cols)
            path = MODELS_DIR / f"lgbm_{cat}.joblib"
            joblib.dump(model, path)
            registry[cat] = {"path": str(path), "metrics": metrics}
            all_metrics.append(metrics)
            logger.info("  MAE=%.2f  MAPE=%.1f%%", metrics["val_mae"] or 0, (metrics["val_mape"] or 0) * 100)
        except Exception as e:
            logger.error("Failed to train %s: %s", cat, e)

    joblib.dump(registry, MODELS_DIR / "category_registry.joblib")
    save_drift_baseline(category_features)
    logger.info("Saved %d category models to %s", len(registry), MODELS_DIR)
    return registry


def train_prophet_model(prophet_df: pd.DataFrame) -> object:
    """Train a Prophet model on overall monthly spending."""
    try:
        from prophet import Prophet
    except ImportError:
        logger.error("Prophet not installed. Run: pip install prophet")
        return None

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.1,
        uncertainty_samples=200,
    )
    model.fit(prophet_df)

    path = MODELS_DIR / "prophet_total.joblib"
    joblib.dump(model, path)
    logger.info("Prophet model saved to %s", path)
    return model


def load_category_models() -> tuple[dict, dict[str, lgb.LGBMRegressor]]:
    """Load all trained category models from disk."""
    registry_path = MODELS_DIR / "category_registry.joblib"
    if not registry_path.exists():
        raise FileNotFoundError("No trained models found. Run train first.")

    registry = joblib.load(registry_path)
    models = {}
    for cat, info in registry.items():
        models[cat] = joblib.load(info["path"])
    return registry, models


def load_prophet_model():
    """Load the trained Prophet model from disk."""
    path = MODELS_DIR / "prophet_total.joblib"
    if not path.exists():
        raise FileNotFoundError("Prophet model not found. Run train first.")
    return joblib.load(path)
