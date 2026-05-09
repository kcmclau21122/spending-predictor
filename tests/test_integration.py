"""Integration test: full pipeline from raw CSV to HTML report."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def _make_synthetic_csv(path: Path) -> None:
    """Write 24 months of generic-format transactions covering 3 spend categories."""
    months = pd.date_range("2022-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(0)
    rows = []
    for m in months:
        rows.append({"Date": (m + pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
                     "Description": "KROGER #123",
                     "Amount": round(float(rng.uniform(200, 500)), 2)})
        rows.append({"Date": (m + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                     "Description": "STARBUCKS",
                     "Amount": round(float(rng.uniform(50, 150)), 2)})
        rows.append({"Date": (m + pd.Timedelta(days=15)).strftime("%Y-%m-%d"),
                     "Description": "AMAZON PURCHASE",
                     "Amount": round(float(rng.uniform(30, 200)), 2)})
        # income (negative = credit)
        rows.append({"Date": (m + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                     "Description": "PAYROLL DIRECT DEPOSIT",
                     "Amount": -5000.0})
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture()
def pipeline_workspace(tmp_path, monkeypatch):
    """Set up isolated data dir, models dir, reports dir and patch MODELS_DIR."""
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    _make_synthetic_csv(data_dir / "transactions.csv")

    # Redirect model persistence to tmp_path so tests don't pollute the repo
    import src.train as train_mod
    monkeypatch.setattr(train_mod, "MODELS_DIR", models_dir)

    return {
        "data_dir": data_dir,
        "models_dir": models_dir,
        "reports_dir": reports_dir,
    }


def test_full_pipeline_no_exception(pipeline_workspace):
    """Full run command completes without raising."""
    ws = pipeline_workspace
    from src.ingest import load_directory, monthly_summary
    from src.features import build_category_features, build_total_features, build_prophet_df, get_feature_columns
    from src.train import train_category_models, train_prophet_model
    from src.predict import predict_next_months, prophet_forecast, rank_reduction_opportunities
    from src.report import retirement_gap_analysis, generate_html_report
    import joblib

    # Ingest
    df = load_directory(ws["data_dir"])
    monthly_df = monthly_summary(df)
    assert not monthly_df.empty

    # Features
    category_features = build_category_features(monthly_df)
    total_df = build_total_features(monthly_df)
    feature_cols = get_feature_columns()
    prophet_df_input = build_prophet_df(total_df)
    assert len(category_features) >= 1

    # Train
    registry = train_category_models(category_features, feature_cols)
    models = {cat: joblib.load(info["path"]) for cat, info in registry.items()}
    assert len(models) >= 1

    # Prophet (optional — skip if not installed)
    prophet_model = None
    try:
        prophet_model = train_prophet_model(prophet_df_input)
    except Exception:
        pass

    # Predict
    forecasts = predict_next_months(models, category_features, feature_cols, n_months=12)
    assert len(forecasts) >= 1
    for cat, vals in forecasts.items():
        assert len(vals) == 12

    if prophet_model is not None:
        p_df = prophet_forecast(prophet_model, periods=12)
        assert set(["ds", "yhat", "yhat_lower", "yhat_upper"]).issubset(p_df.columns)
    else:
        p_df = pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])

    reduction_df = rank_reduction_opportunities(forecasts)
    assert not reduction_df.empty

    # Report
    retirement_result = retirement_gap_analysis(
        forecasts=forecasts,
        monthly_income=5000.0,
        current_savings=10_000.0,
        retirement_target=1_000_000.0,
        months_to_retirement=240,
    )
    report_path = generate_html_report(
        monthly_df=monthly_df,
        forecasts=forecasts,
        prophet_df=p_df,
        reduction_df=reduction_df,
        retirement_result=retirement_result,
        output_path=ws["reports_dir"] / "report.html",
    )
    assert report_path.exists()
    assert report_path.stat().st_size > 0


def test_models_saved(pipeline_workspace):
    """After training, category registry file exists in models dir."""
    ws = pipeline_workspace
    from src.ingest import load_directory, monthly_summary
    from src.features import build_category_features, get_feature_columns
    from src.train import train_category_models

    df = load_directory(ws["data_dir"])
    monthly_df = monthly_summary(df)
    category_features = build_category_features(monthly_df)
    feature_cols = get_feature_columns()
    train_category_models(category_features, feature_cols)

    assert (ws["models_dir"] / "category_registry.joblib").exists()


def test_html_report_created(pipeline_workspace):
    """Generated HTML report file is non-empty and contains 'plotly'."""
    ws = pipeline_workspace
    from src.ingest import load_directory, monthly_summary
    from src.features import build_category_features, get_feature_columns
    from src.train import train_category_models
    from src.predict import predict_next_months, rank_reduction_opportunities
    from src.report import retirement_gap_analysis, generate_html_report
    import joblib
    import pandas as pd

    df = load_directory(ws["data_dir"])
    monthly_df = monthly_summary(df)
    category_features = build_category_features(monthly_df)
    feature_cols = get_feature_columns()
    registry = train_category_models(category_features, feature_cols)
    models = {cat: joblib.load(info["path"]) for cat, info in registry.items()}
    forecasts = predict_next_months(models, category_features, feature_cols, n_months=12)
    reduction_df = rank_reduction_opportunities(forecasts)
    retirement_result = retirement_gap_analysis(
        forecasts=forecasts,
        monthly_income=5000.0,
        current_savings=0.0,
        retirement_target=500_000.0,
        months_to_retirement=120,
    )
    empty_prophet = pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
    report_path = generate_html_report(
        monthly_df=monthly_df,
        forecasts=forecasts,
        prophet_df=empty_prophet,
        reduction_df=reduction_df,
        retirement_result=retirement_result,
        output_path=ws["reports_dir"] / "report.html",
    )

    assert report_path.exists()
    html_text = report_path.read_text(encoding="utf-8")
    assert len(html_text) > 100
    assert "plotly" in html_text.lower()


def test_predictions_non_negative(pipeline_workspace):
    """All predicted spending values are >= 0."""
    ws = pipeline_workspace
    from src.ingest import load_directory, monthly_summary
    from src.features import build_category_features, get_feature_columns
    from src.train import train_category_models
    from src.predict import predict_next_months
    import joblib

    df = load_directory(ws["data_dir"])
    monthly_df = monthly_summary(df)
    category_features = build_category_features(monthly_df)
    feature_cols = get_feature_columns()
    registry = train_category_models(category_features, feature_cols)
    models = {cat: joblib.load(info["path"]) for cat, info in registry.items()}
    forecasts = predict_next_months(models, category_features, feature_cols, n_months=12)

    for cat, vals in forecasts.items():
        assert all(v >= 0 for v in vals), f"Negative prediction for {cat}: {vals}"
