import numpy as np
import pandas as pd
import pytest

from src.report import retirement_gap_analysis, generate_html_report


@pytest.fixture
def simple_forecasts():
    """Two categories, 12 monthly predictions each."""
    return {
        "groceries": [300.0] * 12,
        "dining": [100.0] * 12,
    }


@pytest.fixture
def simple_monthly_df():
    months = pd.date_range("2023-01-01", periods=12, freq="MS")
    rows = []
    for m in months:
        rows.append({"year_month": m, "category": "groceries", "total": 300.0})
        rows.append({"year_month": m, "category": "dining", "total": 100.0})
    return pd.DataFrame(rows)


@pytest.fixture
def simple_prophet_df():
    months = pd.date_range("2024-01-01", periods=12, freq="MS")
    return pd.DataFrame({
        "ds": months,
        "yhat": [400.0] * 12,
        "yhat_lower": [350.0] * 12,
        "yhat_upper": [450.0] * 12,
    })


@pytest.fixture
def simple_reduction_df():
    return pd.DataFrame({
        "category": ["groceries", "dining"],
        "avg_monthly": [300.0, 100.0],
        "total_12m": [3600.0, 1200.0],
    })


def test_retirement_gap_surplus(simple_forecasts):
    """Income clearly exceeds expenses → positive gap."""
    result = retirement_gap_analysis(
        forecasts=simple_forecasts,
        monthly_income=1000.0,
        current_savings=0.0,
        retirement_target=5000.0,
        months_to_retirement=24,
    )
    # monthly expenses = 400, income = 1000, surplus = 600/month
    # after 24 months: 600 * 24 = 14400 savings, target = 5000 → surplus gap
    assert result["gap"] > 0
    assert result["monthly_surplus_avg"] > 0


def test_retirement_gap_shortfall(simple_forecasts):
    """Income less than expenses → negative gap."""
    result = retirement_gap_analysis(
        forecasts=simple_forecasts,
        monthly_income=200.0,
        current_savings=0.0,
        retirement_target=1_000_000.0,
        months_to_retirement=12,
    )
    # monthly expenses = 400, income = 200 → deficit each month
    assert result["gap"] < 0
    assert result["monthly_surplus_avg"] < 0


def test_retirement_trajectory_length(simple_forecasts):
    """Trajectory list has exactly months_to_retirement entries."""
    months = 36
    result = retirement_gap_analysis(
        forecasts=simple_forecasts,
        monthly_income=600.0,
        current_savings=0.0,
        retirement_target=10000.0,
        months_to_retirement=months,
    )
    assert len(result["monthly_savings_trajectory"]) == months


def test_generate_html_creates_file(
    tmp_path, simple_forecasts, simple_monthly_df,
    simple_prophet_df, simple_reduction_df
):
    """generate_html_report creates a non-empty file at the given path."""
    retirement = retirement_gap_analysis(
        forecasts=simple_forecasts,
        monthly_income=800.0,
        current_savings=5000.0,
        retirement_target=100000.0,
        months_to_retirement=12,
    )
    out = tmp_path / "report.html"
    result_path = generate_html_report(
        monthly_df=simple_monthly_df,
        forecasts=simple_forecasts,
        prophet_df=simple_prophet_df,
        reduction_df=simple_reduction_df,
        retirement_result=retirement,
        output_path=out,
    )
    assert result_path.exists()
    assert result_path.stat().st_size > 0


def test_html_contains_plotly(
    tmp_path, simple_forecasts, simple_monthly_df,
    simple_prophet_df, simple_reduction_df
):
    """Output HTML references Plotly."""
    retirement = retirement_gap_analysis(
        forecasts=simple_forecasts,
        monthly_income=800.0,
        current_savings=5000.0,
        retirement_target=100000.0,
        months_to_retirement=12,
    )
    out = tmp_path / "report.html"
    result_path = generate_html_report(
        monthly_df=simple_monthly_df,
        forecasts=simple_forecasts,
        prophet_df=simple_prophet_df,
        reduction_df=simple_reduction_df,
        retirement_result=retirement,
        output_path=out,
    )
    content = result_path.read_text(encoding="utf-8")
    assert "plotly" in content.lower()
