"""
Report generation module.

Provides retirement gap analysis and HTML report generation with embedded
Plotly charts rendered via Jinja2 templates.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Spending Predictor Report</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
    h1 { color: #333; }
    h2 { color: #555; margin-top: 40px; }
    .chart { background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    table { border-collapse: collapse; width: 100%; background: #fff; }
    th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: right; }
    th { background: #4a90d9; color: #fff; text-align: center; }
    td:first-child { text-align: left; }
    .summary-box { background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .gap-positive { color: #27ae60; font-weight: bold; }
    .gap-negative { color: #e74c3c; font-weight: bold; }
    .drift-ok { color: #27ae60; }
    .drift-warn { color: #e74c3c; font-weight: bold; }
    .drift-banner { background: #fff3cd; border-left: 4px solid #f0ad4e; padding: 12px 20px; border-radius: 4px; margin-bottom: 16px; }
  </style>
</head>
<body>
  <h1>Spending Predictor Report</h1>

  <div class="summary-box">
    <h2>Retirement Gap Analysis</h2>
    <p>Projected savings at retirement: <strong>${{ "%.2f"|format(retirement.projected_savings_at_retirement) }}</strong></p>
    <p>Retirement target: <strong>${{ "%.2f"|format(retirement_target) }}</strong></p>
    <p>Gap:
      {% if retirement.gap >= 0 %}
        <span class="gap-positive">${{ "%.2f"|format(retirement.gap) }} surplus</span>
      {% else %}
        <span class="gap-negative">${{ "%.2f"|format(retirement.gap) }} shortfall</span>
      {% endif %}
    </p>
    <p>Average monthly surplus: <strong>${{ "%.2f"|format(retirement.monthly_surplus_avg) }}</strong></p>
  </div>

  <h2>Last 12 Months — Actual Spend by Category</h2>
  <div class="chart" id="chart-actual"></div>

  <h2>Next 12 Months — Forecast by Category</h2>
  <div class="chart" id="chart-forecast"></div>

  <h2>Total Spending Trend (Actual + Prophet Forecast)</h2>
  <div class="chart" id="chart-prophet"></div>

  <h2>Retirement Savings Trajectory</h2>
  <div class="chart" id="chart-retirement"></div>

  <h2>Top Reduction Opportunities</h2>
  <div class="chart">
    <table>
      <tr><th>Category</th><th>Avg Monthly ($)</th><th>Annual Total ($)</th></tr>
      {% for _, row in reduction_df.iterrows() %}
      <tr>
        <td>{{ row.category }}</td>
        <td>${{ "%.2f"|format(row.avg_monthly) }}</td>
        <td>${{ "%.2f"|format(row.total_12m) }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>

  {% if drift_report %}
  <h2>Model Drift Status</h2>
  <div class="summary-box">
    {% set flagged = drift_report | selectattr("needs_retrain") | list %}
    {% if flagged %}
    <div class="drift-banner">
      <strong>Warning:</strong> {{ flagged|length }} categor{{ "y" if flagged|length == 1 else "ies" }}
      show &gt;30% spending drift from training baseline — consider retraining.
    </div>
    {% else %}
    <p class="drift-ok">All categories are within 30% of their training baseline. No retrain needed.</p>
    {% endif %}
    <table>
      <tr><th>Category</th><th>Trained Avg ($)</th><th>Recent Avg ($)</th><th>Drift</th><th>Status</th></tr>
      {% for r in drift_report %}
      <tr>
        <td>{{ r.category }}</td>
        <td>${{ "%.2f"|format(r.trained_mean) }}</td>
        <td>${{ "%.2f"|format(r.recent_mean) }}</td>
        <td>{{ "%.1f"|format(r.drift_pct * 100) }}%</td>
        <td>
          {% if r.needs_retrain %}
            <span class="drift-warn">Retrain recommended</span>
          {% else %}
            <span class="drift-ok">OK</span>
          {% endif %}
        </td>
      </tr>
      {% endfor %}
    </table>
  </div>
  {% endif %}

  <script>
    {{ plotly_scripts }}
  </script>
</body>
</html>
"""


def retirement_gap_analysis(
    forecasts: dict[str, list[float]],
    monthly_income: float,
    current_savings: float,
    retirement_target: float,
    months_to_retirement: int,
) -> dict:
    """
    Compute retirement savings trajectory and gap vs. target.

    Returns dict with keys:
      projected_savings_at_retirement, gap, monthly_surplus_avg,
      monthly_savings_trajectory
    """
    # Sum all forecast categories per month (expense categories only)
    n_months = months_to_retirement
    category_lists = list(forecasts.values())

    trajectory: list[float] = []
    savings = current_savings

    for month_idx in range(n_months):
        total_expenses = 0.0
        for cat_vals in category_lists:
            if month_idx < len(cat_vals):
                total_expenses += cat_vals[month_idx]
            elif cat_vals:
                total_expenses += cat_vals[-1]

        surplus = monthly_income - total_expenses
        savings += surplus
        trajectory.append(float(savings))

    projected = float(trajectory[-1]) if trajectory else current_savings
    gap = projected - retirement_target
    surpluses = [monthly_income - sum(
        (cat_vals[i] if i < len(cat_vals) else (cat_vals[-1] if cat_vals else 0.0))
        for cat_vals in category_lists
    ) for i in range(n_months)]
    monthly_surplus_avg = float(np.mean(surpluses)) if surpluses else 0.0

    return {
        "projected_savings_at_retirement": projected,
        "gap": gap,
        "monthly_surplus_avg": monthly_surplus_avg,
        "monthly_savings_trajectory": trajectory,
    }


def generate_html_report(
    monthly_df: pd.DataFrame,
    forecasts: dict[str, list[float]],
    prophet_df: pd.DataFrame,
    reduction_df: pd.DataFrame,
    retirement_result: dict,
    output_path: "str | Path",
    drift_report: "list[dict] | None" = None,
) -> Path:
    """Render a Jinja2 HTML report with embedded Plotly charts."""
    try:
        from jinja2 import Environment, BaseLoader
    except ImportError as exc:
        raise ImportError("jinja2 is required for report generation") from exc

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError("plotly is required for report generation") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scripts: list[str] = []

    # Chart 1 — last 12 months actual spend by category
    last_12 = monthly_df.copy()
    last_12 = last_12[last_12["category"] != "income"].copy()
    if not last_12.empty:
        last_12 = last_12.sort_values("year_month").groupby("category").tail(12)
        fig1 = go.Figure()
        for cat, grp in last_12.groupby("category"):
            fig1.add_trace(go.Bar(
                x=grp["year_month"].astype(str).tolist(),
                y=grp["total"].tolist(),
                name=str(cat),
            ))
        fig1.update_layout(barmode="stack", title="Actual Spend by Category (Last 12 Months)")
        scripts.append(f"Plotly.newPlot('chart-actual', {fig1.to_json()}.data, {fig1.to_json()}.layout);")

    # Chart 2 — forecast by category
    if forecasts:
        fig2 = go.Figure()
        for cat, vals in forecasts.items():
            fig2.add_trace(go.Scatter(
                x=list(range(1, len(vals) + 1)),
                y=vals,
                mode="lines+markers",
                name=str(cat),
            ))
        fig2.update_layout(title="Forecast Spend by Category (Next 12 Months)", xaxis_title="Month")
        scripts.append(f"Plotly.newPlot('chart-forecast', {fig2.to_json()}.data, {fig2.to_json()}.layout);")

    # Chart 3 — Prophet total trend
    if prophet_df is not None and not prophet_df.empty:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=prophet_df["ds"].astype(str).tolist(),
            y=prophet_df["yhat"].tolist(),
            mode="lines",
            name="Forecast",
            line={"color": "blue"},
        ))
        fig3.add_trace(go.Scatter(
            x=prophet_df["ds"].astype(str).tolist() + prophet_df["ds"].astype(str).tolist()[::-1],
            y=prophet_df["yhat_upper"].tolist() + prophet_df["yhat_lower"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(0,100,255,0.1)",
            line={"color": "rgba(255,255,255,0)"},
            name="Confidence Band",
        ))
        fig3.update_layout(title="Total Spending Trend (Prophet)")
        scripts.append(f"Plotly.newPlot('chart-prophet', {fig3.to_json()}.data, {fig3.to_json()}.layout);")

    # Chart 4 — retirement savings trajectory
    traj = retirement_result.get("monthly_savings_trajectory", [])
    if traj:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=list(range(1, len(traj) + 1)),
            y=traj,
            mode="lines",
            name="Projected Savings",
        ))
        target = retirement_result.get("projected_savings_at_retirement", 0) - retirement_result.get("gap", 0)
        fig4.add_hline(y=target, line_dash="dash", line_color="red", annotation_text="Target")
        fig4.update_layout(title="Retirement Savings Trajectory", xaxis_title="Month")
        scripts.append(f"Plotly.newPlot('chart-retirement', {fig4.to_json()}.data, {fig4.to_json()}.layout);")

    retirement_target = (
        retirement_result.get("projected_savings_at_retirement", 0)
        - retirement_result.get("gap", 0)
    )

    env = Environment(loader=BaseLoader())
    tmpl = env.from_string(_HTML_TEMPLATE)
    html = tmpl.render(
        retirement=retirement_result,
        retirement_target=retirement_target,
        reduction_df=reduction_df,
        drift_report=drift_report or [],
        plotly_scripts="\n".join(scripts),
    )

    output_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", output_path)
    return output_path
