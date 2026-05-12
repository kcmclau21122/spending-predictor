"""
Drift detection for spending category models.

Compares recent actual spending to training-time statistics to flag
categories where model retraining is recommended.
"""

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
BASELINE_PATH = MODELS_DIR / "drift_baseline.joblib"

DRIFT_THRESHOLD = 0.30  # 30% mean shift triggers a retrain recommendation
RECENT_MONTHS = 3       # Recent window to compare against training baseline


def save_drift_baseline(category_features: dict[str, pd.DataFrame]) -> None:
    """Save per-category spending statistics as a drift detection baseline."""
    baseline = {}
    for cat, df in category_features.items():
        vals = df["total"].values
        baseline[cat] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "last_trained": datetime.now().isoformat(),
            "n_samples": int(len(vals)),
        }
    joblib.dump(baseline, BASELINE_PATH)
    logger.info("Drift baseline saved to %s", BASELINE_PATH)


def check_drift(
    category_features: dict[str, pd.DataFrame],
    recent_months: int = RECENT_MONTHS,
    threshold: float = DRIFT_THRESHOLD,
) -> list[dict]:
    """
    Compare recent actual spending to the training-time baseline.

    Returns a list of drift report dicts sorted by drift severity.
    Each dict has: category, trained_mean, recent_mean, drift_pct,
    needs_retrain, last_trained. Returns [] if no baseline exists.
    """
    if not BASELINE_PATH.exists():
        logger.warning("No drift baseline found — run 'train' first.")
        return []

    baseline = joblib.load(BASELINE_PATH)
    report = []

    for cat, df in category_features.items():
        if cat not in baseline:
            continue

        recent_vals = df.tail(recent_months)["total"].values
        if len(recent_vals) == 0:
            continue

        trained_mean = baseline[cat]["mean"]
        recent_mean = float(np.mean(recent_vals))

        if trained_mean > 1e-6:
            drift_pct = abs(recent_mean - trained_mean) / trained_mean
        else:
            drift_pct = 0.0 if recent_mean < 1e-6 else 1.0

        report.append({
            "category": cat,
            "trained_mean": trained_mean,
            "recent_mean": recent_mean,
            "drift_pct": drift_pct,
            "needs_retrain": drift_pct >= threshold,
            "last_trained": baseline[cat].get("last_trained", "unknown"),
        })

    report.sort(key=lambda x: x["drift_pct"], reverse=True)
    return report


def print_drift_report(drift_report: list[dict]) -> None:
    """Print drift report to console, flagging categories that need retraining."""
    if not drift_report:
        print("[drift] No baseline available — run 'train' first.")
        return

    flagged = [r for r in drift_report if r["needs_retrain"]]

    if flagged:
        plural = "y" if len(flagged) == 1 else "ies"
        print(f"\n[drift] WARNING: {len(flagged)} categor{plural} show significant drift (>30% mean shift):")
        for r in flagged:
            direction = "UP" if r["recent_mean"] > r["trained_mean"] else "DOWN"
            print(
                f"  ! {r['category']:22s} "
                f"trained avg: ${r['trained_mean']:8.2f}  "
                f"recent avg: ${r['recent_mean']:8.2f}  "
                f"drift: {r['drift_pct'] * 100:5.1f}% {direction}"
            )
        print("  -> Run 'train' to update models with your new spending data.\n")
    else:
        clean = len(drift_report)
        print(f"[drift] All {clean} categor{'y' if clean == 1 else 'ies'} within normal range — no retrain needed.")
