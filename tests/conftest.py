import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_transaction_df():
    """24 months of synthetic transactions: groceries, dining, and income."""
    months = pd.date_range("2022-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(42)
    rows = []
    for m in months:
        rows.append({
            "date": m + pd.Timedelta(days=5),
            "description": "KROGER #123",
            "amount": float(rng.uniform(200, 500)),
            "category": "groceries",
            "source": "test",
            "format": "generic",
        })
        rows.append({
            "date": m + pd.Timedelta(days=10),
            "description": "STARBUCKS",
            "amount": float(rng.uniform(50, 150)),
            "category": "dining",
            "source": "test",
            "format": "generic",
        })
        rows.append({
            "date": m + pd.Timedelta(days=1),
            "description": "PAYROLL DIRECT DEPOSIT",
            "amount": -5000.0,
            "category": "income",
            "source": "test",
            "format": "generic",
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


@pytest.fixture
def sample_monthly_df(sample_transaction_df):
    """Monthly summary (expenses only) built from sample transactions."""
    from src.ingest import monthly_summary
    return monthly_summary(sample_transaction_df)
