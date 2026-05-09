"""
Unit tests for src/ingest.py — Phase 1.

Coverage:
  - Format auto-detection (_detect_format)
  - Category assignment (_assign_category)
  - CSV loading and schema validation (load_csv)
  - Edge cases: zero-amount rows, source name, income sign
  - Income/expense splitting (split_income_expenses)
  - Monthly aggregation (monthly_summary)
  - Directory loading with deduplication (load_directory)
"""

import textwrap

import pandas as pd
import pytest

from src.ingest import (
    _assign_category,
    _detect_format,
    load_csv,
    load_directory,
    monthly_summary,
    split_income_expenses,
)

# ── Sample CSV strings ────────────────────────────────────────────────────────

CHASE_CSV = textwrap.dedent("""\
    Transaction Date,Post Date,Description,Category,Type,Amount,Memo
    01/15/2024,01/16/2024,KROGER #123,Food & Drink,Sale,-45.67,
    01/20/2024,01/21/2024,AMAZON.COM,Shopping,Sale,-32.10,
    01/25/2024,01/25/2024,PAYROLL DIRECT DEPOSIT,Income,ACH_CREDIT,3500.00,
""")

BOFA_CSV = textwrap.dedent("""\
    Posted Date,Reference Number,Payee,Address,Amount
    01/15/2024,REF001,KROGER #123,,45.67
    01/20/2024,REF002,PAYROLL,,
""")

GENERIC_CSV = textwrap.dedent("""\
    Date,Description,Amount
    2024-01-10,KROGER #123,50.00
    2024-01-20,STARBUCKS,12.50
""")


# ── _detect_format ────────────────────────────────────────────────────────────

def test_detect_format_chase():
    df = pd.DataFrame(columns=["Transaction Date", "Post Date", "Description", "Category"])
    assert _detect_format(df) == "chase"


def test_detect_format_bofa():
    df = pd.DataFrame(columns=["Posted Date", "Reference Number", "Payee"])
    assert _detect_format(df) == "bofa"


def test_detect_format_amex_card_member():
    df = pd.DataFrame(columns=["Date", "Description", "Amount", "Card Member"])
    assert _detect_format(df) == "amex"


def test_detect_format_capital_one():
    df = pd.DataFrame(columns=["Transaction Date", "Transaction Type", "Category", "Memo"])
    assert _detect_format(df) == "capital_one"


def test_detect_format_generic():
    df = pd.DataFrame(columns=["Date", "Description", "Amount"])
    assert _detect_format(df) == "generic"


# ── _assign_category ──────────────────────────────────────────────────────────

def test_assign_category_groceries():
    assert _assign_category("KROGER #523 PURCHASE", "") == "groceries"


def test_assign_category_dining():
    assert _assign_category("STARBUCKS STORE 12345", "") == "dining"


def test_assign_category_income():
    assert _assign_category("PAYROLL DIRECT DEPOSIT", "") == "income"


def test_assign_category_gas():
    assert _assign_category("SHELL OIL 12345", "") == "gas"


def test_assign_category_streaming():
    assert _assign_category("NETFLIX.COM", "") == "streaming"


def test_assign_category_other():
    assert _assign_category("ZZZUNKNOWN VENDOR XYZ", "") == "other"


def test_assign_category_uses_bank_category_hint(tmp_path):
    # Description alone is ambiguous; bank_category hint can match a rule
    result = _assign_category("PURCHASE 12345", "groceries kroger")
    assert result == "groceries"


# ── load_csv — Chase ──────────────────────────────────────────────────────────

def test_load_csv_chase_output_schema(tmp_path):
    f = tmp_path / "chase.csv"
    f.write_text(CHASE_CSV)
    df = load_csv(f)
    assert list(df.columns) == ["date", "description", "amount", "category", "source", "format"]


def test_load_csv_chase_expense_is_positive(tmp_path):
    f = tmp_path / "chase.csv"
    f.write_text(CHASE_CSV)
    df = load_csv(f)
    expenses = df[df["category"] != "income"]
    assert (expenses["amount"] > 0).all(), "Chase expense amounts should be positive after sign flip"


def test_load_csv_chase_income_is_negative(tmp_path):
    f = tmp_path / "chase.csv"
    f.write_text(CHASE_CSV)
    df = load_csv(f)
    income_rows = df[df["category"] == "income"]
    assert len(income_rows) == 1
    assert income_rows.iloc[0]["amount"] < 0, "Chase income should be negative after sign flip"


def test_load_csv_source_equals_file_stem(tmp_path):
    f = tmp_path / "mybank_2024.csv"
    f.write_text(CHASE_CSV)
    df = load_csv(f)
    assert (df["source"] == "mybank_2024").all()


def test_load_csv_format_tag_recorded(tmp_path):
    f = tmp_path / "chase.csv"
    f.write_text(CHASE_CSV)
    df = load_csv(f)
    assert (df["format"] == "chase").all()


def test_load_csv_drops_zero_amount(tmp_path):
    csv = textwrap.dedent("""\
        Transaction Date,Post Date,Description,Category,Type,Amount,Memo
        01/15/2024,01/16/2024,KROGER #123,Food & Drink,Sale,-45.67,
        01/20/2024,01/21/2024,ZERO TXN,Shopping,Sale,0.00,
    """)
    f = tmp_path / "chase.csv"
    f.write_text(csv)
    df = load_csv(f)
    assert len(df) == 1, "Zero-amount rows must be dropped"


def test_load_csv_date_is_datetime(tmp_path):
    f = tmp_path / "chase.csv"
    f.write_text(CHASE_CSV)
    df = load_csv(f)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


# ── load_csv — generic ────────────────────────────────────────────────────────

def test_load_csv_generic_schema(tmp_path):
    f = tmp_path / "generic.csv"
    f.write_text(GENERIC_CSV)
    df = load_csv(f)
    assert list(df.columns) == ["date", "description", "amount", "category", "source", "format"]
    assert len(df) == 2


# ── split_income_expenses ─────────────────────────────────────────────────────

def test_split_income_expenses_counts(tmp_path):
    f = tmp_path / "chase.csv"
    f.write_text(CHASE_CSV)
    df = load_csv(f)
    income, expenses = split_income_expenses(df)
    assert len(income) == 1
    assert len(expenses) == 2


def test_split_income_expenses_no_overlap(tmp_path):
    f = tmp_path / "chase.csv"
    f.write_text(CHASE_CSV)
    df = load_csv(f)
    income, expenses = split_income_expenses(df)
    assert len(income) + len(expenses) == len(df)


def test_split_income_expenses_negative_amounts_go_to_income():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),
        "description": ["REFUND"],
        "amount": [-20.0],
        "category": ["other"],
        "source": ["test"],
        "format": ["generic"],
    })
    income, expenses = split_income_expenses(df)
    assert len(income) == 1
    assert len(expenses) == 0


# ── monthly_summary ───────────────────────────────────────────────────────────

def test_monthly_summary_output_columns():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-10", "2024-01-20"]),
        "description": ["KROGER", "TARGET"],
        "amount": [50.0, 30.0],
        "category": ["groceries", "shopping"],
        "source": ["test", "test"],
        "format": ["generic", "generic"],
    })
    summary = monthly_summary(df)
    assert {"year_month", "category", "total"}.issubset(set(summary.columns))


def test_monthly_summary_aggregates_same_category():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-10", "2024-01-20"]),
        "description": ["KROGER", "KROGER 2"],
        "amount": [50.0, 30.0],
        "category": ["groceries", "groceries"],
        "source": ["test", "test"],
        "format": ["generic", "generic"],
    })
    summary = monthly_summary(df)
    assert len(summary) == 1
    assert abs(summary.iloc[0]["total"] - 80.0) < 1e-6


def test_monthly_summary_excludes_negative_amounts():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-10", "2024-01-25"]),
        "description": ["KROGER", "PAYROLL"],
        "amount": [50.0, -3500.0],
        "category": ["groceries", "income"],
        "source": ["test", "test"],
        "format": ["generic", "generic"],
    })
    summary = monthly_summary(df)
    assert (summary["total"] > 0).all()
    assert "income" not in summary["category"].values


def test_monthly_summary_year_month_is_timestamp():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-10"]),
        "description": ["KROGER"],
        "amount": [50.0],
        "category": ["groceries"],
        "source": ["test"],
        "format": ["generic"],
    })
    summary = monthly_summary(df)
    assert pd.api.types.is_datetime64_any_dtype(summary["year_month"])


# ── load_directory ────────────────────────────────────────────────────────────

def test_load_directory_raises_when_no_csvs(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_directory(tmp_path)


def test_load_directory_deduplicates_identical_rows(tmp_path):
    header = "Transaction Date,Post Date,Description,Category,Type,Amount,Memo\n"
    row = "01/15/2024,01/16/2024,KROGER #123,Food & Drink,Sale,-45.67,\n"
    (tmp_path / "file1.csv").write_text(header + row)
    (tmp_path / "file2.csv").write_text(header + row)
    df = load_directory(tmp_path)
    assert len(df) == 1, "Identical rows from two files must be deduplicated"


def test_load_directory_combines_multiple_files(tmp_path):
    header = "Transaction Date,Post Date,Description,Category,Type,Amount,Memo\n"
    row1 = "01/15/2024,01/16/2024,KROGER #123,Food & Drink,Sale,-45.67,\n"
    row2 = "02/10/2024,02/11/2024,STARBUCKS,Dining,Sale,-12.50,\n"
    (tmp_path / "file1.csv").write_text(header + row1)
    (tmp_path / "file2.csv").write_text(header + row2)
    df = load_directory(tmp_path)
    assert len(df) == 2
