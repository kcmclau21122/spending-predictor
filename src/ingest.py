"""
CSV ingestion module — normalizes bank export formats into a unified schema.

Supported formats: Chase, Bank of America, American Express, Capital One, generic.
Output schema: date (datetime), description (str), amount (float), category (str), source (str)
Positive amount = expense, negative amount = income/credit.
"""

import re
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Category keyword mapping ──────────────────────────────────────────────────

CATEGORY_RULES = [
    ("groceries",    r"kroger|walmart|safeway|publix|whole foods|trader joe|aldi|costco|sam'?s club|meijer|wegmans|sprouts|fresh market|food lion|giant|stop.?&.?shop|heb|winn.?dixie|market basket"),
    ("dining",       r"restaurant|mcdonald|burger king|wendy'?s|taco bell|chipotle|subway|starbucks|dunkin|doordash|grubhub|ubereats|instacart|pizza|sushi|grill|diner|cafe|kitchen|bistro|eatery|panda express|chick.?fil"),
    ("gas",          r"chevron|shell|bp |exxon|mobil|marathon|sunoco|circle k|speedway|pilot|flying j|wawa|sheetz|quiktrip|qt |loves travel|gas station"),
    ("utilities",    r"electric|water bill|natural gas|sewage|trash|waste management|pg&e|con.?ed|duke energy|dominion|xcel energy|centerpoint|nv energy|eversource|national grid"),
    ("internet_phone", r"comcast|xfinity|att|at&t|verizon|t.?mobile|sprint|spectrum|cox communications|optimum|frontier|centurylink|dish network|directv|hulu|sling"),
    ("streaming",    r"netflix|spotify|disney\+|disney plus|amazon prime|apple tv|hbo max|peacock|paramount\+|paramount plus|youtube premium|pandora|tidal|audible"),
    ("healthcare",   r"cvs|walgreens|rite aid|pharmacy|hospital|clinic|doctor|dental|vision|optometrist|urgent care|lab|radiology|medical|health|insurance premium|blue cross|united health|aetna|cigna|humana"),
    ("insurance",    r"geico|state farm|allstate|progressive|farmers|liberty mutual|nationwide|usaa|traveler|insurance"),
    ("shopping",     r"amazon|target|best buy|home depot|lowe'?s|macy'?s|nordstrom|tj maxx|marshalls|ross|old navy|gap|h&m|zara|ikea|wayfair|ebay|etsy|walmart(?!.*grocery)"),
    ("travel",       r"airbnb|vrbo|marriott|hilton|hyatt|expedia|booking\.com|delta|united airlines|american airlines|southwest|spirit|frontier airlines|amtrak|rental car|hertz|enterprise|avis|budget rent"),
    ("transportation", r"uber(?!eats)|lyft|taxi|parking|metro|transit|mta|bart|wmata|septa|mbta|cta |rtd |toll|e-zpass|fastrak"),
    ("entertainment",r"cinema|amc |regal |movie|ticketmaster|eventbrite|concert|theater|museum|zoo|aquarium|sporting|stadium|bowling|arcade|golf|gym|fitness|planet fitness|anytime fitness|ymca|crossfit"),
    ("education",    r"tuition|student loan|udemy|coursera|linkedin learning|skillshare|chegg|textbook|school|university|college"),
    ("subscriptions",r"subscription|annual fee|membership|adobe|microsoft|dropbox|icloud|google one|lastpass|1password"),
    ("home",         r"mortgage|rent |hoa|homeowner|property tax|pest control|lawn|landscaping|cleaning service|home repair|contractor|plumber|electrician|hvac"),
    ("personal_care",r"haircut|salon|barber|nail|spa|massage|skincare|sephora|ulta|beauty"),
    ("pets",         r"petco|petsmart|vet |veterinar|pet supply|pet food|pet insurance"),
    ("children",     r"daycare|childcare|babysit|school supply|kids|children|toy|lego|babies r us|carter'?s"),
    ("charity",      r"donate|donation|charity|nonprofit|red cross|goodwill|salvation army|united way"),
    ("atm_cash",     r"atm |cash withdrawal|cash advance"),
    ("fees_interest",r"late fee|overdraft|annual fee|interest charge|finance charge|service fee"),
    ("income",       r"payroll|direct deposit|salary|wages|transfer from|ach credit|tax refund|dividend|interest income|deposit"),
]

_COMPILED_RULES = [(cat, re.compile(pat, re.IGNORECASE)) for cat, pat in CATEGORY_RULES]

# ── Format detectors ─────────────────────────────────────────────────────────

def _detect_format(df: pd.DataFrame) -> str:
    cols = {c.strip().lower() for c in df.columns}
    if "transaction date" in cols and "post date" in cols and "category" in cols:
        return "chase"
    if "posted date" in cols and "reference number" in cols:
        return "bofa"
    if "card member" in cols or ("date" in cols and "appears on your statement as" in cols):
        return "amex"
    if "transaction type" in cols and "category" in cols and "memo" in cols:
        return "capital_one"
    return "generic"


def _parse_chase(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
    out["description"] = df["Description"].fillna("").str.strip()
    out["amount"] = pd.to_numeric(df["Amount"], errors="coerce") * -1  # Chase negates expenses
    out["bank_category"] = df.get("Category", "").fillna("")
    return out


def _parse_bofa(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["Posted Date"], errors="coerce")
    out["description"] = df["Payee"].fillna("").str.strip()
    # BofA has separate Debits/Credits columns
    debits = pd.to_numeric(df.get("Debits", pd.Series(dtype=float)), errors="coerce").fillna(0)
    credits = pd.to_numeric(df.get("Credits", pd.Series(dtype=float)), errors="coerce").fillna(0)
    out["amount"] = debits - credits
    out["bank_category"] = ""
    return out


def _parse_amex(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    out = pd.DataFrame()
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    desc_col = next((c for c in df.columns if "description" in c.lower() or "appears on" in c.lower()), None)
    out["description"] = df[desc_col].fillna("").str.strip() if desc_col else ""
    amt_col = next((c for c in df.columns if "amount" in c.lower()), None)
    # Amex: positive = charge (expense), negative = credit/payment
    out["amount"] = pd.to_numeric(df[amt_col], errors="coerce") if amt_col else 0.0
    out["bank_category"] = df.get("Category", pd.Series(dtype=str)).fillna("")
    return out


def _parse_capital_one(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
    out["description"] = df.get("Description", df.get("Merchant Name", "")).fillna("").str.strip()
    debit = pd.to_numeric(df.get("Debit", pd.Series(dtype=float)), errors="coerce").fillna(0)
    credit = pd.to_numeric(df.get("Credit", pd.Series(dtype=float)), errors="coerce").fillna(0)
    out["amount"] = debit - credit
    out["bank_category"] = df.get("Category", "").fillna("")
    return out


def _parse_generic(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    # Try to auto-detect date, description, and amount columns
    date_col = next((c for c in df.columns if re.search(r"date|time", c, re.I)), None)
    desc_col = next((c for c in df.columns if re.search(r"desc|payee|merchant|memo|name", c, re.I)), None)
    amt_col = next((c for c in df.columns if re.search(r"amount|debit|charge|price", c, re.I)), None)

    if not all([date_col, desc_col, amt_col]):
        raise ValueError(
            f"Cannot auto-detect columns. Found: {list(df.columns)}. "
            "Rename to 'Date', 'Description', 'Amount' or use --format flag."
        )

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["description"] = df[desc_col].fillna("").str.strip()
    out["amount"] = pd.to_numeric(df[amt_col], errors="coerce")
    out["bank_category"] = ""
    return out


_PARSERS = {
    "chase": _parse_chase,
    "bofa": _parse_bofa,
    "amex": _parse_amex,
    "capital_one": _parse_capital_one,
    "generic": _parse_generic,
}

# ── Category assignment ───────────────────────────────────────────────────────

def _assign_category(description: str, bank_category: str) -> str:
    text = f"{description} {bank_category}"
    for cat, pattern in _COMPILED_RULES:
        if pattern.search(text):
            return cat
    return "other"


# ── Public API ────────────────────────────────────────────────────────────────

def load_csv(path: str | Path, fmt: str | None = None) -> pd.DataFrame:
    """Load a single bank CSV and return a normalized DataFrame."""
    path = Path(path)
    raw = pd.read_csv(path, encoding="utf-8-sig", thousands=",", skipinitialspace=True)
    if raw.empty:
        logger.warning("Empty file: %s", path)
        return pd.DataFrame()

    fmt = fmt or _detect_format(raw)
    logger.info("Detected format '%s' for %s", fmt, path.name)

    parser = _PARSERS.get(fmt, _parse_generic)
    df = parser(raw)
    df["source"] = path.stem
    df["format"] = fmt
    df["category"] = df.apply(
        lambda r: _assign_category(r["description"], r.get("bank_category", "")), axis=1
    )
    df = df.dropna(subset=["date", "amount"])
    df = df[df["amount"] != 0]
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "description", "amount", "category", "source", "format"]]


def load_directory(data_dir: str | Path, fmt: str | None = None) -> pd.DataFrame:
    """Load all CSVs in a directory and concatenate."""
    data_dir = Path(data_dir)
    files = list(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for f in files:
        try:
            frames.append(load_csv(f, fmt=fmt))
        except Exception as e:
            logger.error("Failed to load %s: %s", f.name, e)

    if not frames:
        raise RuntimeError("No CSV files could be loaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "description", "amount"])
    return combined.sort_values("date").reset_index(drop=True)


def split_income_expenses(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate income transactions from expenses."""
    income_mask = (df["category"] == "income") | (df["amount"] < 0)
    return df[income_mask].copy(), df[~income_mask].copy()


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate expenses to monthly totals per category."""
    expenses = df[df["amount"] > 0].copy()
    expenses["year_month"] = expenses["date"].dt.to_period("M")
    summary = (
        expenses.groupby(["year_month", "category"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "total"})
    )
    summary["year_month"] = summary["year_month"].dt.to_timestamp()
    return summary.sort_values(["year_month", "category"])
