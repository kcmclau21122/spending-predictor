# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-05-12

### Added
- `src/drift.py` — drift detection module with `save_drift_baseline`, `check_drift`, and `print_drift_report`
- Drift baseline is saved automatically after every `train` run (per-category mean, std, timestamp)
- `drift` CLI subcommand — standalone check that reports which categories have drifted >30% from baseline
- Drift check runs automatically during `predict` and `run` and prints console warnings for flagged categories
- Drift status table added to HTML report — shows trained avg, recent avg, drift %, and retrain recommendation per category

---

## [1.0.1] - 2026-05-09

### Added
- Phase 2: `tests/test_features.py` — 19 unit tests covering `_add_time_features`, `_add_lag_features`, `_add_rolling_features`, `build_category_features` (zero-fill and min-samples exclusion), `get_feature_columns`, and `build_prophet_df`
- Full regression pass: all 49 tests green (30 Phase 1 + 19 Phase 2)

---

## [1.0.0] - 2026-05-09

### Added
- Initial release
- CSV ingestion for Chase, Bank of America, American Express, Capital One, and generic bank formats
- Automatic transaction categorization (groceries, dining, utilities, entertainment, healthcare, etc.)
- Feature engineering: lag features, rolling averages, seasonality encoding
- LightGBM model for category-level monthly spending forecasting
- Prophet model for overall spending trend with seasonality decomposition
- Spending reduction analysis: identifies top categories with reduction potential
- Retirement gap analysis: projects savings trajectory vs. retirement target
- HTML report generation with charts
- GPU-accelerated training via CUDA (RTX 4080)
- CLI entry point for full pipeline execution
