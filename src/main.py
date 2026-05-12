"""CLI entry point for the spending-predictor pipeline."""

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spending-predictor",
        description="Spending analysis and retirement forecasting pipeline.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing CSV exports (default: data/raw)",
    )
    parser.add_argument(
        "--format",
        choices=["chase", "bofa", "amex", "capital_one", "generic"],
        default=None,
        dest="fmt",
        help="Override CSV format auto-detection",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for output reports (default: reports/)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="Ingest CSVs, build features, train models")

    drift_p = sub.add_parser("drift", help="Check for spending drift vs. trained baseline")
    drift_p.add_argument("--threshold", type=float, default=0.30,
                         help="Drift threshold as a fraction (default: 0.30 = 30%%)")

    predict_p = sub.add_parser("predict", help="Load models and forecast ahead")
    predict_p.add_argument(
        "--months", type=int, default=12, help="Months to forecast (default: 12)"
    )

    report_p = sub.add_parser("report", help="Generate HTML report")
    _add_retirement_args(report_p)
    report_p.add_argument(
        "--months", type=int, default=12, help="Months to forecast (default: 12)"
    )

    run_p = sub.add_parser("run", help="Full pipeline: train → predict → report")
    run_p.add_argument(
        "--months", type=int, default=12, help="Months to forecast (default: 12)"
    )
    _add_retirement_args(run_p)

    return parser


def _add_retirement_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--monthly-income", type=float, default=20_703.0)
    p.add_argument("--current-savings", type=float, default=1_301_757.0)
    p.add_argument("--retirement-target", type=float, default=1_400_000.0)
    p.add_argument("--years-to-retirement", type=int, default=3)


def cmd_drift(args) -> list[dict]:
    from src.ingest import load_directory, monthly_summary
    from src.features import build_category_features
    from src.drift import check_drift, print_drift_report

    print(f"[drift] Loading CSVs from {args.data_dir} ...")
    df = load_directory(args.data_dir, fmt=args.fmt)
    monthly_df = monthly_summary(df)
    category_features = build_category_features(monthly_df)

    threshold = getattr(args, "threshold", 0.30)
    drift_report = check_drift(category_features, threshold=threshold)
    print_drift_report(drift_report)
    return drift_report


def cmd_train(args) -> None:
    from src.ingest import load_directory, monthly_summary
    from src.features import build_category_features, build_total_features, build_prophet_df, get_feature_columns
    from src.train import train_category_models, train_prophet_model

    print(f"[train] Loading CSVs from {args.data_dir} ...")
    df = load_directory(args.data_dir, fmt=args.fmt)
    monthly_df = monthly_summary(df)

    print("[train] Building features ...")
    category_features = build_category_features(monthly_df)
    total_df = build_total_features(monthly_df)
    feature_cols = get_feature_columns()
    prophet_df = build_prophet_df(total_df)

    print(f"[train] Training {len(category_features)} category models ...")
    train_category_models(category_features, feature_cols)

    print("[train] Training Prophet model ...")
    train_prophet_model(prophet_df)

    print("[train] Done.")


def cmd_predict(args) -> dict:
    from src.features import build_category_features, get_feature_columns
    from src.train import load_category_models, load_prophet_model
    from src.predict import predict_next_months, prophet_forecast, rank_reduction_opportunities
    from src.drift import check_drift, print_drift_report

    print("[predict] Loading models ...")
    _registry, models = load_category_models()
    prophet_model = load_prophet_model()

    from src.ingest import load_directory, monthly_summary
    df = load_directory(args.data_dir, fmt=args.fmt)
    monthly_df = monthly_summary(df)
    category_features = build_category_features(monthly_df)
    feature_cols = get_feature_columns()

    drift_report = check_drift(category_features)
    print_drift_report(drift_report)

    print(f"[predict] Forecasting {args.months} months ahead ...")
    forecasts = predict_next_months(models, category_features, feature_cols, n_months=args.months)

    prophet_df = prophet_forecast(prophet_model, periods=args.months) if prophet_model is not None else None
    reduction_df = rank_reduction_opportunities(forecasts)

    print("[predict] Done.")
    return {
        "monthly_df": monthly_df,
        "forecasts": forecasts,
        "prophet_df": prophet_df,
        "reduction_df": reduction_df,
        "drift_report": drift_report,
    }


def cmd_report(args, predict_results: dict | None = None) -> None:
    from src.report import retirement_gap_analysis, generate_html_report

    if predict_results is None:
        predict_results = cmd_predict(args)

    months_to_retirement = args.years_to_retirement * 12
    retirement_result = retirement_gap_analysis(
        forecasts=predict_results["forecasts"],
        monthly_income=args.monthly_income,
        current_savings=args.current_savings,
        retirement_target=args.retirement_target,
        months_to_retirement=months_to_retirement,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "report.html"

    import pandas as pd
    prophet_df = predict_results["prophet_df"]
    if prophet_df is None:
        prophet_df = pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])

    report_path = generate_html_report(
        monthly_df=predict_results["monthly_df"],
        forecasts=predict_results["forecasts"],
        prophet_df=prophet_df,
        reduction_df=predict_results["reduction_df"],
        retirement_result=retirement_result,
        output_path=output_path,
        drift_report=predict_results.get("drift_report"),
    )
    print(f"[report] Report written to {report_path}")


def cmd_run(args) -> None:
    from src.ingest import load_directory, monthly_summary
    from src.features import (
        build_category_features,
        build_total_features,
        build_prophet_df,
        get_feature_columns,
    )
    from src.train import train_category_models, train_prophet_model
    from src.predict import predict_next_months, prophet_forecast, rank_reduction_opportunities
    from src.report import retirement_gap_analysis, generate_html_report
    from src.drift import check_drift, print_drift_report
    import pandas as pd

    print(f"[run] Loading CSVs from {args.data_dir} ...")
    df = load_directory(args.data_dir, fmt=args.fmt)
    monthly_df = monthly_summary(df)

    print("[run] Building features ...")
    category_features = build_category_features(monthly_df)
    total_df = build_total_features(monthly_df)
    feature_cols = get_feature_columns()
    prophet_df_input = build_prophet_df(total_df)

    drift_report = check_drift(category_features)
    print_drift_report(drift_report)

    print(f"[run] Training {len(category_features)} category models ...")
    import joblib as _joblib
    registry = train_category_models(category_features, feature_cols)
    models = {cat: _joblib.load(info["path"]) for cat, info in registry.items()}

    print("[run] Training Prophet model ...")
    prophet_model = train_prophet_model(prophet_df_input)

    print(f"[run] Forecasting {args.months} months ahead ...")
    forecasts = predict_next_months(models, category_features, feature_cols, n_months=args.months)

    prophet_forecast_df = prophet_forecast(prophet_model, periods=args.months) if prophet_model is not None else pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
    reduction_df = rank_reduction_opportunities(forecasts)

    months_to_retirement = args.years_to_retirement * 12
    retirement_result = retirement_gap_analysis(
        forecasts=forecasts,
        monthly_income=args.monthly_income,
        current_savings=args.current_savings,
        retirement_target=args.retirement_target,
        months_to_retirement=months_to_retirement,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "report.html"

    report_path = generate_html_report(
        monthly_df=monthly_df,
        forecasts=forecasts,
        prophet_df=prophet_forecast_df,
        reduction_df=reduction_df,
        retirement_result=retirement_result,
        output_path=output_path,
        drift_report=drift_report,
    )
    print(f"[run] Report written to {report_path}")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        cmd_train(args)
    elif args.command == "drift":
        cmd_drift(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
