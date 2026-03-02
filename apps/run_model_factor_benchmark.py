"""CLI entrypoint for model-factor benchmark workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "core"
if str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))

from factorlab.config import UniverseFilterConfig  # noqa: E402
from factorlab.utils import get_logger  # noqa: E402
from factorlab.workflows import ModelFactorBenchmarkConfig, run_model_factor_benchmark  # noqa: E402

LOGGER = get_logger("factorlab.run_model_factor_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark multiple model factors on one panel.",
        epilog=(
            "Examples:\n"
            "  python apps/run_model_factor_benchmark.py --panel data/panel.parquet --models lgbm,mlp --out outputs/research/model_factor/benchmark\n"
            "  python apps/run_model_factor_benchmark.py --panel data/panel.parquet --models ridge,rf --label-horizon 10 --out outputs/research/model_factor/benchmark_h10\n"
            "  python apps/run_model_factor_benchmark.py --panel data/panel.parquet --models ridge --scoring-metric mse --evaluation-axis time --split-mode expanding --out outputs/research/model_factor/benchmark_mse\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--panel", required=True, help="Panel path (.parquet/.csv)")
    parser.add_argument(
        "--out",
        default="outputs/research/model_factor/benchmark",
        help="Output directory",
    )
    parser.add_argument(
        "--models",
        default="lgbm,mlp",
        help="Comma-separated model names (supported by ModelRegistry)",
    )
    parser.add_argument("--factor-prefix", default="model_factor_oof", help="Output model-factor prefix")
    parser.add_argument("--save-model-artifacts", action="store_true")
    parser.add_argument("--model-artifact-dir", default="artifacts/models/model_factor_benchmark")
    parser.add_argument(
        "--feature-cols",
        default="momentum_20,volatility_20,liquidity_shock,size",
        help="Comma-separated features for model factors",
    )
    parser.add_argument(
        "--extra-report-factors",
        default="",
        help="Additional factor columns to include in final report",
    )
    parser.add_argument("--label-horizon", type=int, default=5)

    parser.add_argument("--train-days", type=int, default=252)
    parser.add_argument("--valid-days", type=int, default=21)
    parser.add_argument("--step-days", type=int, default=21)
    parser.add_argument("--embargo-days", type=int, default=None)
    parser.add_argument("--purge-days", type=int, default=0)
    parser.add_argument("--split-mode", choices=["rolling", "expanding"], default="rolling")
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-valid-rows", type=int, default=100)
    parser.add_argument("--scoring-metric", choices=["rank_ic", "mse"], default="rank_ic")
    parser.add_argument("--evaluation-axis", choices=["cross_section", "time"], default="cross_section")
    parser.add_argument(
        "--model-param-grid-dir",
        default=None,
        help="Optional dir containing <model>.json parameter grids (list[dict])",
    )

    parser.add_argument("--horizons", nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--neutralize", default="both")
    parser.add_argument("--winsorize", default="quantile")
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--ic-rolling-window", type=int, default=20)
    parser.add_argument("--preferred-metric-variant", default="neutralized")

    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--warmup-days", type=int, default=0)
    parser.add_argument("--max-assets", type=int, default=None)
    parser.add_argument("--no-sanitize", action="store_true")
    parser.add_argument("--duplicate-policy", default="last")

    parser.add_argument("--apply-universe-filter", action="store_true")
    parser.add_argument("--min-close", type=float, default=0.0)
    parser.add_argument("--min-history-days", type=int, default=1)
    parser.add_argument("--min-median-dollar-volume", type=float, default=0.0)
    parser.add_argument("--liquidity-lookback", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    workflow_cfg = ModelFactorBenchmarkConfig(
        models=args.models,
        factor_prefix=args.factor_prefix,
        feature_cols=args.feature_cols,
        extra_report_factors=args.extra_report_factors,
        label_horizon=args.label_horizon,
        train_days=args.train_days,
        valid_days=args.valid_days,
        step_days=args.step_days,
        embargo_days=args.embargo_days,
        purge_days=args.purge_days,
        split_mode=args.split_mode,
        min_train_rows=args.min_train_rows,
        min_valid_rows=args.min_valid_rows,
        scoring_metric=args.scoring_metric,
        evaluation_axis=args.evaluation_axis,
        model_param_grid_dir=args.model_param_grid_dir,
        horizons=args.horizons,
        neutralize=args.neutralize,
        winsorize=args.winsorize,
        quantiles=args.quantiles,
        ic_rolling_window=args.ic_rolling_window,
        preferred_metric_variant=args.preferred_metric_variant,
        start_date=args.start_date,
        end_date=args.end_date,
        warmup_days=args.warmup_days,
        max_assets=args.max_assets,
        sanitize=not args.no_sanitize,
        duplicate_policy=args.duplicate_policy,
        apply_universe_filter=args.apply_universe_filter,
        universe_filter=UniverseFilterConfig(
            min_close=args.min_close,
            min_history_days=args.min_history_days,
            min_median_dollar_volume=args.min_median_dollar_volume,
            liquidity_lookback=args.liquidity_lookback,
        ),
        save_model_artifacts=args.save_model_artifacts,
        model_artifact_dir=args.model_artifact_dir,
    )

    res = run_model_factor_benchmark(
        panel_path=args.panel,
        out_dir=args.out,
        config=workflow_cfg,
        repo_root=ROOT,
    )
    LOGGER.info(
        "Model-factor benchmark completed. comparison=%s report=%s manifest=%s",
        res.comparison_csv,
        res.index_html,
        res.run_manifest_json,
    )


if __name__ == "__main__":
    main()
