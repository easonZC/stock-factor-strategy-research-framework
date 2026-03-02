"""模型因子基准评测入口。

在统一面板上批量训练并比较多模型 OOF 因子表现，输出对比结果与研究报告。
"""

from __future__ import annotations

import argparse
from _bootstrap import ensure_core_path
from _ux import render_run_summary, resolve_output_dir

ROOT = ensure_core_path(__file__)

from factorlab.config import UniverseFilterConfig  # noqa: E402
from factorlab.utils import configure_logging, get_logger  # noqa: E402
from factorlab.workflows import ModelFactorBenchmarkConfig, run_model_factor_benchmark  # noqa: E402

LOGGER = get_logger("factorlab.run_model_factor_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在统一面板上对多模型因子做 OOF 基准评测。",
        epilog=(
            "示例:\n"
            "  python apps/run_model_factor_benchmark.py --panel data/panel.parquet --models lgbm,mlp --out outputs/research/model_factor/benchmark\n"
            "  python apps/run_model_factor_benchmark.py --panel data/panel.parquet --models ridge,rf --label-horizon 10 --out outputs/research/model_factor/benchmark_h10\n"
            "  python apps/run_model_factor_benchmark.py --panel data/panel.parquet --models ridge --scoring-metric mse --evaluation-axis time --split-mode expanding --out outputs/research/model_factor/benchmark_mse\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--panel", required=True, help="面板路径（.parquet/.csv）")
    parser.add_argument(
        "--out",
        default=None,
        help="输出目录；不填则自动生成到 outputs/research/model_factor/<name>_<timestamp>",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="运行名称（仅在未提供 --out 时生效）。",
    )
    parser.add_argument(
        "--models",
        default="lgbm,mlp",
        help="模型名列表，逗号分隔（需被 ModelRegistry 支持）",
    )
    parser.add_argument("--factor-prefix", default="model_factor_oof", help="输出模型因子名前缀")
    parser.add_argument("--save-model-artifacts", action="store_true")
    parser.add_argument("--model-artifact-dir", default="artifacts/models/model_factor_benchmark")
    parser.add_argument(
        "--feature-cols",
        default="momentum_20,volatility_20,liquidity_shock,size",
        help="模型特征列，逗号分隔",
    )
    parser.add_argument(
        "--extra-report-factors",
        default="",
        help="额外纳入最终报告的因子列，逗号分隔",
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
        help="可选目录，目录下按 <model>.json 提供参数网格（list[dict]）",
    )
    parser.add_argument(
        "--model-auto-discover",
        action="store_true",
        help="从 --model-plugin-dir 自动发现模型插件。",
    )
    parser.add_argument(
        "--model-plugin-dir",
        dest="model_plugin_dirs",
        action="append",
        default=[],
        help="模型插件目录（可重复）。",
    )
    parser.add_argument(
        "--model-plugin",
        dest="model_plugins",
        action="append",
        default=[],
        help="模型插件模块或规范（可重复）。",
    )
    parser.add_argument(
        "--model-plugin-on-error",
        choices=["raise", "warn_skip"],
        default="raise",
        help="模型插件冲突/报错处理策略。",
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
    parser.add_argument(
        "--log-level",
        default=None,
        help="日志级别（DEBUG/INFO/WARNING/ERROR），也可用环境变量 FACTORLAB_LOG_LEVEL。",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="日志文件路径（可选）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(level=args.log_level, log_file=args.log_file, force=True)

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
        model_auto_discover=args.model_auto_discover,
        model_plugin_dirs=args.model_plugin_dirs,
        model_plugins=args.model_plugins,
        model_plugin_on_error=args.model_plugin_on_error,
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

    out_dir = resolve_output_dir(
        out=args.out,
        run_name=args.name,
        category="model_factor",
        default_name="benchmark",
    )
    res = run_model_factor_benchmark(
        panel_path=args.panel,
        out_dir=out_dir,
        config=workflow_cfg,
        repo_root=ROOT,
    )
    LOGGER.info(
        "Model-factor benchmark completed. comparison=%s report=%s manifest=%s",
        res.comparison_csv,
        res.index_html,
        res.run_manifest_json,
    )
    LOGGER.info(
        "\n%s",
        render_run_summary(
            title="model_benchmark_completed",
            lines={
                "out_dir": res.out_dir,
                "comparison": res.comparison_csv,
                "report": res.index_html,
                "summary": res.summary_csv,
                "manifest": res.run_manifest_json,
            },
        ),
    )


if __name__ == "__main__":
    main()
