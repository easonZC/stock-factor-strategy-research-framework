"""面板文件快速研究入口。

该脚本是 `run_from_config` 的薄封装：
1. 接收常用参数（面板路径、因子、研究参数）
2. 组装标准配置字典
3. 复用统一工作流执行研究
"""

from __future__ import annotations

import argparse
from _bootstrap import ensure_core_path

ROOT = ensure_core_path(__file__)

from factorlab.utils import configure_logging, get_logger  # noqa: E402
from factorlab.workflows import run_from_config  # noqa: E402

LOGGER = get_logger("factorlab.run_factor_research")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从单个面板文件直接运行截面因子研究。",
        epilog=(
            "示例:\n"
            "  python apps/run_factor_research.py --panel data/panel.parquet --factors factor_a,factor_b --out outputs/research/factor/panel\n"
            "  python apps/run_factor_research.py --panel data/panel.parquet --horizons 1 5 10 20 --preprocess-steps winsorize,standardize --out outputs/research/factor/panel_fast\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--panel", required=True, help="面板文件路径（.parquet/.csv）")
    parser.add_argument(
        "--factors",
        default="",
        help="逗号分隔因子名；留空时自动从面板列发现因子。",
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 20], help="前瞻收益窗口")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--neutralize", choices=["none", "size", "industry", "both"], default="both")
    parser.add_argument("--winsorize", choices=["quantile", "mad"], default="quantile")
    parser.add_argument(
        "--standardization",
        choices=["cs_zscore", "cs_rank", "cs_robust_zscore", "none"],
        default="cs_zscore",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["drop", "fill_zero", "ffill_by_asset", "cs_median_by_date", "keep"],
        default="drop",
    )
    parser.add_argument(
        "--preprocess-steps",
        default="winsorize,standardize,neutralize",
        help="预处理步骤，逗号分隔。",
    )
    parser.add_argument("--quantiles", type=int, default=5, help="分位数组数")
    parser.add_argument("--ic-rolling-window", type=int, default=20, help="IC 滚动窗口")
    parser.add_argument(
        "--on-missing-factor",
        choices=["raise", "warn_skip"],
        default="warn_skip",
        help="因子缺失时行为。",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="日志级别（DEBUG/INFO/WARNING/ERROR）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(level=args.log_level, force=True)
    factor_names = [x.strip() for x in str(args.factors).split(",") if x.strip()]
    preprocess_steps = [x.strip().lower() for x in str(args.preprocess_steps).split(",") if x.strip()]

    cfg: dict = {
        "run": {
            "factor_scope": "cs",
            "eval_axis": "cross_section",
            "standardization": args.standardization,
            "stop_after": "research",
            "research_profile": "full",
        },
        "data": {
            "path": args.panel,
            "mode": "panel",
            "fields_required": ["date", "asset", "close"],
        },
        "factor": {
            "on_missing": args.on_missing_factor,
        },
        "research": {
            "horizons": list(args.horizons),
            "quantiles": int(args.quantiles),
            "ic_rolling_window": int(args.ic_rolling_window),
            "missing_policy": args.missing_policy,
            "preprocess_steps": preprocess_steps,
            "winsorize": {
                "enabled": True,
                "method": args.winsorize,
            },
            "neutralize": {
                "enabled": args.neutralize != "none",
                "mode": args.neutralize,
            },
        },
        "backtest": {
            "enabled": False,
        },
    }
    if factor_names:
        cfg["factor"]["names"] = factor_names

    result = run_from_config(config=cfg, out_dir=args.out, repo_root=ROOT, validate_schema=True)
    LOGGER.info(
        "factor research completed: report=%s summary=%s meta=%s",
        result.index_html,
        result.summary_csv,
        result.run_meta_json,
    )


if __name__ == "__main__":
    main()
