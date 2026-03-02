"""配置驱动主入口。

推荐使用 `configs/cs_factor.yaml` 或 `configs/ts_factor.yaml`，
并通过 `data.path` 指向本地数据文件或目录直接运行。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_core_path

ROOT = ensure_core_path(__file__)

import yaml

from factorlab.ops import OutputRetentionManager, RetentionPolicy  # noqa: E402
from factorlab.utils import configure_logging, get_logger  # noqa: E402
from factorlab.workflows import compose_run_config, run_from_config, validate_run_config_schema  # noqa: E402

LOGGER = get_logger("factorlab.run_from_config")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通过 YAML 配置运行因子研究与可选回测。",
        epilog=(
            "示例:\n"
            "  python apps/run_from_config.py --config configs/cs_factor.yaml --set data.path=data/raw --out outputs/research/factor/cs\n"
            "  python apps/run_from_config.py --config configs/ts_factor.yaml --set data.path=data/raw/000001.csv --out outputs/research/factor/ts\n"
            "  python apps/run_from_config.py --config configs/cs_factor.yaml --set factor.names='[factor_a,factor_b]' --out outputs/research/factor/cs_custom\n"
            "  python apps/run_from_config.py --config configs/cs_factor.yaml --set run.std=cs_rank --set research.q=10 --out outputs/research/factor/cs_alias\n"
            "  python apps/run_from_config.py --config configs/cs_factor.yaml --set run.stop_after=research --out outputs/research/factor/cs_no_backtest\n"
            "  python apps/run_from_config.py --config configs/cs_factor.yaml --out outputs/research/factor/cs --cleanup-old-outputs --cleanup-days 14 --cleanup-keep 30\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="YAML config path (repeatable). Later files override earlier files.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help=(
            "Override via dotted path (repeatable). "
            "Supports '=' replace, '+=' append/merge, '-=' remove. "
            "Examples: research.quantiles=10, research.horizons+=20, research.winsorize-='method'."
        ),
    )
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--save-effective-config",
        action="store_true",
        help="Save merged+overridden effective config to <out>/effective_config.yaml.",
    )
    parser.add_argument(
        "--skip-schema-validation",
        action="store_true",
        help="Skip strict pre-validation of config schema before runtime.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate merged config and exit without running research/backtest pipeline.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level (DEBUG/INFO/WARNING/ERROR). Also supports env FACTORLAB_LOG_LEVEL.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional log file path for this run.",
    )
    parser.add_argument(
        "--cleanup-old-outputs",
        action="store_true",
        help="Run retention cleanup for old output runs after this task.",
    )
    parser.add_argument(
        "--cleanup-root",
        default="outputs/research",
        help="Retention cleanup root directory.",
    )
    parser.add_argument(
        "--cleanup-days",
        type=int,
        default=14,
        help="Remove runs older than this number of days (with keep-latest protection).",
    )
    parser.add_argument(
        "--cleanup-keep",
        type=int,
        default=20,
        help="Always keep latest N runs under cleanup root.",
    )
    parser.add_argument(
        "--cleanup-dry-run",
        action="store_true",
        help="Preview retention cleanup without deleting directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(level=args.log_level, log_file=args.log_file, force=True)
    effective_cfg = compose_run_config(config_paths=args.config, overrides=args.overrides)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_effective_config:
        (out_dir / "effective_config.yaml").write_text(
            yaml.safe_dump(effective_cfg, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
    if args.validate_only:
        schema_warnings = validate_run_config_schema(effective_cfg, strict=True)
        LOGGER.info(
            "Config validation passed. warnings=%s out=%s",
            len(schema_warnings),
            out_dir,
        )
        return

    result = run_from_config(
        config=effective_cfg,
        out_dir=args.out,
        repo_root=ROOT,
        validate_schema=not args.skip_schema_validation,
    )
    LOGGER.info(
        "Config run completed. report=%s summary=%s meta=%s",
        result.index_html,
        result.summary_csv,
        result.run_meta_json,
    )
    if args.cleanup_old_outputs:
        cleanup_res = OutputRetentionManager(
            root_dir=args.cleanup_root,
            policy=RetentionPolicy(
                older_than_days=int(args.cleanup_days),
                keep_latest=int(args.cleanup_keep),
                dry_run=bool(args.cleanup_dry_run),
            ),
        ).cleanup()
        LOGGER.info(
            "Retention cleanup finished. root=%s scanned=%s removed=%s kept=%s dry_run=%s",
            cleanup_res.root_dir,
            cleanup_res.scanned,
            cleanup_res.removed,
            cleanup_res.kept,
            args.cleanup_dry_run,
        )


if __name__ == "__main__":
    main()
