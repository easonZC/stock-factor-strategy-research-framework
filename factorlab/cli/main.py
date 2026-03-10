"""统一命令行入口。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

import yaml

from factorlab.cli.utils import (
    add_logging_args,
    add_output_args,
    render_run_summary,
    resolve_output_dir,
    setup_logging_from_args,
)
from factorlab.config import AdapterConfig, UniverseFilterConfig
from factorlab.data import build_data_adapter_registry, build_data_adapter_validator_registry, write_panel
from factorlab.factors import build_factor_registry, describe_factor_registry, factor_definitions_frame
from factorlab.models import ModelRegistry, train_model_factor
from factorlab.ops import OutputRetentionManager, RetentionPolicy
from factorlab.strategies import build_strategy_registry, describe_strategy_registry, strategy_definitions_frame
from factorlab.utils import get_logger
from factorlab.workflows import (
    compose_run_config,
    compose_run_config_with_alias_report,
    ModelFactorBenchmarkConfig,
    PanelFactorResearchConfig,
    run_from_config,
    run_model_factor_benchmark,
    run_panel_factor_research,
    validate_run_config_schema,
)

LOGGER = get_logger("factorlab.cli")


def _catalog_payload(items: list[Any], *, strategy: bool) -> list[dict[str, Any]]:
    key = "constraints" if strategy else "required_columns"
    return [
        {
            "name": item.name,
            "family": item.family,
            "origin": item.origin,
            key: list(item.constraints if strategy else item.required_columns),
            "tags": list(item.tags),
            "description": item.description,
            "parameters": item.parameters,
            "implementation": item.implementation,
            **({} if strategy else {"formula": item.formula}),
        }
        for item in items
    ]


def _configure_run_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "通过 YAML 配置运行因子研究与可选回测。"
    parser.epilog = (
        "示例:\n"
        "  factorlab run --config examples/workflows/cs_factor.yaml --set data.path=data/raw\n"
        "  factorlab run --config examples/workflows/ts_factor.yaml --set data.path=data/raw/000001.csv\n"
    )
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.add_argument("--config", action="append", required=True, help="YAML config path (repeatable).")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override via dotted path.")
    add_output_args(parser, category="factor")
    parser.add_argument("--save-effective-config", action="store_true", help="Save effective config.")
    parser.add_argument("--show-effective-config", action="store_true", help="Print effective config.")
    parser.add_argument("--skip-schema-validation", action="store_true", help="Skip strict pre-validation.")
    parser.add_argument("--validate-only", action="store_true", help="Validate merged config and exit.")
    add_logging_args(parser, include_log_file=True)
    parser.add_argument("--cleanup-old-outputs", action="store_true", help="Run retention cleanup after task.")
    parser.add_argument("--cleanup-root", default="outputs/research", help="Retention cleanup root directory.")
    parser.add_argument("--cleanup-days", type=int, default=14, help="Remove runs older than this number of days.")
    parser.add_argument("--cleanup-keep", type=int, default=20, help="Always keep latest N runs.")
    parser.add_argument("--cleanup-dry-run", action="store_true", help="Preview retention cleanup.")


def _run_command(args: argparse.Namespace) -> None:
    setup_logging_from_args(args)
    effective_cfg = compose_run_config(config_paths=args.config, overrides=args.overrides)
    out_dir = resolve_output_dir(
        out=args.out,
        run_name=args.name,
        category="factor",
        default_name=Path(args.config[-1]).stem if args.config else "factor_run",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.show_effective_config:
        print(yaml.safe_dump(effective_cfg, sort_keys=False, allow_unicode=False))
    if args.save_effective_config:
        (out_dir / "effective_config.yaml").write_text(
            yaml.safe_dump(effective_cfg, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
    if args.validate_only:
        warnings = validate_run_config_schema(effective_cfg, strict=True)
        LOGGER.info(
            "\n%s",
            render_run_summary(
                title="validate_only",
                lines={"config_files": ", ".join(args.config), "warnings": len(warnings), "effective_out_dir": out_dir},
            ),
        )
        return
    result = run_from_config(
        config=effective_cfg,
        out_dir=out_dir,
        repo_root=Path.cwd(),
        validate_schema=not args.skip_schema_validation,
    )
    LOGGER.info(
        "\n%s",
        render_run_summary(
            title="run_completed",
            lines={
                "out_dir": result.out_dir,
                "report": result.index_html,
                "summary": result.summary_csv,
                "meta": result.run_meta_json,
                "manifest": result.run_manifest_json,
                "backtest_summary": result.backtest_summary_csv or "N/A",
            },
        ),
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


def run_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    _configure_run_parser(parser)
    _run_command(parser.parse_args(argv))


def _configure_lint_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "运行前检查配置结构与潜在风险。"
    parser.add_argument("--config", action="append", required=True, help="YAML config path (repeatable).")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Dotted-path override.")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero code if schema has errors.")
    parser.add_argument("--json", action="store_true", help="Output JSON report.")
    parser.add_argument("--save-effective-config", default=None, help="Optional path to save effective config.")


def _build_suggestions(
    cfg: dict[str, Any],
    alias_events: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> list[str]:
    suggestions = [
        f"将别名键 `{event['alias']}` 改为标准键 `{event['canonical']}`，减少后续兼容迁移。"
        for event in alias_events
        if event.get("applied")
    ]
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), dict) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    if (
        str(run_cfg.get("research_profile", "full")).strip().lower() == "full"
        and str(data_cfg.get("adapter", "")).strip().lower() == "synthetic"
    ):
        syn = data_cfg.get("synthetic", {}) if isinstance(data_cfg.get("synthetic"), dict) else {}
        if int(syn.get("n_assets", 0) or 0) * int(syn.get("n_days", 0) or 0) >= 40000:
            suggestions.append("当前样本规模较大且 profile=full，开发阶段可先切换为 `run.research_profile=dev`。")
    return suggestions or (["配置体检通过，当前未发现需要调整的项。"] if not errors and not warnings else [])


def _lint_command(args: argparse.Namespace) -> None:
    effective_cfg, alias_events = compose_run_config_with_alias_report(config_paths=args.config, overrides=args.overrides)
    schema_warnings = validate_run_config_schema(effective_cfg, strict=False)
    errors = [item[len("ERROR_AS_WARNING: ") :] for item in schema_warnings if item.startswith("ERROR_AS_WARNING: ")]
    warnings = [item for item in schema_warnings if not item.startswith("ERROR_AS_WARNING: ")]
    suggestions = _build_suggestions(effective_cfg, alias_events, errors, warnings)
    if args.save_effective_config:
        out = Path(args.save_effective_config)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(effective_cfg, sort_keys=False, allow_unicode=False), encoding="utf-8")
    payload = {
        "ok": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "alias_migration_count": len(alias_events),
        "errors": errors,
        "warnings": warnings,
        "alias_migrations": alias_events,
        "suggestions": suggestions,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"[lint] ok={payload['ok']} errors={len(errors)} warnings={len(warnings)} aliases={len(alias_events)}")
        for section, items in [("errors", errors), ("warnings", warnings), ("suggestions", suggestions)]:
            if items:
                print(f"\n[{section}]")
                for item in items:
                    print(f"- {item}")
    if args.strict and errors:
        raise SystemExit(1)


def lint_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    _configure_lint_parser(parser)
    _lint_command(parser.parse_args(argv))


def _configure_panel_research_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "从单个面板文件直接运行截面因子研究。"
    parser.add_argument("--panel", required=True, help="面板文件路径（.parquet/.csv）")
    parser.add_argument("--factors", default="", help="逗号分隔因子名；留空时自动从面板列发现因子。")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 20], help="前瞻收益窗口")
    add_output_args(parser, category="factor")
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
    parser.add_argument("--preprocess-steps", default="winsorize,standardize,neutralize", help="预处理步骤，逗号分隔。")
    parser.add_argument("--quantiles", type=int, default=5, help="分位数组数")
    parser.add_argument("--ic-rolling-window", type=int, default=20, help="IC 滚动窗口")
    parser.add_argument("--on-missing-factor", choices=["raise", "warn_skip"], default="warn_skip")
    add_logging_args(parser)


def _panel_research_command(args: argparse.Namespace) -> None:
    setup_logging_from_args(args)
    result = run_panel_factor_research(
        panel_path=args.panel,
        out_dir=resolve_output_dir(out=args.out, run_name=args.name, category="factor", default_name="panel_research"),
        config=PanelFactorResearchConfig(
            factors=args.factors,
            horizons=list(args.horizons),
            neutralize=args.neutralize,
            winsorize=args.winsorize,
            standardization=args.standardization,
            missing_policy=args.missing_policy,
            preprocess_steps=args.preprocess_steps,
            quantiles=int(args.quantiles),
            ic_rolling_window=int(args.ic_rolling_window),
            on_missing_factor=args.on_missing_factor,
        ),
        repo_root=Path.cwd(),
        validate_schema=True,
    )
    LOGGER.info(
        "\n%s",
        render_run_summary(
            title="panel_research_completed",
            lines={
                "out_dir": result.out_dir,
                "report": result.index_html,
                "summary": result.summary_csv,
                "meta": result.run_meta_json,
            },
        ),
    )


def panel_research_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    _configure_panel_research_parser(parser)
    _panel_research_command(parser.parse_args(argv))


def _configure_model_benchmark_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "在统一面板上对多模型因子做 OOF 基准评测。"
    parser.add_argument("--panel", required=True, help="面板路径（.parquet/.csv）")
    add_output_args(parser, category="model_factor")
    parser.add_argument("--models", default="lgbm,mlp", help="模型名列表，逗号分隔（需被 ModelRegistry 支持）")
    parser.add_argument("--factor-prefix", default="model_factor_oof", help="输出模型因子名前缀")
    parser.add_argument("--save-model-artifacts", action="store_true")
    parser.add_argument("--model-artifact-dir", default="artifacts/models/model_factor_benchmark")
    parser.add_argument("--feature-cols", default="momentum_20,volatility_20,liquidity_shock,size")
    parser.add_argument("--extra-report-factors", default="")
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
    parser.add_argument("--model-param-grid-dir", default=None)
    parser.add_argument("--model-auto-discover", action="store_true")
    parser.add_argument("--model-plugin-dir", dest="model_plugin_dirs", action="append", default=[])
    parser.add_argument("--model-plugin", dest="model_plugins", action="append", default=[])
    parser.add_argument("--model-plugin-on-error", choices=["raise", "warn_skip"], default="raise")
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
    parser.add_argument("--min-close", type=float, default=None)
    parser.add_argument("--min-history-days", type=int, default=120)
    parser.add_argument("--min-median-dollar-volume", type=float, default=None)
    parser.add_argument("--liquidity-lookback", type=int, default=20)
    add_logging_args(parser)


def _model_benchmark_command(args: argparse.Namespace) -> None:
    setup_logging_from_args(args)
    res = run_model_factor_benchmark(
        panel_path=args.panel,
        out_dir=resolve_output_dir(out=args.out, run_name=args.name, category="model_factor", default_name="benchmark"),
        config=ModelFactorBenchmarkConfig(
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
        ),
        repo_root=Path.cwd(),
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


def model_benchmark_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    _configure_model_benchmark_parser(parser)
    _model_benchmark_command(parser.parse_args(argv))


def _configure_prepare_data_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "使用内置或插件适配器拉取并落盘面板数据。"
    parser.add_argument("--adapter", required=True, help="适配器名称（内置或插件）")
    parser.add_argument("--data-dir", default=None, help="输入目录（sina 适配器必填）")
    parser.add_argument("--symbols", default=None, help="股票列表，逗号分隔（stooq 适配器必填）")
    parser.add_argument("--start-date", default=None, help="起始日期（stooq，可选，YYYY-MM-DD）")
    parser.add_argument("--end-date", default=None, help="结束日期（stooq，可选，YYYY-MM-DD）")
    parser.add_argument("--min-rows-per-asset", type=int, default=30)
    parser.add_argument("--request-timeout-sec", type=int, default=20)
    parser.add_argument("--adapter-plugin-dir", dest="adapter_plugin_dirs", action="append", default=[])
    parser.add_argument("--adapter-plugin", dest="adapter_plugins", action="append", default=[])
    parser.add_argument("--adapter-plugin-on-error", choices=["raise", "warn_skip"], default="raise")
    add_logging_args(parser, include_log_file=True)
    parser.add_argument("--out", required=True, help="输出面板路径（.parquet/.csv）")


def _prepare_data_command(args: argparse.Namespace) -> None:
    setup_logging_from_args(args)
    registry = build_data_adapter_registry(
        plugin_dirs=args.adapter_plugin_dirs,
        plugin_specs=args.adapter_plugins,
        on_plugin_error=args.adapter_plugin_on_error,
        include_defaults=True,
    )
    validator_registry = build_data_adapter_validator_registry(
        plugin_dirs=args.adapter_plugin_dirs,
        plugin_specs=args.adapter_plugins,
        on_plugin_error=args.adapter_plugin_on_error,
        include_defaults=True,
    )
    adapter_name = str(args.adapter).strip().lower()
    if adapter_name not in registry:
        raise KeyError(f"Unknown adapter '{adapter_name}'. Available adapters: {sorted(registry.keys())}")
    symbols = tuple(x.strip() for x in str(args.symbols).split(",") if x.strip()) if args.symbols else ()
    if adapter_name == "sina" and not args.data_dir:
        raise ValueError("--data-dir is required when --adapter=sina")
    if adapter_name == "stooq" and not symbols:
        raise ValueError("--symbols is required when --adapter=stooq")
    adapter_cfg = AdapterConfig(
        data_dir=str(args.data_dir or ""),
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        min_rows_per_asset=int(args.min_rows_per_asset),
        request_timeout_sec=int(args.request_timeout_sec),
    )
    if validator := validator_registry.get(adapter_name):
        validator(adapter_cfg)
    panel = registry[adapter_name](adapter_cfg)
    out = write_panel(panel, args.out)
    LOGGER.info("Prepared panel saved to %s (rows=%s assets=%s)", out, len(panel), panel["asset"].nunique())


def prepare_data_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    _configure_prepare_data_parser(parser)
    _prepare_data_command(parser.parse_args(argv))


def _configure_cleanup_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "按策略清理 outputs 历史运行目录。"
    parser.add_argument("--root", default="outputs/research", help="待清理根目录")
    parser.add_argument("--older-than-days", type=int, default=14, help="仅删除早于该天数的运行目录")
    parser.add_argument("--keep-latest", type=int, default=20, help="至少保留最新 N 个运行目录")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不实际删除")
    parser.add_argument("--json", action="store_true", help="输出 JSON 结果")
    parser.add_argument("--purge-all", action="store_true", help="直接清空 root 下所有内容。")
    add_logging_args(parser)


def _purge_all(root: str) -> dict[str, object]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        return {"root_dir": str(root_path), "scanned": 0, "removed": 0, "kept": 0, "dry_run": False, "removed_paths": []}
    if str(root_path) in {"/", str(Path.home().resolve())}:
        raise ValueError(f"Refuse to purge unsafe root: {root_path}")
    removed_paths: list[str] = []
    for path in list(root_path.iterdir()):
        removed_paths.append(str(path))
        shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)
    root_path.mkdir(parents=True, exist_ok=True)
    return {
        "root_dir": str(root_path),
        "scanned": len(removed_paths),
        "removed": len(removed_paths),
        "kept": 0,
        "dry_run": False,
        "removed_paths": removed_paths,
    }


def _cleanup_command(args: argparse.Namespace) -> None:
    setup_logging_from_args(args)
    payload = (
        _purge_all(args.root)
        if args.purge_all
        else {
            "root_dir": (res := OutputRetentionManager(
                root_dir=args.root,
                policy=RetentionPolicy(
                    older_than_days=int(args.older_than_days),
                    keep_latest=int(args.keep_latest),
                    dry_run=bool(args.dry_run),
                ),
            ).cleanup()).root_dir,
            "scanned": res.scanned,
            "removed": res.removed,
            "kept": res.kept,
            "dry_run": bool(args.dry_run),
            "removed_paths": res.removed_paths,
        }
    )
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    LOGGER.info(
        "cleanup completed: root=%s scanned=%s removed=%s kept=%s dry_run=%s",
        payload["root_dir"],
        payload["scanned"],
        payload["removed"],
        payload["kept"],
        payload["dry_run"],
    )


def cleanup_outputs_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    _configure_cleanup_parser(parser)
    _cleanup_command(parser.parse_args(argv))


def _configure_train_model_factor_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = "用合成面板训练并保存模型因子（支持含 MLP 在内的注册模型）。"
    parser.add_argument(
        "--model",
        default="ridge",
        help=f"模型名；内置模型: {', '.join(ModelRegistry.available_models())}",
    )
    parser.add_argument("--out", default="artifacts/models/model_factor.joblib")


def _train_model_factor_command(args: argparse.Namespace) -> None:
    print(train_model_factor(model_name=args.model, model_out=args.out))


def train_model_factor_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    _configure_train_model_factor_parser(parser)
    _train_model_factor_command(parser.parse_args(argv))


def _configure_catalog_parser(parser: argparse.ArgumentParser, *, strategy: bool) -> None:
    noun = "策略" if strategy else "因子"
    parser.description = f"列出当前可用{noun}及其定义信息。"
    parser.add_argument("--plugin-dir", action="append", default=[], help=f"额外{noun}插件目录（可重复）。")
    parser.add_argument("--plugin", action="append", default=[], help=f"额外{noun}插件规范（可重复）。")
    parser.add_argument("--json", action="store_true", help="输出 JSON。")
    parser.add_argument("--name", action="append", default=[], help=f"仅显示指定{noun}名（可重复）。")
    add_logging_args(parser)


def _factor_catalog_command(args: argparse.Namespace) -> None:
    setup_logging_from_args(args)
    definitions = describe_factor_registry(
        build_factor_registry(plugin_dirs=args.plugin_dir, plugin_specs=args.plugin, on_plugin_error="raise"),
        names=args.name or None,
    )
    if args.json:
        print(json.dumps(_catalog_payload(definitions, strategy=False), ensure_ascii=False, indent=2))
        return
    frame = factor_definitions_frame(definitions)
    print("No factors found." if frame.empty else frame[["name", "family", "required_columns", "tags", "description"]].to_string(index=False))


def list_factors_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    _configure_catalog_parser(parser, strategy=False)
    _factor_catalog_command(parser.parse_args(argv))


def _strategy_catalog_command(args: argparse.Namespace) -> None:
    setup_logging_from_args(args)
    definitions = describe_strategy_registry(
        build_strategy_registry(
            plugin_dirs=args.plugin_dir,
            plugin_specs=args.plugin,
            on_plugin_error="raise",
            include_defaults=True,
        ),
        names=args.name or None,
    )
    if args.json:
        print(json.dumps(_catalog_payload(definitions, strategy=True), ensure_ascii=False, indent=2))
        return
    frame = strategy_definitions_frame(definitions)
    print("No strategies found." if frame.empty else frame[["name", "family", "constraints", "tags", "description"]].to_string(index=False))


def list_strategies_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    _configure_catalog_parser(parser, strategy=True)
    _strategy_catalog_command(parser.parse_args(argv))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="factorlab",
        description="FactorLab 统一入口：研究、体检、因子/策略目录。",
        epilog=(
            "Recommended:\n"
            "  factorlab run --config examples/workflows/cs_factor.yaml --set data.path=data/raw\n"
            "Fallback without installed script:\n"
            "  python -m factorlab run --config examples/workflows/cs_factor.yaml --set data.path=data/raw"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subs = parser.add_subparsers(dest="command", required=True)
    for name, configure, handler in [
        ("run", _configure_run_parser, _run_command),
        ("run-panel-research", _configure_panel_research_parser, _panel_research_command),
        ("run-model-benchmark", _configure_model_benchmark_parser, _model_benchmark_command),
        ("prepare-data", _configure_prepare_data_parser, _prepare_data_command),
        ("cleanup-outputs", _configure_cleanup_parser, _cleanup_command),
        ("train-model-factor", _configure_train_model_factor_parser, _train_model_factor_command),
        ("lint-config", _configure_lint_parser, _lint_command),
        ("list-factors", lambda p: _configure_catalog_parser(p, strategy=False), _factor_catalog_command),
        ("list-strategies", lambda p: _configure_catalog_parser(p, strategy=True), _strategy_catalog_command),
    ]:
        sub = subs.add_parser(name)
        configure(sub)
        sub.set_defaults(handler=handler)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
