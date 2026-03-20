"""Config-driven workflow orchestration entrypoint."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from factorlab.data import apply_universe_filter
from factorlab.factors import (
    apply_factor_combinations,
    apply_factor_expressions,
    build_factor_registry,
    extract_expression_dependencies,
    write_factor_definition_artifacts,
)
from factorlab.ops.lineage import build_experiment_registry, describe_panel_lineage, stable_hash, write_json_artifact
from factorlab.preprocess import build_transform_registry
from factorlab.runtime import RunContext, coerce_run_context
from factorlab.strategies import StrategyDefinition, build_strategy_registry, write_strategy_definition_artifacts
from factorlab.utils import get_logger, summarize_captured_warnings, timed_stage
from factorlab.workflows.backtest_stage import resolve_strategy_definition, run_optional_backtest
from factorlab.workflows.config_compose import (
    apply_config_override,
    compose_run_config,
    compose_run_config_with_alias_report,
    deep_merge_dict,
    load_run_config,
    validate_run_config_schema,
)
from factorlab.workflows.config_normalization import (
    get_nested_value as _get_nested_value,
    is_forbidden_leakage_name as _is_forbidden_leakage_name,
    is_key_present as _is_key_present,
    normalize_backtest_cfg as _normalize_backtest_cfg,
    normalize_data_cfg as _normalize_data_cfg,
    normalize_factor_cfg as _normalize_factor_cfg,
    normalize_factor_scope as _normalize_factor_scope,
    normalize_research_cfg as _normalize_research_cfg,
    normalize_run_config_aliases,
    normalize_run_governance_cfg as _normalize_run_governance_cfg,
    normalize_universe_cfg as _normalize_universe_cfg,
)
from factorlab.workflows.data_stage import DataAdapterWorkflow
from factorlab.workflows.factor_stage import (
    apply_custom_transforms,
    build_effective_factor_definitions,
    compute_factors,
    filter_factors_by_available_columns,
    preflight_custom_transforms,
    preflight_factor_candidates,
    resolve_required_fields,
    validate_required_fields,
)
from factorlab.workflows.reporting_stage import write_stage_only_outputs
from factorlab.workflows.research_stage import run_research_stage


LOGGER = get_logger("factorlab.workflows.config_runner")


@dataclass(slots=True)
class ConfigRunResult:
    out_dir: Path
    index_html: Path
    summary_csv: Path
    run_meta_json: Path
    run_manifest_json: Path
    backtest_summary_csv: Path | None


def _collect_autocorrection(
    corrections: list[dict[str, Any]],
    cfg: dict[str, Any],
    path: tuple[str, ...],
    normalized_value: Any,
    reason: str,
) -> None:
    if not _is_key_present(cfg, path):
        return
    original = _get_nested_value(cfg, path)
    if original == normalized_value:
        return
    corrections.append({"field": ".".join(path), "original": original, "normalized": normalized_value, "reason": reason})


def _collect_normalization_autocorrections(
    cfg: dict[str, Any],
    scope_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    fac_cfg: dict[str, Any],
    research_cfg: dict[str, Any],
    back_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    corrections: list[dict[str, Any]] = []
    for path, normalized, reason in [
        (("run", "factor_scope"), scope_cfg["factor_scope"], "invalid_or_unsupported_value"),
        (("run", "eval_axis"), scope_cfg["eval_axis"], "invalid_or_incompatible_scope"),
        (("run", "standardization"), scope_cfg["standardization"], "invalid_for_scope_or_unsupported_value"),
        (("data", "adapter"), data_cfg["adapter"], "unknown_without_plugins"),
        (("data", "mode"), data_cfg["mode"], "invalid_or_incompatible_scope"),
        (("data", "adapter_plugin_on_error"), data_cfg["adapter_plugin_on_error"], "invalid_enum_fallback"),
        (("data", "request_timeout_sec"), data_cfg["request_timeout_sec"], "bounded_min_value"),
        (("data", "min_rows_per_asset"), data_cfg["min_rows_per_asset"], "bounded_min_value"),
        (("factor", "on_missing"), fac_cfg["on_missing"], "invalid_enum_fallback"),
        (("factor", "plugin_on_error"), fac_cfg["plugin_on_error"], "invalid_enum_fallback"),
        (("factor", "expression_on_error"), fac_cfg["expression_on_error"], "invalid_enum_fallback"),
        (("factor", "combination_on_error"), fac_cfg["combination_on_error"], "invalid_enum_fallback"),
        (("research", "missing_policy"), research_cfg["missing_policy"], "invalid_enum_fallback"),
        (("research", "transform_plugin_on_error"), research_cfg["transform_plugin_on_error"], "invalid_enum_fallback"),
        (("research", "quantiles"), research_cfg["quantiles"], "bounded_min_value"),
        (("research", "ic_rolling_window"), research_cfg["ic_rolling_window"], "bounded_min_value"),
        (("research", "annualization_days"), research_cfg["annualization_days"], "bounded_min_value"),
        (("research", "ts_standardize_window"), research_cfg["ts_standardize_window"], "bounded_min_value"),
        (("research", "ts_quantile_lookback"), research_cfg["ts_quantile_lookback"], "bounded_min_value"),
        (("research", "ts_signal_lags"), research_cfg["ts_signal_lags"], "invalid_values_removed_or_defaulted"),
        (("backtest", "strategy", "plugin_on_error"), back_cfg["strategy_plugin_on_error"], "invalid_enum_fallback"),
        (("backtest", "strategy", "rebalance_every"), back_cfg["rebalance_every"], "bounded_min_value"),
        (("backtest", "execution_delay_days"), back_cfg["execution_delay_days"], "bounded_min_value"),
    ]:
        _collect_autocorrection(corrections, cfg, path, normalized, reason)
    if _is_key_present(cfg, ("research", "preprocess_steps")):
        _collect_autocorrection(
            corrections,
            cfg,
            ("research", "preprocess_steps"),
            research_cfg["preprocess_steps"],
            "unsupported_steps_removed_or_defaulted",
        )
    return corrections


def _handle_autocorrection_governance(corrections: list[dict[str, Any]], governance_cfg: dict[str, Any]) -> None:
    if not corrections:
        return
    mode = str(governance_cfg["config_mode"]).lower()
    fail_on_autocorrect = bool(governance_cfg["fail_on_autocorrect"])
    if mode == "strict" or fail_on_autocorrect:
        sample = corrections[:8]
        detail = "\n".join(f"- {x['field']}: {x['original']!r} -> {x['normalized']!r} ({x['reason']})" for x in sample)
        extra = "" if len(corrections) <= len(sample) else f"\n- ... and {len(corrections) - len(sample)} more"
        raise ValueError(f"Config auto-correction detected under strict governance:\n{detail}{extra}")
    if mode == "warn":
        for row in corrections:
            LOGGER.warning(
                "Config auto-correct applied [%s]: %r -> %r (%s)",
                row["field"],
                row["original"],
                row["normalized"],
                row["reason"],
            )
        return
    LOGGER.info("Config auto-correct count=%s (compat mode).", len(corrections))


def _run_leakage_guard(
    governance_cfg: dict[str, Any],
    panel: pd.DataFrame,
    requested_factors: list[str],
    expression_dependencies: set[str],
    combination_dependencies: set[str],
) -> dict[str, Any]:
    mode = str(governance_cfg["leakage_guard_mode"]).lower()
    referenced = sorted({str(x).strip() for x in [*requested_factors, *expression_dependencies, *combination_dependencies] if str(x).strip()})
    forbidden_referenced = sorted([x for x in referenced if _is_forbidden_leakage_name(x)])
    forbidden_panel_columns = sorted([str(c) for c in panel.columns if _is_forbidden_leakage_name(str(c))])
    issues: list[str] = []
    if forbidden_referenced:
        issues.append(f"forbidden referenced labels/future columns: {forbidden_referenced}")
    if forbidden_panel_columns and forbidden_referenced:
        issues.append(f"panel also contains forbidden columns: {forbidden_panel_columns}")
    report = {
        "mode": mode,
        "issues": issues,
        "forbidden_referenced": forbidden_referenced,
        "forbidden_panel_columns": forbidden_panel_columns,
        "blocked": False,
    }
    if not issues or mode == "off":
        return report
    if mode == "warn":
        for msg in issues:
            LOGGER.warning("Leakage guard warning: %s", msg)
        return report
    report["blocked"] = True
    raise ValueError("Leakage guard blocked run:\n- " + "\n- ".join(issues))


def run_from_config(
    config: str | Path | dict[str, Any],
    out_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
    validate_schema: bool = True,
    run_context: RunContext | None = None,
) -> ConfigRunResult:
    raw_cfg_loaded = load_run_config(config) if not isinstance(config, dict) else dict(config)
    raw_cfg, alias_events = normalize_run_config_aliases(raw_cfg_loaded)
    governance_cfg = _normalize_run_governance_cfg(raw_cfg)
    schema_warnings = validate_run_config_schema(raw_cfg, strict=True) if validate_schema else []
    scope_cfg = _normalize_factor_scope(raw_cfg)
    data_cfg = _normalize_data_cfg(raw_cfg, scope=scope_cfg["factor_scope"])
    fac_cfg = _normalize_factor_cfg(raw_cfg)
    research_cfg = _normalize_research_cfg(raw_cfg, scope=scope_cfg["factor_scope"], profile=str(governance_cfg["research_profile"]))
    back_cfg = _normalize_backtest_cfg(raw_cfg, scope=scope_cfg["factor_scope"])
    universe_cfg = _normalize_universe_cfg(raw_cfg)
    autocorrections = _collect_normalization_autocorrections(
        cfg=raw_cfg,
        scope_cfg=scope_cfg,
        data_cfg=data_cfg,
        fac_cfg=fac_cfg,
        research_cfg=research_cfg,
        back_cfg=back_cfg,
    )
    _handle_autocorrection_governance(autocorrections, governance_cfg=governance_cfg)

    context = coerce_run_context(run_context=run_context, out_dir=out_dir, repo_root=repo_root)
    out = context.outputs.ensure_root()
    timings: dict[str, float] = {}
    captured_warnings: list[Any] = []

    adapter_stage = DataAdapterWorkflow(data_cfg=data_cfg, scope_cfg=scope_cfg, out_dir=out, timings=timings).run()
    panel = adapter_stage.panel
    load_report = adapter_stage.load_report
    mode_report = adapter_stage.mode_report
    adapter_validation_report = adapter_stage.adapter_validation_report
    adapter_audit_tables = adapter_stage.adapter_audit_tables
    data_adapter_registry = adapter_stage.adapter_registry
    data_adapter_validator_registry = adapter_stage.adapter_validator_registry
    data_lineage = describe_panel_lineage(panel=panel, data_cfg=data_cfg, load_report=load_report, mode_report=mode_report)
    data_lineage_path = write_json_artifact(out / "data_lineage.json", data_lineage)
    config_hash = stable_hash(
        {
            "scope": scope_cfg,
            "data": data_cfg,
            "factor": fac_cfg,
            "research": research_cfg,
            "backtest": back_cfg,
            "governance": governance_cfg,
        }
    )

    with timed_stage("build_factor_registry", timings=timings, logger_name="factorlab.workflows.config_runner"):
        factor_registry = build_factor_registry(
            plugin_dirs=fac_cfg["plugin_dirs"] if fac_cfg["auto_discover"] else [],
            plugin_specs=fac_cfg["plugins"],
            on_plugin_error=fac_cfg["plugin_on_error"],
        )
    with timed_stage("build_transform_registry", timings=timings, logger_name="factorlab.workflows.config_runner"):
        transform_registry = build_transform_registry(
            plugin_dirs=research_cfg["transform_plugin_dirs"] if research_cfg["transform_auto_discover"] else [],
            plugin_specs=research_cfg["transform_plugins"],
            on_plugin_error=research_cfg["transform_plugin_on_error"],
            include_defaults=True,
        )

    requested_factors = list(fac_cfg["names"])
    auto_discovered_requested_factors: list[str] = []
    if not requested_factors and fac_cfg["auto_discover_from_panel"]:
        ignored = {"date", "asset", "open", "high", "low", "close", "volume", "mkt_cap", "industry"}
        auto_discovered_requested_factors = [
            str(col) for col in panel.columns if str(col) not in ignored and not str(col).startswith("fwd_ret_")
        ]
        requested_factors = list(auto_discovered_requested_factors)
        if auto_discovered_requested_factors:
            LOGGER.info("factor.names 未显式配置，已从面板自动发现 %s 个因子列。", len(auto_discovered_requested_factors))
        else:
            LOGGER.warning("factor.names 为空，且未在面板中发现可研究的因子列。")

    expressions = dict(fac_cfg["expressions"])
    combinations = list(fac_cfg["combinations"])
    expression_outputs = set(expressions.keys())
    expression_dependencies: set[str] = set()
    for expr in expressions.values():
        expression_dependencies.update(extract_expression_dependencies(expr))
    expression_dependencies -= expression_outputs
    combination_outputs = {str(x.get("name")).strip() for x in combinations if str(x.get("name", "")).strip()}
    combination_dependencies: set[str] = set()
    for spec in combinations:
        weights = spec.get("weights", {})
        if isinstance(weights, dict):
            combination_dependencies.update(str(k).strip() for k in weights if str(k).strip())
        combination_dependencies.update(str(x).strip() for x in spec.get("orthogonalize_to", []) if str(x).strip())
    if not fac_cfg["names"]:
        for name in sorted(expression_outputs | combination_outputs):
            if name and name not in requested_factors:
                requested_factors.append(name)
    if not requested_factors:
        raise RuntimeError("No factors configured or discovered. 请在 factor.names 显式填写因子名，或在 panel 中提供可直接研究的因子列。")

    auto_factor_candidates = sorted(
        (set(requested_factors) - expression_outputs - combination_outputs) | expression_dependencies | combination_dependencies
    )
    factor_preflight_report = preflight_factor_candidates(
        factor_names=auto_factor_candidates,
        panel=panel,
        factor_registry=factor_registry,
        expression_outputs=expression_outputs,
        combination_outputs=combination_outputs,
        on_missing=fac_cfg["on_missing"],
    )
    auto_factor_candidates = list(factor_preflight_report.get("resolved", auto_factor_candidates))
    leakage_guard_report = _run_leakage_guard(
        governance_cfg=governance_cfg,
        panel=panel,
        requested_factors=requested_factors,
        expression_dependencies=expression_dependencies,
        combination_dependencies=combination_dependencies,
    )

    with timed_stage("factor_compute", timings=timings, logger_name="factorlab.workflows.config_runner"):
        candidate_factors, precheck_skipped_factors = filter_factors_by_available_columns(
            panel=panel,
            factor_names=auto_factor_candidates,
            on_missing=fac_cfg["on_missing"],
            factor_registry=factor_registry,
        )
        required_fields = resolve_required_fields(
            scope_cfg=scope_cfg,
            data_cfg=data_cfg,
            factor_names=candidate_factors,
            research_cfg=research_cfg,
            factor_registry=factor_registry,
        )
        validate_required_fields(panel, required=required_fields)
        panel, computed_factors, candidate_after_precheck = compute_factors(
            panel,
            factor_names=candidate_factors,
            on_missing=fac_cfg["on_missing"],
            registry=factor_registry,
        )
        panel, computed_expression_factors, skipped_expression_factors, expression_errors = apply_factor_expressions(
            panel,
            expressions=expressions,
            on_error=fac_cfg["expression_on_error"],
        )
        panel, computed_combination_factors, skipped_combination_factors, combination_errors = apply_factor_combinations(
            panel,
            combinations=combinations,
            on_error=fac_cfg["combination_on_error"],
        )

    unresolved_requested = [f for f in requested_factors if f not in panel.columns]
    if unresolved_requested:
        if fac_cfg["on_missing"] == "warn_skip":
            LOGGER.warning("Skip unresolved requested factors due to factor.on_missing=warn_skip: %s", unresolved_requested)
        else:
            raise KeyError(f"Requested factors missing after compute/expression steps: {unresolved_requested}")
    effective_factors = [f for f in requested_factors if f in panel.columns]
    if not effective_factors:
        raise RuntimeError("No effective requested factors available after compute/expression steps.")

    effective_factor_definitions = build_effective_factor_definitions(
        factor_names=effective_factors,
        factor_registry=factor_registry,
        expressions=expressions,
        combinations=combinations,
        panel_columns=list(panel.columns),
    )
    factor_definition_outputs = write_factor_definition_artifacts(out_dir=out, definitions=effective_factor_definitions)
    overview_extra_files = {
        "data_lineage": data_lineage_path,
        "factor_definitions": factor_definition_outputs["factor_definitions_csv"],
        "factor_definitions_json": factor_definition_outputs["factor_definitions_json"],
    }

    strategy_registry_for_run: dict[str, Any] = {}
    strategy_definition_outputs: dict[str, Path] = {}
    effective_strategy_definition: StrategyDefinition | None = None
    if back_cfg["enabled"]:
        with timed_stage("build_strategy_registry", timings=timings, logger_name="factorlab.workflows.config_runner"):
            strategy_registry_for_run = build_strategy_registry(
                plugin_dirs=back_cfg["strategy_plugin_dirs"] if back_cfg["strategy_auto_discover"] else [],
                plugin_specs=back_cfg["strategy_plugins"],
                on_plugin_error=back_cfg["strategy_plugin_on_error"],
                include_defaults=False,
            )
        effective_strategy_definition = resolve_strategy_definition(back_cfg=back_cfg, strategy_registry=strategy_registry_for_run)
        strategy_definition_outputs = write_strategy_definition_artifacts(out_dir=out, definitions=[effective_strategy_definition])
        overview_extra_files.update(
            {
                "strategy_definitions": strategy_definition_outputs["strategy_definitions_csv"],
                "strategy_definitions_json": strategy_definition_outputs["strategy_definitions_json"],
            }
        )

    universe_report = None
    if scope_cfg["factor_scope"] == "cs" and universe_cfg["enabled"]:
        with timed_stage("universe_filter", timings=timings, logger_name="factorlab.workflows.config_runner"):
            panel, universe_report = apply_universe_filter(panel, config=universe_cfg["config"])

    custom_transform_report: dict[str, Any] = {"applied": [], "skipped": [], "errors": []}
    transform_preflight_report: dict[str, Any] = {
        "kind": "transform",
        "requested": [],
        "available": sorted(transform_registry.keys()),
        "resolved": [],
        "missing": [],
        "on_missing": "raise",
        "alias_hits": {},
        "skipped": "no_custom_transforms",
    }
    if research_cfg["custom_transforms"]:
        with timed_stage("custom_transforms_preflight", timings=timings, logger_name="factorlab.workflows.config_runner"):
            transform_preflight_report = preflight_custom_transforms(
                transform_specs=research_cfg["custom_transforms"],
                transform_registry=transform_registry,
            )
        with timed_stage("custom_transforms", timings=timings, logger_name="factorlab.workflows.config_runner"):
            panel, custom_transform_report = apply_custom_transforms(
                panel=panel,
                factor_names=effective_factors,
                transform_specs=research_cfg["custom_transforms"],
                transform_registry=transform_registry,
            )

    panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)
    if panel.empty:
        raise RuntimeError("No data available after preprocessing.")

    stop_after = str(governance_cfg.get("stop_after", "backtest"))
    backtest_summary_csv: Path | None = None
    strategy_preflight_report: dict[str, Any] = {
        "kind": "strategy",
        "requested": [],
        "available": [],
        "resolved": [],
        "missing": [],
        "on_missing": "raise",
        "alias_hits": {},
        "skipped": "backtest_not_run",
    }

    if stop_after == "factor":
        with timed_stage("factor_stage_export", timings=timings, logger_name="factorlab.workflows.config_runner"):
            stage_data_dir = out / "data"
            stage_data_dir.mkdir(parents=True, exist_ok=True)
            factor_stage_panel = stage_data_dir / "panel_after_factor.parquet"
            panel.to_parquet(factor_stage_panel, index=False)
            outputs = write_stage_only_outputs(
                out_dir=context.outputs,
                stage="factor",
                summary_row={
                    "stage": "factor",
                    "status": "stopped",
                    "rows": int(len(panel)),
                    "assets": int(panel["asset"].nunique()),
                    "factors": ",".join(effective_factors),
                },
                extra_tables=[Path(x) for x in adapter_audit_tables.values()]
                + [factor_definition_outputs["factor_definitions_csv"]]
                + ([strategy_definition_outputs["strategy_definitions_csv"]] if strategy_definition_outputs else []),
                config_payload={"scope": scope_cfg, "research": research_cfg, "effective_factors": effective_factors},
                overview_files=overview_extra_files,
            )
            outputs["factor_stage_panel"] = factor_stage_panel
    else:
        outputs, research_warnings = run_research_stage(
            panel=panel,
            effective_factors=effective_factors,
            scope_cfg=scope_cfg,
            research_cfg=research_cfg,
            out_dir=context.outputs,
            overview_files=overview_extra_files,
            timings=timings,
        )
        captured_warnings.extend(research_warnings)
        if stop_after == "backtest":
            with timed_stage("backtest", timings=timings, logger_name="factorlab.workflows.config_runner"):
                backtest_summary_csv, strategy_preflight_report = run_optional_backtest(
                    panel=panel,
                    factors=effective_factors,
                    scope_cfg=scope_cfg,
                    back_cfg=back_cfg,
                    out_dir=out,
                    strategy_registry=strategy_registry_for_run,
                )

    meta = {
        "scope": scope_cfg,
        "runtime": {"repo_root": str(context.repo_root), "out_dir": str(context.out_dir), "encoding": context.text_encoding},
        "data": {
            "config": data_cfg,
            "lineage": data_lineage,
            "load_report": load_report,
            "adapter_validation_report": adapter_validation_report,
            "mode_report": mode_report,
            "required_fields": required_fields,
            "adapter_audit_tables": adapter_audit_tables,
            "adapter_plugin_config": {
                "auto_discover": data_cfg["adapter_auto_discover"],
                "plugin_dirs": data_cfg["adapter_plugin_dirs"],
                "plugins": data_cfg["adapter_plugins"],
                "plugin_on_error": data_cfg["adapter_plugin_on_error"],
                "registry_size": len(data_adapter_registry),
                "registry_adapters": sorted(data_adapter_registry.keys()),
            },
            "adapter_validator_plugin_config": {
                "auto_discover": data_cfg["adapter_auto_discover"],
                "plugin_dirs": data_cfg["adapter_plugin_dirs"],
                "plugins": data_cfg["adapter_plugins"],
                "plugin_on_error": data_cfg["adapter_plugin_on_error"],
                "registry_size": len(data_adapter_validator_registry),
                "registry_validators": sorted(data_adapter_validator_registry.keys()),
            },
        },
        "factors": {
            "requested": requested_factors,
            "configured_names": list(fac_cfg["names"]),
            "placeholder_detected": bool(fac_cfg["placeholder_detected"]),
            "auto_discover_from_panel": bool(fac_cfg["auto_discover_from_panel"]),
            "auto_discovered_requested": auto_discovered_requested_factors,
            "auto_factor_candidates": auto_factor_candidates,
            "candidate_after_precheck": candidate_after_precheck,
            "skipped_in_precheck": precheck_skipped_factors,
            "effective": effective_factors,
            "computed_factors": computed_factors,
            "computed_expression_factors": computed_expression_factors,
            "skipped_expression_factors": skipped_expression_factors,
            "expression_errors": expression_errors,
            "computed_combination_factors": computed_combination_factors,
            "skipped_combination_factors": skipped_combination_factors,
            "combination_errors": combination_errors,
            "unresolved_requested": unresolved_requested,
            "on_missing": fac_cfg["on_missing"],
            "expression_on_error": fac_cfg["expression_on_error"],
            "combination_on_error": fac_cfg["combination_on_error"],
            "expressions": expressions,
            "expression_dependencies": sorted(expression_dependencies),
            "combinations": combinations,
            "combination_dependencies": sorted(combination_dependencies),
            "preflight_report": factor_preflight_report,
            "effective_definitions": [asdict(item) for item in effective_factor_definitions],
            "definition_artifacts": {k: str(v) for k, v in factor_definition_outputs.items()},
            "plugin_config": {
                "auto_discover": fac_cfg["auto_discover"],
                "plugin_dirs": fac_cfg["plugin_dirs"],
                "plugins": fac_cfg["plugins"],
                "plugin_on_error": fac_cfg["plugin_on_error"],
                "registry_size": len(factor_registry),
                "registry_factors": sorted(factor_registry.keys()),
            },
        },
        "research": {
            "config": research_cfg,
            "custom_transform_report": custom_transform_report,
            "transform_preflight_report": transform_preflight_report,
            "transform_plugin_config": {
                "auto_discover": research_cfg["transform_auto_discover"],
                "plugin_dirs": research_cfg["transform_plugin_dirs"],
                "plugins": research_cfg["transform_plugins"],
                "plugin_on_error": research_cfg["transform_plugin_on_error"],
                "registry_size": len(transform_registry),
                "registry_transforms": sorted(transform_registry.keys()),
            },
        },
        "config_governance": {
            **governance_cfg,
            "alias_migration_count": len(alias_events),
            "alias_migrations": alias_events,
            "autocorrection_count": len(autocorrections),
            "autocorrections": autocorrections,
            "schema_validation_enabled": bool(validate_schema),
        },
        "leakage_guard": leakage_guard_report,
        "schema_warnings": schema_warnings,
        "universe_filter": {
            "enabled": universe_cfg["enabled"],
            "report": universe_report.__dict__ if hasattr(universe_report, "__dict__") else universe_report,
        },
        "backtest": {
            "config": back_cfg,
            "summary_csv": str(backtest_summary_csv) if backtest_summary_csv else None,
            "strategy_preflight_report": strategy_preflight_report,
            "effective_definition": asdict(effective_strategy_definition) if effective_strategy_definition else None,
            "definition_artifacts": {k: str(v) for k, v in strategy_definition_outputs.items()},
            "strategy_registry_size": len(strategy_registry_for_run),
            "strategy_registry": sorted(strategy_registry_for_run.keys()),
        },
        "rows_after_pipeline": int(len(panel)),
        "assets_after_pipeline": int(panel["asset"].nunique()),
        "dates_after_pipeline": int(panel["date"].nunique()),
        "config_hash": config_hash,
        "timings_seconds": timings,
        "warning_summary": summarize_captured_warnings(captured_warnings, logger_name="factorlab.workflows.config_runner"),
        "outputs": {
            **{k: str(v) for k, v in outputs.items()},
            **adapter_audit_tables,
            "data_lineage_json": str(data_lineage_path),
            **{k: str(v) for k, v in factor_definition_outputs.items()},
            **{k: str(v) for k, v in strategy_definition_outputs.items()},
        },
    }
    meta_path = context.outputs.write_text("run_meta.json", json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    manifest_path = context.outputs.write_text("run_manifest.json", json.dumps(context.runtime_manifest, indent=2, ensure_ascii=False))
    experiment_registry_path = write_json_artifact(
        out / "experiment_registry.json",
        build_experiment_registry(
            out_dir=out,
            config_hash=config_hash,
            data_lineage=data_lineage,
            runtime_manifest=context.runtime_manifest,
            scope=scope_cfg,
            governance=governance_cfg,
            factors={"effective": effective_factors, "definition_artifacts": {k: str(v) for k, v in factor_definition_outputs.items()}},
            strategy={
                "enabled": bool(back_cfg["enabled"]),
                "mode": back_cfg["strategy_mode"],
                "definition_artifacts": {k: str(v) for k, v in strategy_definition_outputs.items()},
                "definition": asdict(effective_strategy_definition) if effective_strategy_definition else None,
            },
            outputs={
                **{k: str(v) for k, v in outputs.items()},
                "run_meta_json": str(meta_path),
                "run_manifest_json": str(manifest_path),
                "data_lineage_json": str(data_lineage_path),
                **{k: str(v) for k, v in factor_definition_outputs.items()},
                **{k: str(v) for k, v in strategy_definition_outputs.items()},
            },
        ),
    )
    meta["outputs"]["experiment_registry_json"] = str(experiment_registry_path)
    meta_path = context.outputs.write_text("run_meta.json", json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    return ConfigRunResult(
        out_dir=out,
        index_html=Path(outputs["index_html"]),
        summary_csv=Path(outputs["summary_csv"]),
        run_meta_json=meta_path,
        run_manifest_json=manifest_path,
        backtest_summary_csv=backtest_summary_csv,
    )
