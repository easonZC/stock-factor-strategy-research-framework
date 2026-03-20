"""Config composition, override, and schema validation helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from factorlab.factors import extract_expression_dependencies, normalize_factor_combinations
from factorlab.workflows.config_normalization import (
    CONFIG_RULES,
    as_dict as _as_dict,
    as_list as _as_list,
    infer_adapter_from_path as _infer_adapter_from_path,
    normalize_factor_expressions as _normalize_factor_expressions,
    normalize_requested_factor_names as _normalize_requested_factor_names,
    normalize_run_config_aliases,
)


def _load_run_config_with_imports(path: Path, chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved in chain:
        cycle = " -> ".join(str(x) for x in [*chain, resolved])
        raise ValueError(f"Circular config imports detected: {cycle}")
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")

    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML object: {resolved}")

    import_items: list[str] = []
    for import_key in ("imports", "extends"):
        if import_key not in payload:
            continue
        import_items.extend([str(x).strip() for x in _as_list(payload.pop(import_key)) if str(x).strip()])

    merged: dict[str, Any] = {}
    for item in import_items:
        import_path = Path(item)
        if not import_path.is_absolute():
            import_path = resolved.parent / import_path
        parent_payload = _load_run_config_with_imports(import_path, chain=(*chain, resolved))
        merged = deep_merge_dict(merged, parent_payload)
    return deep_merge_dict(merged, payload)


def load_run_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config with recursive `imports` / `extends` support."""
    return _load_run_config_with_imports(Path(path))


def deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries and return a new object."""
    out = dict(base)
    for key, val in overlay.items():
        cur = out.get(key)
        if isinstance(cur, dict) and isinstance(val, dict):
            out[key] = deep_merge_dict(cur, val)
        else:
            out[key] = val
    return out


def _parse_override(override: str) -> tuple[list[str], str, Any]:
    text = str(override).strip()
    op = None
    pos = -1
    for cand in ("+=", "-=", "="):
        pos = text.find(cand)
        if pos > 0:
            op = cand
            break
    if op is None:
        raise ValueError(f"Invalid override '{override}'. Expected format: key.path=value (or += / -=).")
    path_raw = text[:pos].strip()
    value_raw = text[pos + len(op) :]
    path = [x.strip() for x in path_raw.split(".") if x.strip()]
    if not path:
        raise ValueError(f"Invalid override '{override}'. Empty key path.")
    try:
        value = yaml.safe_load(value_raw)
    except Exception:
        value = value_raw
    return path, op, value


def _apply_override_value(current: Any, op: str, value: Any, override: str) -> Any:
    if op == "=":
        return value

    if op == "+=":
        if current is None:
            out = []
            if isinstance(value, list):
                out.extend(list(value))
            else:
                out.append(value)
            return out

        if isinstance(current, list):
            out = list(current)
            if isinstance(value, list):
                out.extend(list(value))
            else:
                out.append(value)
            return out

        if isinstance(current, dict):
            if not isinstance(value, dict):
                raise ValueError(f"Invalid override '{override}': '+=' on dict target requires object value.")
            return deep_merge_dict(current, value)

        raise ValueError(f"Invalid override '{override}': '+=' only supports list/dict targets.")

    if op == "-=":
        if current is None:
            raise ValueError(f"Invalid override '{override}': '-=' target path does not exist.")

        if isinstance(current, list):
            out = list(current)
            targets = list(value) if isinstance(value, list) else [value]
            for target in targets:
                out = [x for x in out if x != target]
            return out

        if isinstance(current, dict):
            out = dict(current)
            targets = list(value) if isinstance(value, list) else [value]
            for target in targets:
                if not isinstance(target, str):
                    raise ValueError(f"Invalid override '{override}': '-=' dict target requires string key(s).")
                out.pop(target, None)
            return out

        raise ValueError(f"Invalid override '{override}': '-=' only supports list/dict targets.")

    raise ValueError(f"Unsupported override operator in '{override}': {op}")


def apply_config_override(cfg: dict[str, Any], override: str) -> dict[str, Any]:
    """Apply a single override item and return a new config."""
    path, op, value = _parse_override(override)
    out = deepcopy(cfg)
    node: dict[str, Any] = out
    for seg in path[:-1]:
        nxt = node.get(seg)
        if not isinstance(nxt, dict):
            nxt = {}
            node[seg] = nxt
        node = nxt
    leaf = path[-1]
    node[leaf] = _apply_override_value(
        current=node.get(leaf),
        op=op,
        value=value,
        override=override,
    )
    return out


def compose_run_config_with_alias_report(
    config_paths: list[str | Path],
    overrides: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compose runtime config from multiple YAML files plus overrides."""
    if not config_paths:
        raise ValueError("At least one config path is required.")

    merged = load_run_config(config_paths[0])
    for path in config_paths[1:]:
        merged = deep_merge_dict(merged, load_run_config(path))

    for ov in overrides or []:
        merged = apply_config_override(merged, ov)
    return normalize_run_config_aliases(merged)


def compose_run_config(
    config_paths: list[str | Path],
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Compose runtime config from multiple YAML files plus overrides."""
    merged, _ = compose_run_config_with_alias_report(config_paths=config_paths, overrides=overrides)
    return merged


def _validate_data_adapter_fragment(
    data_cfg: dict[str, Any],
    adapter: str,
    errors: list[str],
    warnings_out: list[str],
) -> None:
    if adapter == "synthetic":
        syn = _as_dict(data_cfg.get("synthetic"))
        if not syn:
            warnings_out.append("data.synthetic: not provided; runtime defaults will be used.")
            return
        for fld, min_val in [("n_assets", 1), ("n_days", 60)]:
            if fld not in syn:
                continue
            try:
                val = int(syn.get(fld))
            except Exception:
                errors.append(f"data.synthetic.{fld}: must be an integer >= {min_val}.")
                continue
            if val < min_val:
                errors.append(f"data.synthetic.{fld}: must be >= {min_val}.")
        if "start_date" in syn:
            ts = pd.to_datetime(syn.get("start_date"), errors="coerce")
            if pd.isna(ts):
                errors.append("data.synthetic.start_date: must be parseable date string.")
        return

    if adapter == "stooq":
        symbols = [str(x).strip() for x in _as_list(data_cfg.get("symbols")) if str(x).strip()]
        if not symbols:
            errors.append("data.symbols: requires non-empty list for stooq adapter.")
        for fld in ["start_date", "end_date"]:
            raw = data_cfg.get(fld)
            if raw is None:
                continue
            ts = pd.to_datetime(raw, errors="coerce")
            if pd.isna(ts):
                errors.append(f"data.{fld}: must be parseable date string when provided.")
        try:
            timeout = int(data_cfg.get("request_timeout_sec", 20))
            if timeout <= 0:
                errors.append("data.request_timeout_sec: must be > 0.")
        except Exception:
            errors.append("data.request_timeout_sec: must be an integer > 0.")
        try:
            min_rows = int(data_cfg.get("min_rows_per_asset", 30))
            if min_rows <= 0:
                errors.append("data.min_rows_per_asset: must be > 0.")
        except Exception:
            errors.append("data.min_rows_per_asset: must be an integer > 0.")
        return

    if adapter == "sina":
        try:
            min_rows = int(data_cfg.get("min_rows_per_asset", 30))
            if min_rows <= 0:
                errors.append("data.min_rows_per_asset: must be > 0.")
        except Exception:
            errors.append("data.min_rows_per_asset: must be an integer > 0.")
        return

    if adapter in {"parquet", "csv"}:
        path = data_cfg.get("path")
        if path:
            p = Path(str(path))
            if p.suffix:
                suffix = p.suffix.lower()
                if adapter == "parquet" and suffix != ".parquet":
                    warnings_out.append("data.path: adapter=parquet but file suffix is not '.parquet'.")
                if adapter == "csv" and suffix != ".csv":
                    warnings_out.append("data.path: adapter=csv but file suffix is not '.csv'.")
            else:
                warnings_out.append("data.path: no file suffix detected; if using directory, set data.adapter=raw_dir.")
        return

    if adapter == "raw_dir":
        path = data_cfg.get("path")
        if not path:
            errors.append("data.path: required when data.adapter is raw_dir.")
        pattern = str(data_cfg.get("raw_pattern", "*.parquet,*.csv")).strip()
        if not pattern:
            errors.append("data.raw_pattern: cannot be empty when data.adapter is raw_dir.")


def validate_run_config_schema(cfg: dict[str, Any], strict: bool = True) -> list[str]:
    """Validate config structure before runtime and return non-blocking warnings."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a YAML object (dict).")
    cfg, alias_events = normalize_run_config_aliases(cfg)
    for event in alias_events:
        if event["applied"]:
            warnings.append(
                f"alias migrated: {event['alias']} -> {event['canonical']} "
                "(建议改为标准键，避免后续歧义)。"
            )

    root_allowed = CONFIG_RULES.root_allowed_sections
    root_required = CONFIG_RULES.root_required_sections

    for key in sorted(set(cfg) - root_allowed):
        warnings.append(f"{key}: unknown root section (ignored by workflow).")
    for key in sorted(root_required):
        if key not in cfg or not isinstance(cfg.get(key), dict):
            errors.append(f"{key}: required object section is missing.")
    for key in ["run", "factor", "research", "backtest"]:
        if key not in cfg:
            warnings.append(f"{key}: missing section; runtime defaults will be used.")

    run_cfg = _as_dict(cfg.get("run"))
    scope = str(run_cfg.get("factor_scope", "cs")).strip().lower()
    default_eval_axis = CONFIG_RULES.default_eval_axis(scope)
    eval_axis = str(run_cfg.get("eval_axis", default_eval_axis)).strip().lower()
    default_std = CONFIG_RULES.default_standardization(scope)
    standardization = str(run_cfg.get("standardization", default_std)).strip().lower()
    if scope not in CONFIG_RULES.scopes:
        errors.append(f"run.factor_scope: must be one of {sorted(CONFIG_RULES.scopes)}.")
    if eval_axis not in CONFIG_RULES.eval_axes:
        errors.append(f"run.eval_axis: must be one of {sorted(CONFIG_RULES.eval_axes)}.")
    config_mode = str(run_cfg.get("config_mode", "compat")).strip().lower()
    if config_mode not in CONFIG_RULES.config_modes:
        errors.append(f"run.config_mode: must be one of {sorted(CONFIG_RULES.config_modes)}.")
    leakage_guard_mode = str(run_cfg.get("leakage_guard_mode", "strict")).strip().lower()
    if leakage_guard_mode not in CONFIG_RULES.leakage_guard_modes:
        errors.append(f"run.leakage_guard_mode: must be one of {sorted(CONFIG_RULES.leakage_guard_modes)}.")
    stop_after = str(run_cfg.get("stop_after", "backtest")).strip().lower()
    if stop_after not in CONFIG_RULES.stop_after_modes:
        errors.append(f"run.stop_after: must be one of {sorted(CONFIG_RULES.stop_after_modes)}.")
    research_profile = str(run_cfg.get("research_profile", "full")).strip().lower()
    if research_profile not in CONFIG_RULES.research_profiles:
        errors.append(f"run.research_profile: must be one of {sorted(CONFIG_RULES.research_profiles)}.")
    if "fail_on_autocorrect" in run_cfg and not isinstance(run_cfg.get("fail_on_autocorrect"), bool):
        errors.append("run.fail_on_autocorrect: must be boolean when provided.")

    cs_std = CONFIG_RULES.allowed_standardization("cs")
    ts_std = CONFIG_RULES.allowed_standardization("ts")
    if scope == "cs" and standardization not in cs_std:
        errors.append(f"run.standardization: for cs scope use one of {sorted(cs_std)}.")
    if scope == "ts" and standardization not in ts_std:
        errors.append(f"run.standardization: for ts scope use one of {sorted(ts_std)}.")

    data_cfg = _as_dict(cfg.get("data"))
    adapter = str(data_cfg.get("adapter", "")).strip().lower()
    path_like = data_cfg.get("path")
    if adapter in {"", "auto", "infer"}:
        adapter = _infer_adapter_from_path(path_like) or "synthetic"
    if adapter == "raw":
        adapter = "raw_dir"
    mode_default = CONFIG_RULES.default_data_mode(scope)
    mode = str(data_cfg.get("mode", mode_default)).strip().lower()
    builtin_adapters = CONFIG_RULES.builtin_adapters
    adapter_plugin_dirs = _as_list(data_cfg.get("adapter_plugin_dirs"))
    adapter_plugins = _as_list(data_cfg.get("adapter_plugins"))
    has_adapter_plugins = bool(adapter_plugin_dirs) or bool(adapter_plugins)
    if adapter not in builtin_adapters and not has_adapter_plugins:
        errors.append(
            "data.adapter: unknown adapter without data adapter plugins. "
            "Use built-in ['synthetic', 'sina', 'stooq', 'parquet', 'csv', 'raw_dir'] "
            "or configure data.adapter_plugin_dirs/data.adapter_plugins."
        )
    if mode not in set(CONFIG_RULES.default_data_mode_by_scope.values()):
        errors.append(f"data.mode: must be one of {sorted(set(CONFIG_RULES.default_data_mode_by_scope.values()))}.")
    if adapter in {"parquet", "csv"} and not data_cfg.get("path"):
        errors.append("data.path: required when data.adapter is parquet/csv.")
    if adapter == "raw_dir" and not data_cfg.get("path"):
        errors.append("data.path: required when data.adapter is raw_dir.")
    if adapter == "sina" and not data_cfg.get("data_dir"):
        errors.append("data.data_dir: required when data.adapter is sina.")
    if adapter == "stooq" and not _as_list(data_cfg.get("symbols")):
        errors.append("data.symbols: required when data.adapter is stooq.")
    adapter_plugin_on_error = str(data_cfg.get("adapter_plugin_on_error", "raise")).strip().lower()
    if adapter_plugin_on_error not in CONFIG_RULES.on_error_modes:
        errors.append(f"data.adapter_plugin_on_error: must be one of {sorted(CONFIG_RULES.on_error_modes)}.")
    if adapter in builtin_adapters:
        _validate_data_adapter_fragment(data_cfg=data_cfg, adapter=adapter, errors=errors, warnings_out=warnings)

    factor_cfg = _as_dict(cfg.get("factor"))
    factor_names = _as_list(factor_cfg.get("names"))
    if not factor_names:
        warnings.append("factor.names: not provided; runtime will auto-discover factor columns from panel.")
    normalized_factor_names, placeholder_detected = _normalize_requested_factor_names(factor_cfg.get("names"))
    if placeholder_detected:
        warnings.append(
            "factor.names: placeholder names detected (e.g. factor_name); runtime will ignore placeholders and auto-discover factors."
        )
    if any(not str(x).strip() for x in factor_names):
        errors.append("factor.names: contains empty factor name.")
    if not normalized_factor_names and factor_names and not placeholder_detected:
        warnings.append("factor.names: all provided names were empty/invalid after normalization; runtime will auto-discover.")
    if "auto_discover_from_panel" in factor_cfg and not isinstance(factor_cfg.get("auto_discover_from_panel"), bool):
        errors.append("factor.auto_discover_from_panel: must be boolean when provided.")
    on_missing = str(factor_cfg.get("on_missing", "raise")).strip().lower()
    if on_missing not in CONFIG_RULES.on_error_modes:
        errors.append(f"factor.on_missing: must be one of {sorted(CONFIG_RULES.on_error_modes)}.")
    plugin_on_error = str(factor_cfg.get("plugin_on_error", "raise")).strip().lower()
    if plugin_on_error not in CONFIG_RULES.on_error_modes:
        errors.append(f"factor.plugin_on_error: must be one of {sorted(CONFIG_RULES.on_error_modes)}.")
    expression_on_error = str(factor_cfg.get("expression_on_error", "raise")).strip().lower()
    if expression_on_error not in CONFIG_RULES.on_error_modes:
        errors.append(f"factor.expression_on_error: must be one of {sorted(CONFIG_RULES.on_error_modes)}.")
    combination_on_error = str(factor_cfg.get("combination_on_error", "raise")).strip().lower()
    if combination_on_error not in CONFIG_RULES.on_error_modes:
        errors.append(f"factor.combination_on_error: must be one of {sorted(CONFIG_RULES.on_error_modes)}.")

    try:
        expressions = _normalize_factor_expressions(factor_cfg.get("expressions"), strict=True)
        for name, expr in expressions.items():
            if not str(name).strip():
                errors.append("factor.expressions: expression output name cannot be empty.")
            try:
                extract_expression_dependencies(expr)
            except Exception as exc:
                errors.append(f"factor.expressions[{name}]: invalid expression ({exc}).")
    except Exception as exc:
        errors.append(f"factor.expressions: invalid format ({exc}).")

    try:
        combinations = normalize_factor_combinations(factor_cfg.get("combinations"), strict=True)
        for spec in combinations:
            if not str(spec.get("name", "")).strip():
                errors.append("factor.combinations: each combination must have non-empty name.")
            weights = spec.get("weights", {})
            if not isinstance(weights, dict) or not weights:
                errors.append("factor.combinations: each combination must have non-empty weights mapping.")
                break
    except Exception as exc:
        errors.append(f"factor.combinations: invalid format ({exc}).")

    research_cfg = _as_dict(cfg.get("research"))
    horizons = _as_list(research_cfg.get("horizons"))
    if horizons:
        for h in horizons:
            try:
                if int(h) <= 0:
                    raise ValueError
            except Exception:
                errors.append(f"research.horizons: invalid horizon '{h}'.")
                break

    try:
        if int(research_cfg.get("quantiles", 5)) < 2:
            errors.append("research.quantiles: must be >= 2.")
    except Exception:
        errors.append("research.quantiles: must be an integer.")

    try:
        if int(research_cfg.get("ic_rolling_window", 20)) < 5:
            errors.append("research.ic_rolling_window: must be >= 5.")
    except Exception:
        errors.append("research.ic_rolling_window: must be an integer.")
    try:
        if int(research_cfg.get("annualization_days", 252)) < 1:
            errors.append("research.annualization_days: must be >= 1.")
    except Exception:
        errors.append("research.annualization_days: must be an integer.")

    ts_signal_lags_raw = _as_list(research_cfg.get("ts_signal_lags"))
    if ts_signal_lags_raw:
        for lag in ts_signal_lags_raw:
            try:
                if int(lag) < 0:
                    raise ValueError
            except Exception:
                errors.append(f"research.ts_signal_lags: invalid lag '{lag}', must be non-negative int.")
                break

    missing_policy = str(research_cfg.get("missing_policy", "drop")).strip().lower()
    allowed_missing = CONFIG_RULES.missing_policies
    if missing_policy not in allowed_missing:
        errors.append(f"research.missing_policy: must be one of {sorted(allowed_missing)}.")

    step_values = [str(x).strip().lower() for x in _as_list(research_cfg.get("preprocess_steps")) if str(x).strip()]
    allowed_steps = CONFIG_RULES.preprocess_steps
    for step in step_values:
        if step not in allowed_steps:
            errors.append(f"research.preprocess_steps: unsupported step '{step}'.")
            break

    transform_plugin_on_error = str(research_cfg.get("transform_plugin_on_error", "raise")).strip().lower()
    if transform_plugin_on_error not in CONFIG_RULES.on_error_modes:
        errors.append(f"research.transform_plugin_on_error: must be one of {sorted(CONFIG_RULES.on_error_modes)}.")

    custom_transforms = _as_list(research_cfg.get("custom_transforms"))
    for entry in custom_transforms:
        if isinstance(entry, str):
            if not entry.strip():
                errors.append("research.custom_transforms: transform name string cannot be empty.")
                break
            continue
        if not isinstance(entry, dict):
            errors.append("research.custom_transforms: each entry must be string or object.")
            break
        name = str(entry.get("name", "")).strip()
        if not name:
            errors.append("research.custom_transforms: object entry requires non-empty 'name'.")
            break
        kwargs = entry.get("kwargs")
        if kwargs is not None and not isinstance(kwargs, dict):
            errors.append(f"research.custom_transforms[{name}].kwargs: must be an object.")
            break
        on_error = str(entry.get("on_error", "raise")).strip().lower()
        if on_error not in {"raise", "warn_skip"}:
            errors.append(f"research.custom_transforms[{name}].on_error: must be 'raise' or 'warn_skip'.")
            break

    back_cfg = _as_dict(cfg.get("backtest"))
    strategy_cfg = _as_dict(back_cfg.get("strategy"))
    mode_val = str(strategy_cfg.get("mode", "")).strip().lower()
    builtin_modes = CONFIG_RULES.strategy_builtin_modes
    if mode_val and mode_val not in builtin_modes:
        has_plugins = bool(_as_list(strategy_cfg.get("plugins"))) or bool(_as_list(strategy_cfg.get("plugin_dirs")))
        if not has_plugins:
            errors.append(
                "backtest.strategy.mode: unknown custom mode without strategy plugins. "
                "Provide backtest.strategy.plugins or backtest.strategy.plugin_dirs."
            )

    strategy_plugin_on_error = str(strategy_cfg.get("plugin_on_error", "raise")).strip().lower()
    if strategy_plugin_on_error not in CONFIG_RULES.on_error_modes:
        errors.append(f"backtest.strategy.plugin_on_error: must be one of {sorted(CONFIG_RULES.on_error_modes)}.")

    bench_mode = str(back_cfg.get("benchmark_mode", "none")).strip().lower()
    if bench_mode not in CONFIG_RULES.benchmark_modes:
        errors.append(f"backtest.benchmark_mode: must be one of {sorted(CONFIG_RULES.benchmark_modes)}.")

    for fld in ["max_turnover", "max_abs_weight", "max_gross_exposure", "max_net_exposure"]:
        raw = back_cfg.get(fld, None)
        if raw is None:
            continue
        try:
            val = float(raw)
        except Exception:
            errors.append(f"backtest.{fld}: must be a number when provided.")
            continue
        if val < 0:
            errors.append(f"backtest.{fld}: must be >= 0.")

    if errors and strict:
        msg = "Run config schema validation failed:\n" + "\n".join(f"- {x}" for x in errors)
        raise ValueError(msg)

    if errors:
        warnings.extend(f"ERROR_AS_WARNING: {x}" for x in errors)
    return warnings


__all__ = [
    "apply_config_override",
    "compose_run_config",
    "compose_run_config_with_alias_report",
    "deep_merge_dict",
    "load_run_config",
    "validate_run_config_schema",
]
