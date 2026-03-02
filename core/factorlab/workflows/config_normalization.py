"""运行配置归一化与基础类型转换工具。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from factorlab.config import UniverseFilterConfig
from factorlab.factors import normalize_factor_combinations
from factorlab.utils import get_logger

LOGGER = get_logger("factorlab.workflows.config_normalization")

FactorScope = Literal["ts", "cs"]

FORBIDDEN_LEAKAGE_PREFIXES = ("fwd_ret_", "label_", "target_", "future_")
FORBIDDEN_LEAKAGE_EXACT = {"label", "target", "y", "future_return"}

PLACEHOLDER_FACTOR_NAMES = {"factor_name", "factor", "your_factor"}
NON_FACTOR_BASE_COLUMNS = {
    "date",
    "asset",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "mkt_cap",
    "industry",
    "is_tradable",
    "can_buy",
    "can_sell",
    "benchmark_ret",
}

CONFIG_ALIAS_PATHS: dict[tuple[str, ...], tuple[str, ...]] = {
    ("run", "scope"): ("run", "factor_scope"),
    ("run", "axis"): ("run", "eval_axis"),
    ("run", "std"): ("run", "standardization"),
    ("run", "standardize"): ("run", "standardization"),
    ("run", "profile"): ("run", "research_profile"),
    ("run", "stage_stop"): ("run", "stop_after"),
    ("data", "required_cols"): ("data", "fields_required"),
    ("data", "min_rows"): ("data", "min_rows_per_asset"),
    ("data", "source"): ("data", "path"),
    ("data", "dataset"): ("data", "path"),
    ("data", "input"): ("data", "path"),
    ("data", "raw_dir"): ("data", "path"),
    ("factor", "list"): ("factor", "names"),
    ("factor", "missing"): ("factor", "on_missing"),
    ("research", "ic_window"): ("research", "ic_rolling_window"),
    ("research", "q"): ("research", "quantiles"),
    ("research", "steps"): ("research", "preprocess_steps"),
    ("research", "missing"): ("research", "missing_policy"),
    ("backtest", "strategy", "type"): ("backtest", "strategy", "mode"),
}

RESEARCH_PROFILE_DEFAULTS: dict[str, dict[str, dict[str, Any]]] = {
    "cs": {
        "fast": {
            "horizons": [1, 5],
            "quantiles": 3,
            "ic_rolling_window": 10,
            "annualization_days": 252,
            "missing_policy": "drop",
            "preprocess_steps": ["standardize"],
        },
        "dev": {
            "horizons": [1, 5, 10],
            "quantiles": 5,
            "ic_rolling_window": 15,
            "annualization_days": 252,
            "missing_policy": "drop",
            "preprocess_steps": ["winsorize", "standardize"],
        },
        "full": {
            "horizons": [1, 5, 10, 20],
            "quantiles": 5,
            "ic_rolling_window": 20,
            "annualization_days": 252,
            "missing_policy": "drop",
            "preprocess_steps": ["winsorize", "standardize", "neutralize"],
        },
    },
    "ts": {
        "fast": {
            "horizons": [1, 5],
            "quantiles": 3,
            "ic_rolling_window": 15,
            "annualization_days": 252,
            "missing_policy": "drop",
            "ts_standardize_window": 40,
            "ts_quantile_lookback": 60,
            "ts_signal_lags": [0, 1, 2],
        },
        "dev": {
            "horizons": [1, 5, 10],
            "quantiles": 5,
            "ic_rolling_window": 20,
            "annualization_days": 252,
            "missing_policy": "drop",
            "ts_standardize_window": 60,
            "ts_quantile_lookback": 80,
            "ts_signal_lags": [0, 1, 2, 5],
        },
        "full": {
            "horizons": [1, 5, 10, 20],
            "quantiles": 5,
            "ic_rolling_window": 30,
            "annualization_days": 252,
            "missing_policy": "drop",
            "ts_standardize_window": 60,
            "ts_quantile_lookback": 80,
            "ts_signal_lags": [0, 1, 2, 5, 10],
        },
    },
}


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, list):
        return value
    return [value]


def to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def get_nested_value(obj: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def is_key_present(obj: dict[str, Any], path: tuple[str, ...]) -> bool:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return False
        cur = cur[key]
    return True


def set_nested_value(obj: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cur: dict[str, Any] = obj
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def pop_nested_value(obj: dict[str, Any], path: tuple[str, ...]) -> tuple[Any, bool]:
    parents: list[tuple[dict[str, Any], str]] = []
    cur: dict[str, Any] | None = obj
    for key in path[:-1]:
        if not isinstance(cur, dict) or key not in cur or not isinstance(cur[key], dict):
            return None, False
        parents.append((cur, key))
        cur = cur[key]
    if not isinstance(cur, dict):
        return None, False
    leaf = path[-1]
    if leaf not in cur:
        return None, False
    value = cur.pop(leaf)
    for parent, key in reversed(parents):
        node = parent.get(key)
        if isinstance(node, dict) and not node:
            parent.pop(key, None)
        else:
            break
    return value, True


def normalize_run_config_aliases(cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """将配置中的别名键迁移为标准键，并返回迁移记录。"""
    out = deepcopy(cfg)
    alias_events: list[dict[str, Any]] = []
    for alias_path, canonical_path in CONFIG_ALIAS_PATHS.items():
        value, exists = pop_nested_value(out, alias_path)
        if not exists:
            continue
        alias_name = ".".join(alias_path)
        canonical_name = ".".join(canonical_path)
        if is_key_present(out, canonical_path):
            prev_value = get_nested_value(out, canonical_path)
            set_nested_value(out, canonical_path, value)
            alias_events.append(
                {
                    "alias": alias_name,
                    "canonical": canonical_name,
                    "applied": True,
                    "reason": "canonical_overridden_by_alias",
                    "alias_value": value,
                    "canonical_prev_value": prev_value,
                }
            )
            continue
        set_nested_value(out, canonical_path, value)
        alias_events.append(
            {
                "alias": alias_name,
                "canonical": canonical_name,
                "applied": True,
                "reason": "migrated",
                "alias_value": value,
            }
        )
    return out, alias_events


def infer_adapter_from_path(path_like: Any) -> str:
    if path_like is None:
        return ""
    text = str(path_like).strip()
    if not text:
        return ""
    p = Path(text)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".csv":
        return "csv"
    return "raw_dir"


def is_forbidden_leakage_name(name: str) -> bool:
    key = str(name).strip().lower()
    if key in FORBIDDEN_LEAKAGE_EXACT:
        return True
    return key.startswith(FORBIDDEN_LEAKAGE_PREFIXES)


def discover_panel_factor_columns(panel: pd.DataFrame) -> list[str]:
    """从面板中自动发现可作为因子研究对象的列。"""
    names: list[str] = []
    for col in panel.columns:
        col_name = str(col).strip()
        if not col_name:
            continue
        if col_name in NON_FACTOR_BASE_COLUMNS:
            continue
        if is_forbidden_leakage_name(col_name):
            continue
        if col_name.startswith("fwd_ret_"):
            continue
        names.append(col_name)
    return names


def normalize_run_governance_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    run_cfg = as_dict(cfg.get("run"))
    config_mode = str(run_cfg.get("config_mode", "compat")).strip().lower()
    if config_mode not in {"strict", "warn", "compat"}:
        raise ValueError("run.config_mode must be one of ['strict', 'warn', 'compat'].")

    leakage_guard_mode = str(run_cfg.get("leakage_guard_mode", "strict")).strip().lower()
    if leakage_guard_mode not in {"strict", "warn", "off"}:
        raise ValueError("run.leakage_guard_mode must be one of ['strict', 'warn', 'off'].")
    stop_after = str(run_cfg.get("stop_after", "backtest")).strip().lower()
    if stop_after not in {"factor", "research", "backtest"}:
        raise ValueError("run.stop_after must be one of ['factor', 'research', 'backtest'].")
    research_profile = str(run_cfg.get("research_profile", "full")).strip().lower()
    if research_profile not in {"fast", "dev", "full"}:
        raise ValueError("run.research_profile must be one of ['fast', 'dev', 'full'].")

    fail_on_autocorrect = to_bool(run_cfg.get("fail_on_autocorrect"), False)
    return {
        "config_mode": config_mode,
        "leakage_guard_mode": leakage_guard_mode,
        "stop_after": stop_after,
        "research_profile": research_profile,
        "fail_on_autocorrect": fail_on_autocorrect,
    }


def normalize_factor_scope(cfg: dict[str, Any]) -> dict[str, Any]:
    run_cfg = as_dict(cfg.get("run"))
    scope = str(run_cfg.get("factor_scope", "cs")).strip().lower()
    if scope not in {"ts", "cs"}:
        LOGGER.warning("Invalid run.factor_scope='%s'. Use 'cs' as fallback.", scope)
        scope = "cs"

    default_eval = "time" if scope == "ts" else "cross_section"
    eval_axis = str(run_cfg.get("eval_axis", default_eval)).strip().lower()
    if eval_axis not in {"time", "cross_section"}:
        LOGGER.warning("Invalid run.eval_axis='%s'. Use '%s'.", eval_axis, default_eval)
        eval_axis = default_eval
    if scope == "ts" and eval_axis != "time":
        LOGGER.warning("TS scope only supports eval_axis=time. Auto-correct applied.")
        eval_axis = "time"
    if scope == "cs" and eval_axis != "cross_section":
        LOGGER.warning("CS scope only supports eval_axis=cross_section. Auto-correct applied.")
        eval_axis = "cross_section"

    if scope == "ts":
        default_std = "ts_rolling_zscore"
        allowed_std = {"ts_rolling_zscore", "zscore", "none"}
    else:
        default_std = "cs_zscore"
        allowed_std = {"cs_zscore", "cs_rank", "cs_robust_zscore", "none"}
    standardization = str(run_cfg.get("standardization", default_std)).strip().lower()
    if standardization not in allowed_std:
        LOGGER.warning(
            "Invalid run.standardization='%s' for scope=%s. Use '%s'.",
            standardization,
            scope,
            default_std,
        )
        standardization = default_std

    return {
        "factor_scope": scope,
        "eval_axis": eval_axis,
        "standardization": standardization,
    }


def normalize_data_cfg(cfg: dict[str, Any], scope: FactorScope) -> dict[str, Any]:
    data_cfg = as_dict(cfg.get("data"))
    adapter_plugin_dirs = [str(x).strip() for x in as_list(data_cfg.get("adapter_plugin_dirs")) if str(x).strip()]
    adapter_plugins = [x for x in as_list(data_cfg.get("adapter_plugins")) if x is not None and x != ""]
    adapter_auto_discover = to_bool(data_cfg.get("adapter_auto_discover"), bool(adapter_plugin_dirs))
    adapter_plugin_on_error = str(data_cfg.get("adapter_plugin_on_error", "raise")).strip().lower()
    if adapter_plugin_on_error not in {"raise", "warn_skip"}:
        LOGGER.warning("Invalid data.adapter_plugin_on_error='%s'. Use 'raise'.", adapter_plugin_on_error)
        adapter_plugin_on_error = "raise"

    path_like = data_cfg.get("path")
    inferred_adapter = infer_adapter_from_path(path_like)
    adapter_raw = str(data_cfg.get("adapter", "")).strip().lower()
    if adapter_raw in {"", "auto", "infer"}:
        adapter = inferred_adapter or "synthetic"
    else:
        adapter = "raw_dir" if adapter_raw == "raw" else adapter_raw
        if adapter == "synthetic" and inferred_adapter:
            LOGGER.info("data.adapter 显式为 synthetic，忽略 data.path 的读取器推断。")

    builtin = {"synthetic", "sina", "stooq", "parquet", "csv", "raw_dir"}
    if adapter not in builtin and not (adapter_plugins or adapter_plugin_dirs):
        if inferred_adapter:
            LOGGER.warning(
                "Unknown data.adapter='%s' and no adapter plugins configured. "
                "Fallback to adapter inferred from data.path: '%s'.",
                adapter,
                inferred_adapter,
            )
            adapter = inferred_adapter
        else:
            LOGGER.warning("Unknown data.adapter='%s' and no adapter plugins configured. Use synthetic.", adapter)
            adapter = "synthetic"

    mode_default = "single_asset" if scope == "ts" else "panel"
    mode = str(data_cfg.get("mode", mode_default)).strip().lower()
    if mode not in {"single_asset", "panel"}:
        LOGGER.warning("Invalid data.mode='%s'. Use '%s'.", mode, mode_default)
        mode = mode_default

    default_fields = ["date", "close"] if scope == "ts" else ["date", "asset", "close"]
    fields_required = [str(x).strip() for x in as_list(data_cfg.get("fields_required")) if str(x).strip()]
    if not fields_required:
        fields_required = default_fields

    return {
        "mode": mode,
        "adapter": adapter,
        "path": path_like,
        "data_dir": data_cfg.get("data_dir"),
        "raw_pattern": str(data_cfg.get("raw_pattern", "*.parquet,*.csv")).strip() or "*.parquet,*.csv",
        "raw_asset_from_filename": to_bool(data_cfg.get("raw_asset_from_filename"), True),
        "symbols": [str(x).strip() for x in as_list(data_cfg.get("symbols")) if str(x).strip()],
        "start_date": data_cfg.get("start_date"),
        "end_date": data_cfg.get("end_date"),
        "request_timeout_sec": max(3, to_int(data_cfg.get("request_timeout_sec"), 20)),
        "min_rows_per_asset": max(1, to_int(data_cfg.get("min_rows_per_asset"), 30)),
        "asset": data_cfg.get("asset"),
        "sanitize": to_bool(data_cfg.get("sanitize"), True),
        "duplicate_policy": str(data_cfg.get("duplicate_policy", "last")).strip().lower(),
        "fields_required": fields_required,
        "synthetic": as_dict(data_cfg.get("synthetic")),
        "adapter_auto_discover": adapter_auto_discover,
        "adapter_plugin_dirs": adapter_plugin_dirs,
        "adapter_plugins": adapter_plugins,
        "adapter_plugin_on_error": adapter_plugin_on_error,
    }


def normalize_factor_expressions(raw: Any, strict: bool = False) -> dict[str, str]:
    out: dict[str, str] = {}

    def _add(name: Any, expr: Any) -> None:
        fac_name = str(name).strip()
        fac_expr = str(expr).strip()
        if not fac_name or not fac_expr:
            raise ValueError(f"Invalid expression entry: name={name!r}, expression={expr!r}")
        out[fac_name] = fac_expr

    if raw is None:
        return {}

    try:
        if isinstance(raw, dict):
            for k, v in raw.items():
                _add(k, v)
            return out
        if isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, str):
                    text = entry.strip()
                    if "=" not in text:
                        raise ValueError(f"Expression list item must be 'name=expr', got: {entry!r}")
                    name, expr = text.split("=", 1)
                    _add(name, expr)
                    continue
                if isinstance(entry, dict):
                    name = entry.get("name")
                    expr = entry.get("expression")
                    if name is None or expr is None:
                        raise ValueError(f"Expression dict item requires keys 'name' and 'expression': {entry!r}")
                    _add(name, expr)
                    continue
                raise ValueError(f"Unsupported expression list item type: {type(entry)}")
            return out
        if isinstance(raw, str):
            text = raw.strip()
            if "=" not in text:
                raise ValueError(f"Expression string must be 'name=expr', got: {raw!r}")
            name, expr = text.split("=", 1)
            _add(name, expr)
            return out
        raise ValueError(f"Unsupported factor.expressions type: {type(raw)}")
    except Exception:
        if strict:
            raise
        LOGGER.warning("Ignore invalid factor.expressions value: %r", raw)
        return {}


def normalize_requested_factor_names(raw: Any) -> tuple[list[str], bool]:
    names: list[str] = []
    placeholder_detected = False
    seen: set[str] = set()
    for item in as_list(raw):
        name = str(item).strip()
        if not name:
            continue
        if name.lower() in PLACEHOLDER_FACTOR_NAMES:
            placeholder_detected = True
            continue
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names, placeholder_detected


def normalize_factor_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    fac_cfg = as_dict(cfg.get("factor"))
    names, placeholder_detected = normalize_requested_factor_names(fac_cfg.get("names"))
    auto_discover_from_panel = to_bool(fac_cfg.get("auto_discover_from_panel"), True)
    on_missing = str(fac_cfg.get("on_missing", "raise")).strip().lower()
    if on_missing not in {"raise", "warn_skip"}:
        LOGGER.warning("Invalid factor.on_missing='%s'. Use 'raise'.", on_missing)
        on_missing = "raise"
    plugin_on_error = str(fac_cfg.get("plugin_on_error", "raise")).strip().lower()
    if plugin_on_error not in {"raise", "warn_skip"}:
        LOGGER.warning("Invalid factor.plugin_on_error='%s'. Use 'raise'.", plugin_on_error)
        plugin_on_error = "raise"

    plugin_dirs = [str(x).strip() for x in as_list(fac_cfg.get("plugin_dirs")) if str(x).strip()]
    auto_discover = to_bool(fac_cfg.get("auto_discover"), bool(plugin_dirs))
    plugins = [x for x in as_list(fac_cfg.get("plugins")) if x is not None and x != ""]
    expression_on_error = str(fac_cfg.get("expression_on_error", "raise")).strip().lower()
    if expression_on_error not in {"raise", "warn_skip"}:
        LOGGER.warning("Invalid factor.expression_on_error='%s'. Use 'raise'.", expression_on_error)
        expression_on_error = "raise"

    expressions = normalize_factor_expressions(fac_cfg.get("expressions"), strict=False)
    combination_on_error = str(fac_cfg.get("combination_on_error", "raise")).strip().lower()
    if combination_on_error not in {"raise", "warn_skip"}:
        LOGGER.warning("Invalid factor.combination_on_error='%s'. Use 'raise'.", combination_on_error)
        combination_on_error = "raise"
    combinations = normalize_factor_combinations(fac_cfg.get("combinations"), strict=False)

    return {
        "names": names,
        "placeholder_detected": placeholder_detected,
        "auto_discover_from_panel": auto_discover_from_panel,
        "on_missing": on_missing,
        "plugins": plugins,
        "plugin_dirs": plugin_dirs,
        "auto_discover": auto_discover,
        "plugin_on_error": plugin_on_error,
        "expression_on_error": expression_on_error,
        "expressions": expressions,
        "combinations": combinations,
        "combination_on_error": combination_on_error,
    }


def normalize_custom_transforms(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in as_list(raw):
        if isinstance(entry, str):
            name = entry.strip()
            if not name:
                continue
            out.append({"name": name, "kwargs": {}, "on_error": "raise"})
            continue
        if isinstance(entry, dict):
            name = str(entry.get("name", "")).strip()
            if not name:
                LOGGER.warning("Ignore custom transform without name: %r", entry)
                continue
            kwargs_raw = entry.get("kwargs", {})
            if not isinstance(kwargs_raw, dict):
                LOGGER.warning("Ignore custom transform '%s': kwargs must be a dict.", name)
                continue
            on_error = str(entry.get("on_error", "raise")).strip().lower()
            if on_error not in {"raise", "warn_skip"}:
                LOGGER.warning("Invalid custom transform on_error='%s' for '%s'. Use 'raise'.", on_error, name)
                on_error = "raise"
            out.append({"name": name, "kwargs": kwargs_raw, "on_error": on_error})
            continue
        LOGGER.warning("Ignore unsupported custom transform entry type: %s", type(entry))
    return out


def normalize_research_cfg(cfg: dict[str, Any], scope: FactorScope, profile: str) -> dict[str, Any]:
    raw = as_dict(cfg.get("research"))
    profile_key = profile if profile in {"fast", "dev", "full"} else "full"
    profile_defaults = RESEARCH_PROFILE_DEFAULTS[scope][profile_key]

    horizons_raw = as_list(raw.get("horizons", profile_defaults["horizons"]))
    horizons = sorted({_to for _to in (to_int(x, 0) for x in horizons_raw) if _to > 0})
    if not horizons:
        horizons = list(profile_defaults["horizons"])

    missing_policy = str(raw.get("missing_policy", profile_defaults["missing_policy"])).strip().lower()
    allowed_missing = {"drop", "fill_zero", "ffill_by_asset", "cs_median_by_date", "keep"}
    if missing_policy not in allowed_missing:
        LOGGER.warning("Invalid research.missing_policy='%s'. Use 'drop'.", missing_policy)
        missing_policy = "drop"

    default_steps = list(profile_defaults["preprocess_steps"]) if scope == "cs" else []
    steps_raw = [str(x).strip().lower() for x in as_list(raw.get("preprocess_steps", default_steps)) if str(x).strip()]
    allowed_steps = {"winsorize", "standardize", "neutralize"}
    preprocess_steps: list[str] = []
    for step in steps_raw:
        if step not in allowed_steps:
            LOGGER.warning("Ignore unsupported preprocess step: %s", step)
            continue
        if step not in preprocess_steps:
            preprocess_steps.append(step)
    if scope == "cs" and not preprocess_steps:
        preprocess_steps = default_steps.copy()

    ts_signal_lags: list[int] = []
    for x in as_list(raw.get("ts_signal_lags", profile_defaults.get("ts_signal_lags"))):
        try:
            v = int(x)
        except Exception:
            continue
        if v < 0:
            continue
        ts_signal_lags.append(v)
    if not ts_signal_lags:
        ts_signal_lags = list(profile_defaults.get("ts_signal_lags", [0, 1, 2, 5, 10]))
    ts_signal_lags = sorted(set(ts_signal_lags))
    if 0 not in ts_signal_lags:
        ts_signal_lags = [0, *ts_signal_lags]

    transform_plugin_dirs = [str(x).strip() for x in as_list(raw.get("transform_plugin_dirs")) if str(x).strip()]
    transform_plugins = [x for x in as_list(raw.get("transform_plugins")) if x is not None and x != ""]
    transform_auto_discover = to_bool(raw.get("transform_auto_discover"), bool(transform_plugin_dirs))
    transform_plugin_on_error = str(raw.get("transform_plugin_on_error", "raise")).strip().lower()
    if transform_plugin_on_error not in {"raise", "warn_skip"}:
        LOGGER.warning("Invalid research.transform_plugin_on_error='%s'. Use 'raise'.", transform_plugin_on_error)
        transform_plugin_on_error = "raise"

    out = {
        "profile": profile_key,
        "horizons": horizons,
        "quantiles": max(2, to_int(raw.get("quantiles"), int(profile_defaults["quantiles"]))),
        "ic_rolling_window": max(5, to_int(raw.get("ic_rolling_window"), int(profile_defaults["ic_rolling_window"]))),
        "annualization_days": max(1, to_int(raw.get("annualization_days"), int(profile_defaults["annualization_days"]))),
        "ts_standardize_window": max(5, to_int(raw.get("ts_standardize_window"), int(profile_defaults.get("ts_standardize_window", 60)))),
        "ts_quantile_lookback": max(10, to_int(raw.get("ts_quantile_lookback"), int(profile_defaults.get("ts_quantile_lookback", 60)))),
        "ts_signal_lags": ts_signal_lags,
        "missing_policy": missing_policy,
        "preprocess_steps": preprocess_steps,
        "transform_auto_discover": transform_auto_discover,
        "transform_plugin_dirs": transform_plugin_dirs,
        "transform_plugins": transform_plugins,
        "transform_plugin_on_error": transform_plugin_on_error,
        "custom_transforms": normalize_custom_transforms(raw.get("custom_transforms")),
    }
    if scope == "cs":
        out["winsorize"] = as_dict(raw.get("winsorize"))
        out["neutralize"] = as_dict(raw.get("neutralize"))
    return out


def normalize_backtest_cfg(cfg: dict[str, Any], scope: FactorScope) -> dict[str, Any]:
    raw = as_dict(cfg.get("backtest"))
    enabled = to_bool(raw.get("enabled"), False)
    strategy_cfg = as_dict(raw.get("strategy"))
    default_mode = "sign" if scope == "ts" else "longshort"
    strategy_mode = str(strategy_cfg.get("mode", default_mode)).strip().lower() or default_mode
    builtin_modes = {"sign", "topk", "longshort", "flex", "meanvar"}
    if strategy_mode not in builtin_modes:
        LOGGER.info("Using custom strategy mode '%s'. Will resolve via strategy plugin registry.", strategy_mode)

    strategy_plugin_on_error = str(strategy_cfg.get("plugin_on_error", "raise")).strip().lower()
    if strategy_plugin_on_error not in {"raise", "warn_skip"}:
        LOGGER.warning("Invalid backtest.strategy.plugin_on_error='%s'. Use 'raise'.", strategy_plugin_on_error)
        strategy_plugin_on_error = "raise"

    strategy_plugin_dirs = [str(x).strip() for x in as_list(strategy_cfg.get("plugin_dirs")) if str(x).strip()]
    strategy_auto_discover = to_bool(strategy_cfg.get("auto_discover"), bool(strategy_plugin_dirs))
    strategy_plugins = [x for x in as_list(strategy_cfg.get("plugins")) if x is not None and x != ""]

    return {
        "enabled": enabled,
        "strategy_mode": strategy_mode,
        "top_k": max(1, to_int(strategy_cfg.get("top_k"), 20)),
        "long_short_quantile": max(0.05, min(0.49, to_float(strategy_cfg.get("long_short_quantile"), 0.2))),
        "long_fraction": max(0.05, min(0.95, to_float(strategy_cfg.get("long_fraction"), 0.2))),
        "short_fraction": max(0.0, min(0.95, to_float(strategy_cfg.get("short_fraction"), 0.2))),
        "long_only": to_bool(strategy_cfg.get("long_only"), False),
        "rebalance_every": max(1, to_int(strategy_cfg.get("rebalance_every"), 1)),
        "weight_scheme": str(strategy_cfg.get("weight_scheme", "equal")).strip().lower(),
        "max_weight": strategy_cfg.get("max_weight"),
        "risk_aversion": max(1e-6, to_float(strategy_cfg.get("risk_aversion"), 5.0)),
        "gross_target": max(0.1, to_float(strategy_cfg.get("gross_target"), 1.0)),
        "net_target": to_float(strategy_cfg.get("net_target"), 0.0),
        "sign_threshold": max(0.0, to_float(strategy_cfg.get("sign_threshold"), 0.0)),
        "commission_bps": to_float(raw.get("commission_bps"), 3.0),
        "slippage_bps": to_float(raw.get("slippage_bps"), 2.0),
        "leverage": max(0.1, to_float(raw.get("leverage"), 1.0)),
        "execution_delay_days": max(0, to_int(raw.get("execution_delay_days"), 1)),
        "execution_price_col": str(raw.get("execution_price_col", "close")).strip(),
        "max_turnover": raw.get("max_turnover"),
        "max_abs_weight": raw.get("max_abs_weight"),
        "max_gross_exposure": raw.get("max_gross_exposure"),
        "max_net_exposure": raw.get("max_net_exposure"),
        "enforce_industry_neutral": to_bool(raw.get("enforce_industry_neutral"), False),
        "industry_col": str(raw.get("industry_col", "industry")).strip(),
        "benchmark_mode": str(raw.get("benchmark_mode", "none")).strip().lower(),
        "benchmark_return_col": str(raw.get("benchmark_return_col", "benchmark_ret")).strip(),
        "strategy_plugin_dirs": strategy_plugin_dirs,
        "strategy_plugins": strategy_plugins,
        "strategy_auto_discover": strategy_auto_discover,
        "strategy_plugin_on_error": strategy_plugin_on_error,
    }


def normalize_universe_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = as_dict(cfg.get("universe_filter"))
    return {
        "enabled": to_bool(raw.get("enabled"), False),
        "config": UniverseFilterConfig(
            min_close=to_float(raw.get("min_close"), 0.0),
            min_history_days=max(1, to_int(raw.get("min_history_days"), 1)),
            min_median_dollar_volume=max(0.0, to_float(raw.get("min_median_dollar_volume"), 0.0)),
            liquidity_lookback=max(2, to_int(raw.get("liquidity_lookback"), 20)),
        ),
    }


__all__ = [
    "NON_FACTOR_BASE_COLUMNS",
    "PLACEHOLDER_FACTOR_NAMES",
    "RESEARCH_PROFILE_DEFAULTS",
    "as_dict",
    "as_list",
    "discover_panel_factor_columns",
    "get_nested_value",
    "infer_adapter_from_path",
    "is_forbidden_leakage_name",
    "is_key_present",
    "normalize_backtest_cfg",
    "normalize_custom_transforms",
    "normalize_data_cfg",
    "normalize_factor_cfg",
    "normalize_factor_expressions",
    "normalize_factor_scope",
    "normalize_research_cfg",
    "normalize_requested_factor_names",
    "normalize_run_config_aliases",
    "normalize_run_governance_cfg",
    "normalize_universe_cfg",
    "pop_nested_value",
    "set_nested_value",
    "to_bool",
    "to_float",
    "to_int",
    "to_optional_float",
]
