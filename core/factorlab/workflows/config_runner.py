"""配置驱动工作流主入口。

职责：
1. 解析并合并多份配置（支持 imports 递归分层与 CLI 覆盖）
2. 统一配置归一化（含别名兼容、研究档位 profile、治理策略）
3. 执行数据加载、因子计算、研究报告与可选回测
4. 输出可审计元数据（run_meta / run_manifest / 质量审计表）
"""

from __future__ import annotations

import json
import time
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from factorlab.backtest import run_backtest
from factorlab.config import (
    AdapterConfig,
    BacktestConfig,
    CostConfig,
    NeutralizationConfig,
    ResearchConfig,
    SyntheticConfig,
    UniverseFilterConfig,
)
from factorlab.data import (
    PanelSanitizationConfig,
    apply_universe_filter,
    build_data_adapter_registry,
    build_data_adapter_validator_registry,
    generate_synthetic_panel,
    read_panel_directory,
    read_panel,
)
from factorlab.factors import (
    apply_factor_combinations,
    apply_factor_expressions,
    apply_factors,
    build_factor_registry,
    extract_expression_dependencies,
    normalize_factor_combinations,
)
from factorlab.preprocess import build_transform_registry
from factorlab.research import FactorResearchPipeline, TSResearchConfig, TimeSeriesFactorResearchPipeline
from factorlab.research.report import render_report
from factorlab.strategies import (
    FlexibleLongShortStrategy,
    LongShortQuantileStrategy,
    MeanVarianceOptimizerStrategy,
    Strategy,
    TopKLongStrategy,
    build_strategy_registry,
)
from factorlab.utils import get_logger, safe_slug, summarize_captured_warnings, timed_stage
from factorlab.workflows.runtime import collect_runtime_manifest

LOGGER = get_logger("factorlab.workflows.config_runner")


FactorScope = Literal["ts", "cs"]
EvalAxis = Literal["time", "cross_section"]
ConfigGovernanceMode = Literal["strict", "warn", "compat"]
LeakageGuardMode = Literal["strict", "warn", "off"]

FORBIDDEN_LEAKAGE_PREFIXES = ("fwd_ret_", "label_", "target_", "future_")
FORBIDDEN_LEAKAGE_EXACT = {"label", "target", "y", "future_return"}

FACTOR_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "momentum_20": {"close"},
    "volatility_20": {"close"},
    "liquidity_shock": {"volume"},
    "size": {"mkt_cap"},
}

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

# 配置别名兼容映射（alias -> canonical）
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

# 研究档位默认值（用于快速切换研究粒度与运行成本）
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


@dataclass(slots=True)
class ConfigRunResult:
    """配置运行结果文件指针。"""

    out_dir: Path
    index_html: Path
    summary_csv: Path
    run_meta_json: Path
    run_manifest_json: Path
    backtest_summary_csv: Path | None


@dataclass(slots=True)
class DataAdapterWorkflowResult:
    """数据适配器阶段输出。"""

    panel: pd.DataFrame
    load_report: dict[str, Any]
    mode_report: dict[str, Any]
    adapter_validation_report: dict[str, Any]
    adapter_audit_tables: dict[str, str]
    adapter_registry: dict[str, Any]
    adapter_validator_registry: dict[str, Any]
    adapter_cfg: AdapterConfig | None


class DataAdapterWorkflow:
    """数据适配器生命周期管理器（注册、校验、加载、审计）。"""

    def __init__(
        self,
        data_cfg: dict[str, Any],
        scope_cfg: dict[str, Any],
        out_dir: Path,
        timings: dict[str, float],
    ):
        self.data_cfg = data_cfg
        self.scope_cfg = scope_cfg
        self.out_dir = out_dir
        self.timings = timings
        self.logger_name = "factorlab.workflows.config_runner"

    def _build_adapter_registry(self) -> dict[str, Any]:
        with timed_stage("build_data_adapter_registry", timings=self.timings, logger_name=self.logger_name):
            return build_data_adapter_registry(
                plugin_dirs=self.data_cfg["adapter_plugin_dirs"] if self.data_cfg["adapter_auto_discover"] else [],
                plugin_specs=self.data_cfg["adapter_plugins"],
                on_plugin_error=self.data_cfg["adapter_plugin_on_error"],
                include_defaults=True,
            )

    def _build_adapter_validator_registry(self) -> dict[str, Any]:
        with timed_stage(
            "build_data_adapter_validator_registry",
            timings=self.timings,
            logger_name=self.logger_name,
        ):
            return build_data_adapter_validator_registry(
                plugin_dirs=self.data_cfg["adapter_plugin_dirs"] if self.data_cfg["adapter_auto_discover"] else [],
                plugin_specs=self.data_cfg["adapter_plugins"],
                on_plugin_error=self.data_cfg["adapter_plugin_on_error"],
                include_defaults=True,
            )

    def _validate_adapter(
        self,
        adapter_registry: dict[str, Any],
        validator_registry: dict[str, Any],
    ) -> tuple[AdapterConfig | None, dict[str, Any]]:
        adapter_cfg: AdapterConfig | None = None
        report: dict[str, Any] = {
            "adapter": self.data_cfg["adapter"],
            "validated": False,
            "validator_found": False,
            "validation_seconds": 0.0,
            "warnings": [],
        }
        with timed_stage(
            "validate_data_adapter_config",
            timings=self.timings,
            logger_name=self.logger_name,
        ):
            if self.data_cfg["adapter"] in adapter_registry:
                adapter_cfg = _build_adapter_config(self.data_cfg)
                report = _validate_adapter_config(
                    adapter=self.data_cfg["adapter"],
                    adapter_cfg=adapter_cfg,
                    validator_registry=validator_registry,
                )
        return adapter_cfg, report

    def _load_and_shape(
        self,
        adapter_registry: dict[str, Any],
        adapter_cfg: AdapterConfig | None,
    ) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
        with timed_stage("load_data", timings=self.timings, logger_name=self.logger_name):
            panel, load_report = _load_data(
                self.data_cfg,
                self.scope_cfg,
                adapter_registry=adapter_registry,
                adapter_cfg=adapter_cfg,
            )
            panel, mode_report = _ensure_mode_shape(panel, data_cfg=self.data_cfg)
        return panel, load_report, mode_report

    def _audit_quality(self, panel: pd.DataFrame, load_report: dict[str, Any]) -> dict[str, str]:
        with timed_stage("adapter_quality_audit", timings=self.timings, logger_name=self.logger_name):
            return _write_adapter_quality_audit_tables(
                panel=panel,
                data_cfg=self.data_cfg,
                load_report=load_report,
                out_dir=self.out_dir,
            )

    def run(self) -> DataAdapterWorkflowResult:
        adapter_registry = self._build_adapter_registry()
        validator_registry = self._build_adapter_validator_registry()
        adapter_cfg, adapter_validation_report = self._validate_adapter(
            adapter_registry=adapter_registry,
            validator_registry=validator_registry,
        )
        panel, load_report, mode_report = self._load_and_shape(
            adapter_registry=adapter_registry,
            adapter_cfg=adapter_cfg,
        )
        adapter_audit_tables = self._audit_quality(panel=panel, load_report=load_report)
        return DataAdapterWorkflowResult(
            panel=panel,
            load_report=load_report,
            mode_report=mode_report,
            adapter_validation_report=adapter_validation_report,
            adapter_audit_tables=adapter_audit_tables,
            adapter_registry=adapter_registry,
            adapter_validator_registry=validator_registry,
            adapter_cfg=adapter_cfg,
        )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, list):
        return value
    return [value]


def _to_bool(value: Any, default: bool) -> bool:
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


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _get_nested_value(obj: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _is_key_present(obj: dict[str, Any], path: tuple[str, ...]) -> bool:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return False
        cur = cur[key]
    return True


def _set_nested_value(obj: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cur: dict[str, Any] = obj
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def _pop_nested_value(obj: dict[str, Any], path: tuple[str, ...]) -> tuple[Any, bool]:
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
    # 清理因删除而产生的空字典，避免留下无意义节点。
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
        value, exists = _pop_nested_value(out, alias_path)
        if not exists:
            continue
        alias_name = ".".join(alias_path)
        canonical_name = ".".join(canonical_path)
        if _is_key_present(out, canonical_path):
            prev_value = _get_nested_value(out, canonical_path)
            _set_nested_value(out, canonical_path, value)
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
        _set_nested_value(out, canonical_path, value)
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


def _normalize_run_governance_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    run_cfg = _as_dict(cfg.get("run"))
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

    fail_on_autocorrect = _to_bool(run_cfg.get("fail_on_autocorrect"), False)
    return {
        "config_mode": config_mode,
        "leakage_guard_mode": leakage_guard_mode,
        "stop_after": stop_after,
        "research_profile": research_profile,
        "fail_on_autocorrect": fail_on_autocorrect,
    }


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
    corrections.append(
        {
            "field": ".".join(path),
            "original": original,
            "normalized": normalized_value,
            "reason": reason,
        }
    )


def _collect_normalization_autocorrections(
    cfg: dict[str, Any],
    scope_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    fac_cfg: dict[str, Any],
    research_cfg: dict[str, Any],
    back_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    corrections: list[dict[str, Any]] = []

    _collect_autocorrection(corrections, cfg, ("run", "factor_scope"), scope_cfg["factor_scope"], "invalid_or_unsupported_value")
    _collect_autocorrection(corrections, cfg, ("run", "eval_axis"), scope_cfg["eval_axis"], "invalid_or_incompatible_scope")
    _collect_autocorrection(
        corrections,
        cfg,
        ("run", "standardization"),
        scope_cfg["standardization"],
        "invalid_for_scope_or_unsupported_value",
    )

    _collect_autocorrection(corrections, cfg, ("data", "adapter"), data_cfg["adapter"], "unknown_without_plugins")
    _collect_autocorrection(corrections, cfg, ("data", "mode"), data_cfg["mode"], "invalid_or_incompatible_scope")
    _collect_autocorrection(
        corrections,
        cfg,
        ("data", "adapter_plugin_on_error"),
        data_cfg["adapter_plugin_on_error"],
        "invalid_enum_fallback",
    )
    _collect_autocorrection(corrections, cfg, ("data", "request_timeout_sec"), data_cfg["request_timeout_sec"], "bounded_min_value")
    _collect_autocorrection(corrections, cfg, ("data", "min_rows_per_asset"), data_cfg["min_rows_per_asset"], "bounded_min_value")

    _collect_autocorrection(corrections, cfg, ("factor", "on_missing"), fac_cfg["on_missing"], "invalid_enum_fallback")
    _collect_autocorrection(corrections, cfg, ("factor", "plugin_on_error"), fac_cfg["plugin_on_error"], "invalid_enum_fallback")
    _collect_autocorrection(
        corrections,
        cfg,
        ("factor", "expression_on_error"),
        fac_cfg["expression_on_error"],
        "invalid_enum_fallback",
    )
    _collect_autocorrection(
        corrections,
        cfg,
        ("factor", "combination_on_error"),
        fac_cfg["combination_on_error"],
        "invalid_enum_fallback",
    )

    _collect_autocorrection(corrections, cfg, ("research", "missing_policy"), research_cfg["missing_policy"], "invalid_enum_fallback")
    _collect_autocorrection(
        corrections,
        cfg,
        ("research", "transform_plugin_on_error"),
        research_cfg["transform_plugin_on_error"],
        "invalid_enum_fallback",
    )
    _collect_autocorrection(corrections, cfg, ("research", "quantiles"), research_cfg["quantiles"], "bounded_min_value")
    _collect_autocorrection(
        corrections,
        cfg,
        ("research", "ic_rolling_window"),
        research_cfg["ic_rolling_window"],
        "bounded_min_value",
    )
    _collect_autocorrection(
        corrections,
        cfg,
        ("research", "annualization_days"),
        research_cfg["annualization_days"],
        "bounded_min_value",
    )
    _collect_autocorrection(
        corrections,
        cfg,
        ("research", "ts_standardize_window"),
        research_cfg["ts_standardize_window"],
        "bounded_min_value",
    )
    _collect_autocorrection(
        corrections,
        cfg,
        ("research", "ts_quantile_lookback"),
        research_cfg["ts_quantile_lookback"],
        "bounded_min_value",
    )
    _collect_autocorrection(
        corrections,
        cfg,
        ("research", "ts_signal_lags"),
        research_cfg["ts_signal_lags"],
        "invalid_values_removed_or_defaulted",
    )
    if _is_key_present(cfg, ("research", "preprocess_steps")):
        _collect_autocorrection(
            corrections,
            cfg,
            ("research", "preprocess_steps"),
            research_cfg["preprocess_steps"],
            "unsupported_steps_removed_or_defaulted",
        )

    _collect_autocorrection(
        corrections,
        cfg,
        ("backtest", "strategy", "plugin_on_error"),
        back_cfg["strategy_plugin_on_error"],
        "invalid_enum_fallback",
    )
    _collect_autocorrection(
        corrections,
        cfg,
        ("backtest", "strategy", "rebalance_every"),
        back_cfg["rebalance_every"],
        "bounded_min_value",
    )
    _collect_autocorrection(
        corrections,
        cfg,
        ("backtest", "execution_delay_days"),
        back_cfg["execution_delay_days"],
        "bounded_min_value",
    )
    return corrections


def _handle_autocorrection_governance(
    corrections: list[dict[str, Any]],
    governance_cfg: dict[str, Any],
) -> None:
    if not corrections:
        return

    mode = str(governance_cfg["config_mode"]).lower()
    fail_on_autocorrect = bool(governance_cfg["fail_on_autocorrect"])
    if mode == "strict" or fail_on_autocorrect:
        sample = corrections[:8]
        detail = "\n".join(
            f"- {row['field']}: {row['original']!r} -> {row['normalized']!r} ({row['reason']})"
            for row in sample
        )
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


def _is_forbidden_leakage_name(name: str) -> bool:
    key = str(name).strip().lower()
    if key in FORBIDDEN_LEAKAGE_EXACT:
        return True
    return key.startswith(FORBIDDEN_LEAKAGE_PREFIXES)


def _run_leakage_guard(
    governance_cfg: dict[str, Any],
    panel: pd.DataFrame,
    requested_factors: list[str],
    expression_dependencies: set[str],
    combination_dependencies: set[str],
) -> dict[str, Any]:
    mode = str(governance_cfg["leakage_guard_mode"]).lower()
    referenced = sorted(
        {
            str(x).strip()
            for x in [*requested_factors, *sorted(expression_dependencies), *sorted(combination_dependencies)]
            if str(x).strip()
        }
    )
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


def _normalize_factor_scope(cfg: dict[str, Any]) -> dict[str, Any]:
    run_cfg = _as_dict(cfg.get("run"))
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


def _infer_adapter_from_path(path_like: Any) -> str:
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


def _normalize_data_cfg(cfg: dict[str, Any], scope: FactorScope) -> dict[str, Any]:
    data_cfg = _as_dict(cfg.get("data"))
    adapter_plugin_dirs = [
        str(x).strip() for x in _as_list(data_cfg.get("adapter_plugin_dirs")) if str(x).strip()
    ]
    adapter_plugins = [x for x in _as_list(data_cfg.get("adapter_plugins")) if x is not None and x != ""]
    adapter_auto_discover = _to_bool(data_cfg.get("adapter_auto_discover"), bool(adapter_plugin_dirs))
    adapter_plugin_on_error = str(data_cfg.get("adapter_plugin_on_error", "raise")).strip().lower()
    if adapter_plugin_on_error not in {"raise", "warn_skip"}:
        LOGGER.warning("Invalid data.adapter_plugin_on_error='%s'. Use 'raise'.", adapter_plugin_on_error)
        adapter_plugin_on_error = "raise"

    path_like = data_cfg.get("path")
    inferred_adapter = _infer_adapter_from_path(path_like)
    adapter_raw = str(data_cfg.get("adapter", "")).strip().lower()
    if adapter_raw in {"", "auto", "infer"}:
        adapter = inferred_adapter or "synthetic"
    else:
        adapter = "raw_dir" if adapter_raw == "raw" else adapter_raw
        if adapter == "synthetic" and inferred_adapter:
            LOGGER.info(
                "data.path/source provided with adapter=synthetic; auto-switch to inferred adapter '%s'.",
                inferred_adapter,
            )
            adapter = inferred_adapter

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
    fields_required = [str(x).strip() for x in _as_list(data_cfg.get("fields_required")) if str(x).strip()]
    if not fields_required:
        fields_required = default_fields

    return {
        "mode": mode,
        "adapter": adapter,
        "path": path_like,
        "data_dir": data_cfg.get("data_dir"),
        "raw_pattern": str(data_cfg.get("raw_pattern", "*.parquet,*.csv")).strip() or "*.parquet,*.csv",
        "raw_asset_from_filename": _to_bool(data_cfg.get("raw_asset_from_filename"), True),
        "symbols": [str(x).strip() for x in _as_list(data_cfg.get("symbols")) if str(x).strip()],
        "start_date": data_cfg.get("start_date"),
        "end_date": data_cfg.get("end_date"),
        "request_timeout_sec": max(3, _to_int(data_cfg.get("request_timeout_sec"), 20)),
        "min_rows_per_asset": max(1, _to_int(data_cfg.get("min_rows_per_asset"), 30)),
        "asset": data_cfg.get("asset"),
        "sanitize": _to_bool(data_cfg.get("sanitize"), True),
        "duplicate_policy": str(data_cfg.get("duplicate_policy", "last")).strip().lower(),
        "fields_required": fields_required,
        "synthetic": _as_dict(data_cfg.get("synthetic")),
        "adapter_auto_discover": adapter_auto_discover,
        "adapter_plugin_dirs": adapter_plugin_dirs,
        "adapter_plugins": adapter_plugins,
        "adapter_plugin_on_error": adapter_plugin_on_error,
    }


def _normalize_factor_expressions(raw: Any, strict: bool = False) -> dict[str, str]:
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


def _normalize_requested_factor_names(raw: Any) -> tuple[list[str], bool]:
    """归一化显式请求的因子名，并识别占位因子名。"""
    names: list[str] = []
    placeholder_detected = False
    seen: set[str] = set()
    for item in _as_list(raw):
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


def _discover_panel_factor_columns(panel: pd.DataFrame) -> list[str]:
    """从面板中自动发现可作为因子研究对象的列。"""
    names: list[str] = []
    for col in panel.columns:
        col_name = str(col).strip()
        if not col_name:
            continue
        if col_name in NON_FACTOR_BASE_COLUMNS:
            continue
        if _is_forbidden_leakage_name(col_name):
            continue
        if col_name.startswith("fwd_ret_"):
            continue
        names.append(col_name)
    return names


def _normalize_factor_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    fac_cfg = _as_dict(cfg.get("factor"))
    names, placeholder_detected = _normalize_requested_factor_names(fac_cfg.get("names"))
    auto_discover_from_panel = _to_bool(fac_cfg.get("auto_discover_from_panel"), True)
    on_missing = str(fac_cfg.get("on_missing", "raise")).strip().lower()
    if on_missing not in {"raise", "warn_skip"}:
        LOGGER.warning("Invalid factor.on_missing='%s'. Use 'raise'.", on_missing)
        on_missing = "raise"
    plugin_on_error = str(fac_cfg.get("plugin_on_error", "raise")).strip().lower()
    if plugin_on_error not in {"raise", "warn_skip"}:
        LOGGER.warning("Invalid factor.plugin_on_error='%s'. Use 'raise'.", plugin_on_error)
        plugin_on_error = "raise"

    plugin_dirs = [str(x).strip() for x in _as_list(fac_cfg.get("plugin_dirs")) if str(x).strip()]
    auto_discover = _to_bool(fac_cfg.get("auto_discover"), bool(plugin_dirs))
    plugins = [x for x in _as_list(fac_cfg.get("plugins")) if x is not None and x != ""]
    expression_on_error = str(fac_cfg.get("expression_on_error", "raise")).strip().lower()
    if expression_on_error not in {"raise", "warn_skip"}:
        LOGGER.warning("Invalid factor.expression_on_error='%s'. Use 'raise'.", expression_on_error)
        expression_on_error = "raise"

    expressions = _normalize_factor_expressions(fac_cfg.get("expressions"), strict=False)
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


def _normalize_custom_transforms(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for entry in _as_list(raw):
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
                LOGGER.warning(
                    "Invalid custom transform on_error='%s' for '%s'. Use 'raise'.",
                    on_error,
                    name,
                )
                on_error = "raise"
            out.append({"name": name, "kwargs": kwargs_raw, "on_error": on_error})
            continue
        LOGGER.warning("Ignore unsupported custom transform entry type: %s", type(entry))
    return out


def _normalize_research_cfg(cfg: dict[str, Any], scope: FactorScope, profile: str) -> dict[str, Any]:
    raw = _as_dict(cfg.get("research"))
    profile_key = profile if profile in {"fast", "dev", "full"} else "full"
    profile_defaults = RESEARCH_PROFILE_DEFAULTS[scope][profile_key]

    horizons_raw = _as_list(raw.get("horizons", profile_defaults["horizons"]))
    horizons = sorted(
        {
            _to_int(x, 0)
            for x in horizons_raw
            if _to_int(x, 0) > 0
        }
    )
    if not horizons:
        horizons = list(profile_defaults["horizons"])

    missing_policy = str(raw.get("missing_policy", profile_defaults["missing_policy"])).strip().lower()
    allowed_missing = {"drop", "fill_zero", "ffill_by_asset", "cs_median_by_date", "keep"}
    if missing_policy not in allowed_missing:
        LOGGER.warning("Invalid research.missing_policy='%s'. Use 'drop'.", missing_policy)
        missing_policy = "drop"

    default_steps = (
        list(profile_defaults["preprocess_steps"])
        if scope == "cs"
        else []
    )
    steps_raw = [
        str(x).strip().lower()
        for x in _as_list(raw.get("preprocess_steps", default_steps))
        if str(x).strip()
    ]
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
    for x in _as_list(raw.get("ts_signal_lags", profile_defaults.get("ts_signal_lags"))):
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

    transform_plugin_dirs = [
        str(x).strip() for x in _as_list(raw.get("transform_plugin_dirs")) if str(x).strip()
    ]
    transform_plugins = [x for x in _as_list(raw.get("transform_plugins")) if x is not None and x != ""]
    transform_auto_discover = _to_bool(
        raw.get("transform_auto_discover"),
        bool(transform_plugin_dirs),
    )
    transform_plugin_on_error = str(raw.get("transform_plugin_on_error", "raise")).strip().lower()
    if transform_plugin_on_error not in {"raise", "warn_skip"}:
        LOGGER.warning(
            "Invalid research.transform_plugin_on_error='%s'. Use 'raise'.",
            transform_plugin_on_error,
        )
        transform_plugin_on_error = "raise"

    out = {
        "profile": profile_key,
        "horizons": horizons,
        "quantiles": max(2, _to_int(raw.get("quantiles"), int(profile_defaults["quantiles"]))),
        "ic_rolling_window": max(5, _to_int(raw.get("ic_rolling_window"), int(profile_defaults["ic_rolling_window"]))),
        "annualization_days": max(1, _to_int(raw.get("annualization_days"), int(profile_defaults["annualization_days"]))),
        "ts_standardize_window": max(
            5,
            _to_int(raw.get("ts_standardize_window"), int(profile_defaults.get("ts_standardize_window", 60))),
        ),
        "ts_quantile_lookback": max(
            10,
            _to_int(raw.get("ts_quantile_lookback"), int(profile_defaults.get("ts_quantile_lookback", 60))),
        ),
        "ts_signal_lags": ts_signal_lags,
        "missing_policy": missing_policy,
        "preprocess_steps": preprocess_steps,
        "transform_auto_discover": transform_auto_discover,
        "transform_plugin_dirs": transform_plugin_dirs,
        "transform_plugins": transform_plugins,
        "transform_plugin_on_error": transform_plugin_on_error,
        "custom_transforms": _normalize_custom_transforms(raw.get("custom_transforms")),
    }
    if scope == "cs":
        out["winsorize"] = _as_dict(raw.get("winsorize"))
        out["neutralize"] = _as_dict(raw.get("neutralize"))
    return out


def _normalize_backtest_cfg(cfg: dict[str, Any], scope: FactorScope) -> dict[str, Any]:
    raw = _as_dict(cfg.get("backtest"))
    enabled = _to_bool(raw.get("enabled"), False)
    strategy_cfg = _as_dict(raw.get("strategy"))
    default_mode = "sign" if scope == "ts" else "longshort"
    strategy_mode = str(strategy_cfg.get("mode", default_mode)).strip().lower() or default_mode
    builtin_modes = {"sign", "topk", "longshort", "flex", "meanvar"}
    if strategy_mode not in builtin_modes:
        LOGGER.info(
            "Using custom strategy mode '%s'. Will resolve via strategy plugin registry.",
            strategy_mode,
        )

    strategy_plugin_on_error = str(strategy_cfg.get("plugin_on_error", "raise")).strip().lower()
    if strategy_plugin_on_error not in {"raise", "warn_skip"}:
        LOGGER.warning(
            "Invalid backtest.strategy.plugin_on_error='%s'. Use 'raise'.",
            strategy_plugin_on_error,
        )
        strategy_plugin_on_error = "raise"

    strategy_plugin_dirs = [str(x).strip() for x in _as_list(strategy_cfg.get("plugin_dirs")) if str(x).strip()]
    strategy_auto_discover = _to_bool(strategy_cfg.get("auto_discover"), bool(strategy_plugin_dirs))
    strategy_plugins = [x for x in _as_list(strategy_cfg.get("plugins")) if x is not None and x != ""]

    return {
        "enabled": enabled,
        "strategy_mode": strategy_mode,
        "top_k": max(1, _to_int(strategy_cfg.get("top_k"), 20)),
        "long_short_quantile": max(0.05, min(0.49, _to_float(strategy_cfg.get("long_short_quantile"), 0.2))),
        "long_fraction": max(0.05, min(0.95, _to_float(strategy_cfg.get("long_fraction"), 0.2))),
        "short_fraction": max(0.0, min(0.95, _to_float(strategy_cfg.get("short_fraction"), 0.2))),
        "long_only": _to_bool(strategy_cfg.get("long_only"), False),
        "rebalance_every": max(1, _to_int(strategy_cfg.get("rebalance_every"), 1)),
        "weight_scheme": str(strategy_cfg.get("weight_scheme", "equal")).strip().lower(),
        "max_weight": strategy_cfg.get("max_weight"),
        "risk_aversion": max(1e-6, _to_float(strategy_cfg.get("risk_aversion"), 5.0)),
        "gross_target": max(0.1, _to_float(strategy_cfg.get("gross_target"), 1.0)),
        "net_target": _to_float(strategy_cfg.get("net_target"), 0.0),
        "sign_threshold": max(0.0, _to_float(strategy_cfg.get("sign_threshold"), 0.0)),
        "commission_bps": _to_float(raw.get("commission_bps"), 3.0),
        "slippage_bps": _to_float(raw.get("slippage_bps"), 2.0),
        "leverage": max(0.1, _to_float(raw.get("leverage"), 1.0)),
        "execution_delay_days": max(0, _to_int(raw.get("execution_delay_days"), 1)),
        "execution_price_col": str(raw.get("execution_price_col", "close")).strip(),
        "max_turnover": raw.get("max_turnover"),
        "max_abs_weight": raw.get("max_abs_weight"),
        "max_gross_exposure": raw.get("max_gross_exposure"),
        "max_net_exposure": raw.get("max_net_exposure"),
        "enforce_industry_neutral": _to_bool(raw.get("enforce_industry_neutral"), False),
        "industry_col": str(raw.get("industry_col", "industry")).strip(),
        "benchmark_mode": str(raw.get("benchmark_mode", "none")).strip().lower(),
        "benchmark_return_col": str(raw.get("benchmark_return_col", "benchmark_ret")).strip(),
        "strategy_plugin_dirs": strategy_plugin_dirs,
        "strategy_plugins": strategy_plugins,
        "strategy_auto_discover": strategy_auto_discover,
        "strategy_plugin_on_error": strategy_plugin_on_error,
    }


def _normalize_universe_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = _as_dict(cfg.get("universe_filter"))
    return {
        "enabled": _to_bool(raw.get("enabled"), False),
        "config": UniverseFilterConfig(
            min_close=_to_float(raw.get("min_close"), 0.0),
            min_history_days=max(1, _to_int(raw.get("min_history_days"), 1)),
            min_median_dollar_volume=max(0.0, _to_float(raw.get("min_median_dollar_volume"), 0.0)),
            liquidity_lookback=max(2, _to_int(raw.get("liquidity_lookback"), 20)),
        ),
    }


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
    """从文件加载 YAML 运行配置，支持 `imports`/`extends` 分层。"""
    return _load_run_config_with_imports(Path(path))


def deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """递归合并字典并返回新对象。"""
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
        raise ValueError(
            f"Invalid override '{override}'. Expected format: key.path=value (or += / -=)."
        )
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
                raise ValueError(
                    f"Invalid override '{override}': '+=' on dict target requires object value."
                )
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
                    raise ValueError(
                        f"Invalid override '{override}': '-=' dict target requires string key(s)."
                    )
                out.pop(target, None)
            return out

        raise ValueError(f"Invalid override '{override}': '-=' only supports list/dict targets.")

    raise ValueError(f"Unsupported override operator in '{override}': {op}")


def apply_config_override(cfg: dict[str, Any], override: str) -> dict[str, Any]:
    """应用单条覆盖项并返回新配置。

    支持：
    - `a.b=value` 替换
    - `a.list+=value` 列表追加（value 为列表时扩展）
    - `a.list-=value` 列表删除（value 为列表时批量删除）
    - `a.dict+= {...}` 字典深度合并
    - `a.dict-= key` 或 `a.dict-=[k1,k2]` 字典删键
    """
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
    """由多份 YAML 与覆盖项合成有效运行配置，并返回别名迁移记录。"""
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
    """由多份 YAML 与覆盖项合成有效运行配置。"""
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
        return


def validate_run_config_schema(cfg: dict[str, Any], strict: bool = True) -> list[str]:
    """运行前进行配置结构校验，并返回非阻断告警。"""
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

    root_allowed = {"run", "data", "factor", "research", "backtest", "universe_filter", "imports", "extends"}
    root_required = {"data"}

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
    default_eval_axis = "time" if scope == "ts" else "cross_section"
    eval_axis = str(run_cfg.get("eval_axis", default_eval_axis)).strip().lower()
    default_std = "ts_rolling_zscore" if scope == "ts" else "cs_zscore"
    standardization = str(run_cfg.get("standardization", default_std)).strip().lower()
    if scope not in {"cs", "ts"}:
        errors.append("run.factor_scope: must be one of ['cs', 'ts'].")
    if eval_axis not in {"cross_section", "time"}:
        errors.append("run.eval_axis: must be one of ['cross_section', 'time'].")
    config_mode = str(run_cfg.get("config_mode", "compat")).strip().lower()
    if config_mode not in {"strict", "warn", "compat"}:
        errors.append("run.config_mode: must be one of ['strict', 'warn', 'compat'].")
    leakage_guard_mode = str(run_cfg.get("leakage_guard_mode", "strict")).strip().lower()
    if leakage_guard_mode not in {"strict", "warn", "off"}:
        errors.append("run.leakage_guard_mode: must be one of ['strict', 'warn', 'off'].")
    stop_after = str(run_cfg.get("stop_after", "backtest")).strip().lower()
    if stop_after not in {"factor", "research", "backtest"}:
        errors.append("run.stop_after: must be one of ['factor', 'research', 'backtest'].")
    research_profile = str(run_cfg.get("research_profile", "full")).strip().lower()
    if research_profile not in {"fast", "dev", "full"}:
        errors.append("run.research_profile: must be one of ['fast', 'dev', 'full'].")
    if "fail_on_autocorrect" in run_cfg and not isinstance(run_cfg.get("fail_on_autocorrect"), bool):
        errors.append("run.fail_on_autocorrect: must be boolean when provided.")

    cs_std = {"cs_zscore", "cs_rank", "cs_robust_zscore", "none"}
    ts_std = {"ts_rolling_zscore", "zscore", "none"}
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
    mode_default = "single_asset" if scope == "ts" else "panel"
    mode = str(data_cfg.get("mode", mode_default)).strip().lower()
    builtin_adapters = {"synthetic", "sina", "stooq", "parquet", "csv", "raw_dir"}
    adapter_plugin_dirs = _as_list(data_cfg.get("adapter_plugin_dirs"))
    adapter_plugins = _as_list(data_cfg.get("adapter_plugins"))
    has_adapter_plugins = bool(adapter_plugin_dirs) or bool(adapter_plugins)
    if adapter not in builtin_adapters and not has_adapter_plugins:
        errors.append(
            "data.adapter: unknown adapter without data adapter plugins. "
            "Use built-in ['synthetic', 'sina', 'stooq', 'parquet', 'csv', 'raw_dir'] "
            "or configure data.adapter_plugin_dirs/data.adapter_plugins."
        )
    if mode not in {"single_asset", "panel"}:
        errors.append("data.mode: must be one of ['single_asset', 'panel'].")
    if adapter in {"parquet", "csv"} and not data_cfg.get("path"):
        errors.append("data.path: required when data.adapter is parquet/csv.")
    if adapter == "raw_dir" and not data_cfg.get("path"):
        errors.append("data.path: required when data.adapter is raw_dir.")
    if adapter == "sina" and not data_cfg.get("data_dir"):
        errors.append("data.data_dir: required when data.adapter is sina.")
    if adapter == "stooq" and not _as_list(data_cfg.get("symbols")):
        errors.append("data.symbols: required when data.adapter is stooq.")
    adapter_plugin_on_error = str(data_cfg.get("adapter_plugin_on_error", "raise")).strip().lower()
    if adapter_plugin_on_error not in {"raise", "warn_skip"}:
        errors.append("data.adapter_plugin_on_error: must be one of ['raise', 'warn_skip'].")
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
    if on_missing not in {"raise", "warn_skip"}:
        errors.append("factor.on_missing: must be one of ['raise', 'warn_skip'].")
    plugin_on_error = str(factor_cfg.get("plugin_on_error", "raise")).strip().lower()
    if plugin_on_error not in {"raise", "warn_skip"}:
        errors.append("factor.plugin_on_error: must be one of ['raise', 'warn_skip'].")
    expression_on_error = str(factor_cfg.get("expression_on_error", "raise")).strip().lower()
    if expression_on_error not in {"raise", "warn_skip"}:
        errors.append("factor.expression_on_error: must be one of ['raise', 'warn_skip'].")
    combination_on_error = str(factor_cfg.get("combination_on_error", "raise")).strip().lower()
    if combination_on_error not in {"raise", "warn_skip"}:
        errors.append("factor.combination_on_error: must be one of ['raise', 'warn_skip'].")

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
    allowed_missing = {"drop", "fill_zero", "ffill_by_asset", "cs_median_by_date", "keep"}
    if missing_policy not in allowed_missing:
        errors.append(f"research.missing_policy: must be one of {sorted(allowed_missing)}.")

    step_values = [str(x).strip().lower() for x in _as_list(research_cfg.get("preprocess_steps")) if str(x).strip()]
    allowed_steps = {"winsorize", "standardize", "neutralize"}
    for step in step_values:
        if step not in allowed_steps:
            errors.append(f"research.preprocess_steps: unsupported step '{step}'.")
            break

    transform_plugin_on_error = str(research_cfg.get("transform_plugin_on_error", "raise")).strip().lower()
    if transform_plugin_on_error not in {"raise", "warn_skip"}:
        errors.append("research.transform_plugin_on_error: must be one of ['raise', 'warn_skip'].")

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
    builtin_modes = {"sign", "topk", "longshort", "flex", "meanvar"}
    if mode_val and mode_val not in builtin_modes:
        has_plugins = bool(_as_list(strategy_cfg.get("plugins"))) or bool(_as_list(strategy_cfg.get("plugin_dirs")))
        if not has_plugins:
            errors.append(
                "backtest.strategy.mode: unknown custom mode without strategy plugins. "
                "Provide backtest.strategy.plugins or backtest.strategy.plugin_dirs."
            )

    strategy_plugin_on_error = str(strategy_cfg.get("plugin_on_error", "raise")).strip().lower()
    if strategy_plugin_on_error not in {"raise", "warn_skip"}:
        errors.append("backtest.strategy.plugin_on_error: must be one of ['raise', 'warn_skip'].")

    bench_mode = str(back_cfg.get("benchmark_mode", "none")).strip().lower()
    if bench_mode not in {"none", "cross_sectional_mean", "panel_column"}:
        errors.append("backtest.benchmark_mode: must be one of ['none', 'cross_sectional_mean', 'panel_column'].")

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


def _build_adapter_config(data_cfg: dict[str, Any]) -> AdapterConfig:
    return AdapterConfig(
        data_dir=str(data_cfg.get("data_dir") or ""),
        required_cols=tuple(str(c).strip() for c in data_cfg.get("fields_required", []) if str(c).strip())
        or ("date", "close"),
        min_rows_per_asset=int(data_cfg.get("min_rows_per_asset", 30)),
        symbols=tuple(str(x).strip() for x in data_cfg.get("symbols", []) if str(x).strip()),
        start_date=str(data_cfg.get("start_date")) if data_cfg.get("start_date") else None,
        end_date=str(data_cfg.get("end_date")) if data_cfg.get("end_date") else None,
        request_timeout_sec=int(data_cfg.get("request_timeout_sec", 20)),
    )


def _normalize_adapter_validation_warnings(result: Any) -> list[str]:
    if result is None:
        return []
    if isinstance(result, bool):
        if result:
            return []
        raise ValueError("Adapter config validator returned False.")
    if isinstance(result, str):
        text = result.strip()
        return [text] if text else []
    if isinstance(result, dict):
        raw = result.get("warnings")
        if raw is None:
            return []
        return [str(x).strip() for x in _as_list(raw) if str(x).strip()]
    if isinstance(result, (list, tuple, set)):
        return [str(x).strip() for x in result if str(x).strip()]
    raise TypeError(f"Unsupported validator return type: {type(result)}")


def _validate_adapter_config(
    adapter: str,
    adapter_cfg: AdapterConfig,
    validator_registry: dict[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "adapter": adapter,
        "validated": False,
        "validator_found": False,
        "validation_seconds": 0.0,
        "warnings": [],
    }
    validator = validator_registry.get(adapter)
    if validator is None:
        return report

    report["validator_found"] = True
    t0 = time.perf_counter()
    result = validator(adapter_cfg)
    report["validation_seconds"] = float(time.perf_counter() - t0)
    report["warnings"] = _normalize_adapter_validation_warnings(result)
    report["validated"] = True
    return report


def _write_adapter_quality_audit_tables(
    panel: pd.DataFrame,
    data_cfg: dict[str, Any],
    load_report: dict[str, Any],
    out_dir: Path,
) -> dict[str, str]:
    table_dir = out_dir / "tables" / "data"
    table_dir.mkdir(parents=True, exist_ok=True)

    df = panel.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    rows = int(len(df))
    assets = int(df["asset"].nunique()) if ("asset" in df.columns and rows > 0) else 0
    dates = int(df["date"].nunique()) if ("date" in df.columns and rows > 0) else 0
    min_rows_threshold = int(data_cfg.get("min_rows_per_asset", 0))

    summary_rows: list[dict[str, Any]] = [
        {"category": "source", "metric": "adapter", "value": str(data_cfg.get("adapter", "")), "note": ""},
        {"category": "shape", "metric": "rows", "value": rows, "note": ""},
        {"category": "shape", "metric": "assets", "value": assets, "note": ""},
        {"category": "shape", "metric": "dates", "value": dates, "note": ""},
        {
            "category": "threshold",
            "metric": "min_rows_per_asset",
            "value": min_rows_threshold,
            "note": "用于资产样本量达标判定",
        },
        {
            "category": "timing",
            "metric": "adapter_load_seconds",
            "value": float(load_report.get("adapter_load_seconds", 0.0)),
            "note": "",
        },
    ]

    field_rows: list[dict[str, Any]] = []
    for col in [str(x).strip() for x in data_cfg.get("fields_required", []) if str(x).strip()]:
        if col not in df.columns:
            missing_rate = 1.0
            coverage_rate = 0.0
        else:
            missing_rate = float(df[col].isna().mean()) if rows > 0 else 1.0
            coverage_rate = float(1.0 - missing_rate)
        field_rows.append(
            {
                "field": col,
                "missing_rate": missing_rate,
                "coverage_rate": coverage_rate,
            }
        )
        summary_rows.append(
            {
                "category": "missing",
                "metric": f"missing_rate__{col}",
                "value": missing_rate,
                "note": "",
            }
        )

    asset_rows = pd.DataFrame(columns=["asset", "rows", "meets_min_rows"])
    if "asset" in df.columns and rows > 0:
        asset_counts = (
            df.groupby("asset", as_index=False)
            .size()
            .rename(columns={"size": "rows"})
            .sort_values("rows", ascending=False)
            .reset_index(drop=True)
        )
        asset_counts["meets_min_rows"] = (
            asset_counts["rows"] >= min_rows_threshold if min_rows_threshold > 0 else True
        )
        asset_rows = asset_counts

        summary_rows.extend(
            [
                {
                    "category": "asset_rows",
                    "metric": "asset_rows_min",
                    "value": int(asset_counts["rows"].min()),
                    "note": "",
                },
                {
                    "category": "asset_rows",
                    "metric": "asset_rows_median",
                    "value": float(asset_counts["rows"].median()),
                    "note": "",
                },
                {
                    "category": "asset_rows",
                    "metric": "asset_rows_max",
                    "value": int(asset_counts["rows"].max()),
                    "note": "",
                },
                {
                    "category": "threshold",
                    "metric": "assets_meet_min_rows",
                    "value": int(asset_counts["meets_min_rows"].sum()),
                    "note": "",
                },
                {
                    "category": "threshold",
                    "metric": "assets_below_min_rows",
                    "value": int((~asset_counts["meets_min_rows"]).sum()),
                    "note": "",
                },
                {
                    "category": "threshold",
                    "metric": "assets_meet_min_rows_rate",
                    "value": float(asset_counts["meets_min_rows"].mean()),
                    "note": "",
                },
            ]
        )

    date_cov_rows = pd.DataFrame(columns=["date", "assets_covered", "coverage_rate"])
    if {"date", "asset"}.issubset(df.columns) and rows > 0 and assets > 0:
        date_cov = (
            df.groupby("date", as_index=False)["asset"]
            .nunique()
            .rename(columns={"asset": "assets_covered"})
            .sort_values("date")
            .reset_index(drop=True)
        )
        date_cov["coverage_rate"] = date_cov["assets_covered"] / max(assets, 1)
        date_cov_rows = date_cov
        summary_rows.extend(
            [
                {
                    "category": "coverage",
                    "metric": "date_coverage_mean",
                    "value": float(date_cov["coverage_rate"].mean()),
                    "note": "按交易日统计的资产覆盖率均值",
                },
                {
                    "category": "coverage",
                    "metric": "date_coverage_min",
                    "value": float(date_cov["coverage_rate"].min()),
                    "note": "",
                },
                {
                    "category": "coverage",
                    "metric": "date_coverage_p10",
                    "value": float(date_cov["coverage_rate"].quantile(0.1)),
                    "note": "",
                },
                {
                    "category": "coverage",
                    "metric": "date_coverage_p50",
                    "value": float(date_cov["coverage_rate"].quantile(0.5)),
                    "note": "",
                },
                {
                    "category": "coverage",
                    "metric": "date_coverage_p90",
                    "value": float(date_cov["coverage_rate"].quantile(0.9)),
                    "note": "",
                },
            ]
        )

    summary_path = table_dir / "adapter_quality_audit.csv"
    fields_path = table_dir / "field_missing_rates.csv"
    asset_path = table_dir / "asset_row_counts.csv"
    coverage_path = table_dir / "date_coverage.csv"

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(field_rows).to_csv(fields_path, index=False)
    asset_rows.to_csv(asset_path, index=False)
    date_cov_rows.to_csv(coverage_path, index=False)

    return {
        "adapter_quality_audit_csv": str(summary_path),
        "field_missing_rates_csv": str(fields_path),
        "asset_row_counts_csv": str(asset_path),
        "date_coverage_csv": str(coverage_path),
    }


def _load_data(
    data_cfg: dict[str, Any],
    scope_cfg: dict[str, Any],
    adapter_registry: dict[str, Any],
    adapter_cfg: AdapterConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    adapter = data_cfg["adapter"]
    sanitize = bool(data_cfg["sanitize"])
    duplicate_policy = str(data_cfg["duplicate_policy"])
    loader_report: dict[str, Any] = {"adapter": adapter, "sanitize": sanitize}

    def _attach_panel_profile(df: pd.DataFrame, source: str) -> None:
        if df.empty:
            loader_report["panel_profile"] = {
                "source": source,
                "rows": 0,
                "assets": 0,
                "dates": 0,
                "date_min": None,
                "date_max": None,
                "columns": int(len(df.columns)),
            }
            return
        d = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else pd.Series(dtype="datetime64[ns]")
        loader_report["panel_profile"] = {
            "source": source,
            "rows": int(len(df)),
            "assets": int(df["asset"].nunique()) if "asset" in df.columns else 0,
            "dates": int(d.nunique()) if not d.empty else 0,
            "date_min": str(d.min()) if not d.empty else None,
            "date_max": str(d.max()) if not d.empty else None,
            "columns": int(len(df.columns)),
        }

    if adapter == "synthetic":
        s = data_cfg["synthetic"]
        n_assets_default = 1 if data_cfg["mode"] == "single_asset" else 40
        syn_cfg = SyntheticConfig(
            n_assets=max(1, _to_int(s.get("n_assets"), n_assets_default)),
            n_days=max(60, _to_int(s.get("n_days"), 260)),
            seed=_to_int(s.get("seed"), 7),
            start_date=str(s.get("start_date", "2021-01-01")),
        )
        t0 = time.perf_counter()
        panel = generate_synthetic_panel(syn_cfg)
        loader_report["adapter_load_seconds"] = float(time.perf_counter() - t0)
        loader_report["synthetic"] = syn_cfg.__dict__ if hasattr(syn_cfg, "__dict__") else {
            "n_assets": syn_cfg.n_assets,
            "n_days": syn_cfg.n_days,
            "seed": syn_cfg.seed,
            "start_date": syn_cfg.start_date,
        }
        _attach_panel_profile(panel, source="synthetic")
        return panel, loader_report

    if adapter in {"parquet", "csv"}:
        path = data_cfg.get("path")
        if not path:
            raise ValueError("data.path is required when data.adapter is parquet/csv")
        if Path(str(path)).is_dir():
            LOGGER.info("data.path points to directory; auto-switch to raw_dir loader.")
            adapter = "raw_dir"
        else:
            t0 = time.perf_counter()
            read_res = read_panel(
                path=str(path),
                sanitize=sanitize,
                sanitization_config=PanelSanitizationConfig(duplicate_policy=duplicate_policy),
                return_report=sanitize,
            )
            loader_report["adapter_load_seconds"] = float(time.perf_counter() - t0)
            if sanitize:
                panel, report = read_res
                loader_report["sanitization_report"] = report.__dict__ if hasattr(report, "__dict__") else str(report)
            else:
                panel = read_res
            _attach_panel_profile(panel, source="file_io")
            return panel, loader_report

    if adapter == "raw_dir":
        path = data_cfg.get("path")
        if not path:
            raise ValueError("data.path is required when data.adapter is raw_dir")
        t0 = time.perf_counter()
        read_res = read_panel_directory(
            directory=str(path),
            pattern=str(data_cfg.get("raw_pattern", "*.parquet,*.csv")),
            sanitize=sanitize,
            sanitization_config=PanelSanitizationConfig(duplicate_policy=duplicate_policy),
            return_report=True,
            asset_from_filename=bool(data_cfg.get("raw_asset_from_filename", True)),
        )
        panel, dir_report = read_res
        loader_report["adapter_load_seconds"] = float(time.perf_counter() - t0)
        loader_report["directory_report"] = asdict(dir_report)
        _attach_panel_profile(panel, source="raw_dir")
        return panel, loader_report

    if adapter not in adapter_registry:
        raise KeyError(
            f"Unknown data adapter '{adapter}'. Available adapters: {sorted(adapter_registry.keys())}"
        )

    adapter_fn = adapter_registry[adapter]
    adapter_cfg = adapter_cfg or _build_adapter_config(data_cfg)
    t0 = time.perf_counter()
    panel = adapter_fn(adapter_cfg)
    loader_report["adapter_load_seconds"] = float(time.perf_counter() - t0)
    loader_report["adapter_registry_size"] = int(len(adapter_registry))
    loader_report["symbols"] = list(adapter_cfg.symbols)
    loader_report["start_date"] = adapter_cfg.start_date
    loader_report["end_date"] = adapter_cfg.end_date
    _attach_panel_profile(panel, source="adapter")
    return panel, loader_report


def _ensure_mode_shape(panel: pd.DataFrame, data_cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = panel.copy()
    mode = data_cfg["mode"]
    report: dict[str, Any] = {"mode": mode}

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "asset" not in out.columns:
        asset_name = str(data_cfg.get("asset") or "SINGLE_ASSET")
        out["asset"] = asset_name

    out["asset"] = out["asset"].astype(str)
    out = out.dropna(subset=["date"]).sort_values(["date", "asset"]).reset_index(drop=True)
    if mode == "single_asset":
        candidate = data_cfg.get("asset")
        if candidate is None:
            candidate = out["asset"].astype(str).drop_duplicates().iloc[0]
            LOGGER.info("data.mode=single_asset and no asset provided; using first asset: %s", candidate)
        candidate = str(candidate)
        out = out[out["asset"] == candidate].copy()
        report["selected_asset"] = candidate
    report["rows_after_mode"] = int(len(out))
    report["assets_after_mode"] = int(out["asset"].nunique()) if not out.empty else 0
    return out.reset_index(drop=True), report


def _resolve_required_fields(
    scope_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    factor_names: list[str],
    research_cfg: dict[str, Any],
) -> list[str]:
    required = set(str(x).strip() for x in data_cfg["fields_required"] if str(x).strip())
    required.update({"date", "close"})
    if scope_cfg["factor_scope"] == "cs":
        required.add("asset")
    if "size" in factor_names:
        required.add("mkt_cap")

    if scope_cfg["factor_scope"] == "cs":
        neutral = _as_dict(research_cfg.get("neutralize"))
        steps = {str(x).strip().lower() for x in _as_list(research_cfg.get("preprocess_steps")) if str(x).strip()}
        neutralize_in_pipeline = ("neutralize" in steps) if steps else True
        if _to_bool(neutral.get("enabled"), True) and neutralize_in_pipeline:
            mode = str(neutral.get("mode", "both")).strip().lower()
            if mode in {"size", "both"}:
                required.add("mkt_cap")
            if mode in {"industry", "both"}:
                required.add("industry")
    return sorted(required)


def _filter_factors_by_available_columns(
    panel: pd.DataFrame,
    factor_names: list[str],
    on_missing: str,
) -> tuple[list[str], list[str]]:
    if not factor_names:
        return [], []
    if on_missing != "warn_skip":
        return factor_names, []

    selected: list[str] = []
    skipped: list[str] = []
    cols = set(panel.columns)
    for name in factor_names:
        required = FACTOR_REQUIRED_COLUMNS.get(name, set())
        missing = sorted(required - cols)
        if missing:
            LOGGER.warning(
                "Skip factor '%s': required input columns missing=%s and factor.on_missing=warn_skip",
                name,
                missing,
            )
            skipped.append(name)
            continue
        selected.append(name)
    return selected, skipped


def _validate_required_fields(panel: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise KeyError(f"Data missing required fields: {missing}")


def _compute_factors(
    panel: pd.DataFrame,
    factor_names: list[str],
    on_missing: str,
    registry: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = panel.copy()
    if not factor_names:
        return out, [], []
    missing = [f for f in factor_names if f not in out.columns]
    computable = [f for f in missing if f in registry]
    if computable:
        out = apply_factors(out, computable, inplace=True, registry=registry)
    unresolved = [f for f in factor_names if f not in out.columns]
    if unresolved:
        if on_missing == "warn_skip":
            LOGGER.warning("Skip unresolved factors due to factor.on_missing=warn_skip: %s", unresolved)
        else:
            raise KeyError(f"Factors missing and not computable: {unresolved}")
    selected = [f for f in factor_names if f not in unresolved]
    return out, computable, selected


def _call_transform_fn(fn: Any, panel: pd.DataFrame, factor_col: str, kwargs: dict[str, Any]) -> pd.Series:
    try:
        value = fn(panel=panel, factor_col=factor_col, **kwargs)
    except TypeError:
        value = fn(panel, factor_col, **kwargs)

    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise TypeError(
                f"Transform output for '{factor_col}' must be a Series or single-column DataFrame, got shape={value.shape}."
            )
        value = value.iloc[:, 0]

    if not isinstance(value, pd.Series):
        if isinstance(value, np.ndarray):
            value = pd.Series(value, index=panel.index, dtype=float)
        else:
            raise TypeError(f"Transform output for '{factor_col}' must be a pandas Series, got {type(value)}.")

    if len(value) != len(panel):
        raise ValueError(
            f"Transform output length mismatch for '{factor_col}': expected {len(panel)} rows, got {len(value)}."
        )
    return value.reindex(panel.index)


def _apply_custom_transforms(
    panel: pd.DataFrame,
    factor_names: list[str],
    transform_specs: list[dict[str, Any]],
    transform_registry: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = panel.copy()
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for fac in factor_names:
        if fac not in out.columns:
            errors.append(
                {
                    "factor": fac,
                    "transform": None,
                    "error": "factor column missing on panel before transform stage",
                }
            )
            continue

        for spec in transform_specs:
            name = str(spec.get("name", "")).strip()
            kwargs = dict(spec.get("kwargs", {}) or {})
            on_error = str(spec.get("on_error", "raise")).strip().lower()
            fn = transform_registry.get(name)
            if fn is None:
                msg = f"custom transform '{name}' is not registered"
                if on_error == "warn_skip":
                    LOGGER.warning("Skip transform for factor '%s': %s", fac, msg)
                    skipped.append({"factor": fac, "transform": name, "reason": msg})
                    continue
                raise KeyError(msg)
            try:
                out[fac] = _call_transform_fn(fn=fn, panel=out, factor_col=fac, kwargs=kwargs)
                applied.append({"factor": fac, "transform": name, "kwargs": kwargs})
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                if on_error == "warn_skip":
                    LOGGER.warning(
                        "Skip transform '%s' for factor '%s' due to on_error=warn_skip: %s",
                        name,
                        fac,
                        msg,
                    )
                    skipped.append({"factor": fac, "transform": name, "reason": msg})
                    continue
                raise

    return out, {"applied": applied, "skipped": skipped, "errors": errors}


def _build_strategy(back_cfg: dict[str, Any], strategy_registry: dict[str, Any]) -> Strategy:
    mode = back_cfg["strategy_mode"]
    if mode == "topk":
        return TopKLongStrategy(
            name="topk_long",
            top_k=int(back_cfg["top_k"]),
            rebalance_every=int(back_cfg["rebalance_every"]),
            weight_scheme=back_cfg["weight_scheme"],
            max_weight=back_cfg["max_weight"],
        )
    if mode == "longshort":
        return LongShortQuantileStrategy(
            name="long_short_quantile",
            quantile=float(back_cfg["long_short_quantile"]),
            rebalance_every=int(back_cfg["rebalance_every"]),
            weight_scheme=back_cfg["weight_scheme"],
            max_weight=back_cfg["max_weight"],
        )
    if mode == "flex":
        return FlexibleLongShortStrategy(
            name="flexible_long_short",
            long_fraction=float(back_cfg["long_fraction"]),
            short_fraction=float(back_cfg["short_fraction"]),
            long_only=bool(back_cfg["long_only"]),
            rebalance_every=int(back_cfg["rebalance_every"]),
            weight_scheme=back_cfg["weight_scheme"],
            max_weight=back_cfg["max_weight"],
        )
    if mode == "meanvar":
        return MeanVarianceOptimizerStrategy(
            name="mean_variance_optimizer",
            risk_aversion=float(back_cfg["risk_aversion"]),
            long_only=bool(back_cfg["long_only"]),
            gross_target=float(back_cfg["gross_target"]),
            net_target=float(back_cfg["net_target"]),
            rebalance_every=int(back_cfg["rebalance_every"]),
            max_weight=back_cfg["max_weight"],
        )
    if mode not in strategy_registry:
        raise KeyError(
            f"Unknown strategy mode '{mode}'. Available strategy plugins: {sorted(strategy_registry.keys())}"
        )
    obj = strategy_registry[mode]()
    if not isinstance(obj, Strategy):
        raise TypeError(f"Registry constructor for '{mode}' did not return Strategy instance.")
    return obj


def _build_sign_weights(score_df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    tmp = score_df[["date", "asset", "score"]].copy()
    tmp["score"] = pd.to_numeric(tmp["score"], errors="coerce")
    w = np.sign(tmp["score"]).astype(float)
    if threshold > 0:
        w[np.abs(tmp["score"]) <= float(threshold)] = 0.0
    out = tmp[["date", "asset"]].copy()
    out["weight"] = w
    return out


def _build_unique_slug_map(names: list[str], default: str = "item") -> dict[str, str]:
    """为任意名称列表生成去重后的安全 slug 映射。"""
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for raw in names:
        base = safe_slug(raw, default=default)
        candidate = base
        seq = 2
        while candidate in used:
            candidate = f"{base}_{seq}"
            seq += 1
        used.add(candidate)
        mapping[raw] = candidate
    return mapping


def _run_optional_backtest(
    panel: pd.DataFrame,
    factors: list[str],
    scope_cfg: dict[str, Any],
    back_cfg: dict[str, Any],
    out_dir: Path,
) -> Path | None:
    if not back_cfg["enabled"]:
        return None
    bt_dir = out_dir / "backtest"
    bt_dir.mkdir(parents=True, exist_ok=True)

    bt_cfg = BacktestConfig(
        cost=CostConfig(
            commission_bps=float(back_cfg["commission_bps"]),
            slippage_bps=float(back_cfg["slippage_bps"]),
            annualization_days=252,
        ),
        long_short_leverage=float(back_cfg["leverage"]),
        execution_delay_days=int(back_cfg["execution_delay_days"]),
        execution_price_col=back_cfg["execution_price_col"],
        max_turnover=_to_optional_float(back_cfg["max_turnover"]),
        max_abs_weight=_to_optional_float(back_cfg["max_abs_weight"]),
        max_gross_exposure=_to_optional_float(back_cfg["max_gross_exposure"]),
        max_net_exposure=_to_optional_float(back_cfg["max_net_exposure"]),
        enforce_industry_neutral=bool(back_cfg["enforce_industry_neutral"]),
        industry_col=back_cfg["industry_col"],
        benchmark_mode=back_cfg["benchmark_mode"],  # type: ignore[arg-type]
        benchmark_return_col=back_cfg["benchmark_return_col"],
    )

    strategy_registry = build_strategy_registry(
        plugin_dirs=back_cfg["strategy_plugin_dirs"] if back_cfg["strategy_auto_discover"] else [],
        plugin_specs=back_cfg["strategy_plugins"],
        on_plugin_error=back_cfg["strategy_plugin_on_error"],
        include_defaults=False,
    )
    strategy = None
    if back_cfg["strategy_mode"] != "sign":
        strategy = _build_strategy(back_cfg, strategy_registry=strategy_registry)

    factor_slug_map = _build_unique_slug_map(factors, default="factor")
    rows: list[dict[str, Any]] = []
    for fac in factors:
        score_df = panel[["date", "asset", fac]].rename(columns={fac: "score"})
        if back_cfg["strategy_mode"] == "sign":
            weights = _build_sign_weights(score_df, threshold=float(back_cfg["sign_threshold"]))
        else:
            assert strategy is not None
            weights = strategy.generate_weights(score_df)

        res = run_backtest(panel=panel, weights=weights, config=bt_cfg)
        fac_dir = bt_dir / factor_slug_map[fac]
        fac_dir.mkdir(parents=True, exist_ok=True)
        weights.to_csv(fac_dir / "weights.csv", index=False)
        res.daily.to_csv(fac_dir / "daily.csv", index=False)
        res.metrics.to_csv(fac_dir / "metrics.csv", index=False)

        row = {"factor": fac, "scope": scope_cfg["factor_scope"], "strategy_mode": back_cfg["strategy_mode"]}
        row.update(res.metrics.iloc[0].to_dict())
        rows.append(row)

    if not rows:
        return None
    summary_path = bt_dir / "backtest_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path


def _write_stage_only_outputs(
    out_dir: Path,
    stage: str,
    summary_row: dict[str, Any],
    extra_tables: list[Path] | None = None,
    config_payload: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """在提前停止阶段时生成最小可审计输出。"""
    assets_dir = out_dir / "assets"
    tables_dir = out_dir / "tables"
    assets_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([summary_row])
    summary_path = tables_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    table_map: dict[str, list[Path]] = {"global": [summary_path]}
    if extra_tables:
        table_map["global"].extend(extra_tables)

    index_html = render_report(
        out_dir=out_dir,
        summary=summary,
        figure_map={},
        table_map=table_map,
    )
    config_json = out_dir / "config.json"
    config_json.write_text(
        json.dumps(
            {
                "stop_after": stage,
                **(config_payload or {}),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "index_html": index_html,
        "summary_csv": summary_path,
        "config_json": config_json,
        "assets_dir": assets_dir,
        "tables_dir": tables_dir,
    }


def run_from_config(
    config: str | Path | dict[str, Any],
    out_dir: str | Path,
    repo_root: str | Path | None = None,
    validate_schema: bool = True,
) -> ConfigRunResult:
    """按 YAML/字典配置执行 TS/CS 因子研究流程。"""
    raw_cfg_loaded = load_run_config(config) if not isinstance(config, dict) else dict(config)
    raw_cfg, alias_events = normalize_run_config_aliases(raw_cfg_loaded)
    governance_cfg = _normalize_run_governance_cfg(raw_cfg)
    schema_warnings: list[str] = []
    if validate_schema:
        schema_warnings = validate_run_config_schema(raw_cfg, strict=True)
    scope_cfg = _normalize_factor_scope(raw_cfg)
    data_cfg = _normalize_data_cfg(raw_cfg, scope=scope_cfg["factor_scope"])
    fac_cfg = _normalize_factor_cfg(raw_cfg)
    research_cfg = _normalize_research_cfg(
        raw_cfg,
        scope=scope_cfg["factor_scope"],
        profile=str(governance_cfg["research_profile"]),
    )
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

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    captured_warnings: list[Any] = []

    adapter_stage = DataAdapterWorkflow(
        data_cfg=data_cfg,
        scope_cfg=scope_cfg,
        out_dir=out,
        timings=timings,
    ).run()
    panel = adapter_stage.panel
    load_report = adapter_stage.load_report
    mode_report = adapter_stage.mode_report
    adapter_validation_report = adapter_stage.adapter_validation_report
    adapter_audit_tables = adapter_stage.adapter_audit_tables
    data_adapter_registry = adapter_stage.adapter_registry
    data_adapter_validator_registry = adapter_stage.adapter_validator_registry

    with timed_stage("build_factor_registry", timings=timings, logger_name="factorlab.workflows.config_runner"):
        factor_registry = build_factor_registry(
            plugin_dirs=fac_cfg["plugin_dirs"] if fac_cfg["auto_discover"] else [],
            plugin_specs=fac_cfg["plugins"],
            on_plugin_error=fac_cfg["plugin_on_error"],
        )
    with timed_stage("build_transform_registry", timings=timings, logger_name="factorlab.workflows.config_runner"):
        transform_registry = build_transform_registry(
            plugin_dirs=(
                research_cfg["transform_plugin_dirs"] if research_cfg["transform_auto_discover"] else []
            ),
            plugin_specs=research_cfg["transform_plugins"],
            on_plugin_error=research_cfg["transform_plugin_on_error"],
            include_defaults=True,
        )
    requested_factors = list(fac_cfg["names"])
    auto_discovered_requested_factors: list[str] = []
    if not requested_factors and fac_cfg["auto_discover_from_panel"]:
        auto_discovered_requested_factors = _discover_panel_factor_columns(panel)
        requested_factors = list(auto_discovered_requested_factors)
        if auto_discovered_requested_factors:
            LOGGER.info(
                "factor.names 未显式配置，已从面板自动发现 %s 个因子列。",
                len(auto_discovered_requested_factors),
            )
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
        combination_dependencies.update(
            str(x).strip() for x in spec.get("orthogonalize_to", []) if str(x).strip()
        )
    if not fac_cfg["names"]:
        for name in sorted(expression_outputs | combination_outputs):
            if name and name not in requested_factors:
                requested_factors.append(name)
    if not requested_factors:
        raise RuntimeError(
            "No factors configured or discovered. "
            "请在 factor.names 显式填写因子名，或在 panel 中提供可直接研究的因子列。"
        )

    auto_factor_candidates = sorted(
        (set(requested_factors) - expression_outputs - combination_outputs)
        | expression_dependencies
        | combination_dependencies
    )
    leakage_guard_report = _run_leakage_guard(
        governance_cfg=governance_cfg,
        panel=panel,
        requested_factors=requested_factors,
        expression_dependencies=expression_dependencies,
        combination_dependencies=combination_dependencies,
    )

    with timed_stage("factor_compute", timings=timings, logger_name="factorlab.workflows.config_runner"):
        candidate_factors, precheck_skipped_factors = _filter_factors_by_available_columns(
            panel=panel,
            factor_names=auto_factor_candidates,
            on_missing=fac_cfg["on_missing"],
        )
        required_fields = _resolve_required_fields(
            scope_cfg=scope_cfg,
            data_cfg=data_cfg,
            factor_names=candidate_factors,
            research_cfg=research_cfg,
        )
        _validate_required_fields(panel, required=required_fields)
        panel, computed_factors, effective_factors = _compute_factors(
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
            LOGGER.warning(
                "Skip unresolved requested factors due to factor.on_missing=warn_skip: %s",
                unresolved_requested,
            )
        else:
            raise KeyError(f"Requested factors missing after compute/expression steps: {unresolved_requested}")
    effective_factors = [f for f in requested_factors if f in panel.columns]
    if not effective_factors:
        raise RuntimeError("No effective requested factors available after compute/expression steps.")

    universe_report = None
    if scope_cfg["factor_scope"] == "cs" and universe_cfg["enabled"]:
        with timed_stage("universe_filter", timings=timings, logger_name="factorlab.workflows.config_runner"):
            panel, universe_report = apply_universe_filter(panel, config=universe_cfg["config"])

    custom_transform_report: dict[str, Any] = {"applied": [], "skipped": [], "errors": []}
    if research_cfg["custom_transforms"]:
        with timed_stage("custom_transforms", timings=timings, logger_name="factorlab.workflows.config_runner"):
            panel, custom_transform_report = _apply_custom_transforms(
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

    if stop_after == "factor":
        with timed_stage("factor_stage_export", timings=timings, logger_name="factorlab.workflows.config_runner"):
            stage_data_dir = out / "data"
            stage_data_dir.mkdir(parents=True, exist_ok=True)
            factor_stage_panel = stage_data_dir / "panel_after_factor.parquet"
            panel.to_parquet(factor_stage_panel, index=False)
            outputs = _write_stage_only_outputs(
                out_dir=out,
                stage="factor",
                summary_row={
                    "stage": "factor",
                    "status": "stopped",
                    "rows": int(len(panel)),
                    "assets": int(panel["asset"].nunique()),
                    "factors": ",".join(effective_factors),
                },
                extra_tables=[Path(x) for x in adapter_audit_tables.values()],
                config_payload={
                    "scope": scope_cfg,
                    "research": research_cfg,
                    "effective_factors": effective_factors,
                },
            )
            outputs["factor_stage_panel"] = factor_stage_panel
    else:
        with warnings.catch_warnings(record=True) as _caught:
            warnings.simplefilter("always")

            with timed_stage("research", timings=timings, logger_name="factorlab.workflows.config_runner"):
                if scope_cfg["factor_scope"] == "cs":
                    wins = _as_dict(research_cfg.get("winsorize"))
                    neu = _as_dict(research_cfg.get("neutralize"))
                    neutral_mode = (
                        str(neu.get("mode", "both")).strip().lower() if _to_bool(neu.get("enabled"), True) else "none"
                    )
                    if neutral_mode not in {"none", "size", "industry", "both"}:
                        LOGGER.warning("Invalid research.neutralize.mode='%s'. Use 'both'.", neutral_mode)
                        neutral_mode = "both"

                    cs_cfg = ResearchConfig(
                        horizons=research_cfg["horizons"],
                        quantiles=int(research_cfg["quantiles"]),
                        ic_rolling_window=int(research_cfg["ic_rolling_window"]),
                        annualization_days=int(research_cfg["annualization_days"]),
                        standardization=scope_cfg["standardization"],  # type: ignore[arg-type]
                        winsorize_enabled=_to_bool(wins.get("enabled"), True),
                        winsorize_method=str(wins.get("method", "quantile")).strip().lower(),
                        lower_q=_to_float(wins.get("lower_q"), 0.01),
                        upper_q=_to_float(wins.get("upper_q"), 0.99),
                        mad_scale=max(1.0, _to_float(wins.get("mad_scale"), 5.0)),
                        missing_policy=str(research_cfg.get("missing_policy", "drop")),
                        preprocess_steps=research_cfg.get("preprocess_steps")
                        or ["winsorize", "standardize", "neutralize"],
                        neutralization=NeutralizationConfig(mode=neutral_mode),  # type: ignore[arg-type]
                    )
                    outputs = FactorResearchPipeline(cs_cfg).run(panel=panel, factors=effective_factors, out_dir=out)
                else:
                    ts_cfg = TSResearchConfig(
                        horizons=research_cfg["horizons"],
                        quantiles=int(research_cfg["quantiles"]),
                        ic_rolling_window=int(research_cfg["ic_rolling_window"]),
                        annualization_days=int(research_cfg["annualization_days"]),
                        standardization=scope_cfg["standardization"],  # type: ignore[arg-type]
                        ts_standardize_window=int(research_cfg["ts_standardize_window"]),
                        ts_quantile_lookback=int(research_cfg["ts_quantile_lookback"]),
                        ts_signal_lags=list(research_cfg["ts_signal_lags"]),
                    )
                    outputs = TimeSeriesFactorResearchPipeline(ts_cfg).run(
                        panel=panel,
                        factors=effective_factors,
                        out_dir=out,
                    )

            if stop_after == "backtest":
                with timed_stage("backtest", timings=timings, logger_name="factorlab.workflows.config_runner"):
                    backtest_summary_csv = _run_optional_backtest(
                        panel=panel,
                        factors=effective_factors,
                        scope_cfg=scope_cfg,
                        back_cfg=back_cfg,
                        out_dir=out,
                    )
            captured_warnings.extend(list(_caught))

    meta = {
        "scope": scope_cfg,
        "data": {
            "config": data_cfg,
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
            "candidate_after_precheck": candidate_factors,
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
            "plugin_config": {
                "auto_discover": fac_cfg["auto_discover"],
                "plugin_dirs": fac_cfg["plugin_dirs"],
                "plugins": fac_cfg["plugins"],
                "plugin_on_error": fac_cfg["plugin_on_error"],
                "registry_size": len(factor_registry),
            },
        },
        "research": {
            "config": research_cfg,
            "custom_transform_report": custom_transform_report,
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
        "backtest": {"config": back_cfg, "summary_csv": str(backtest_summary_csv) if backtest_summary_csv else None},
        "rows_after_pipeline": int(len(panel)),
        "assets_after_pipeline": int(panel["asset"].nunique()),
        "dates_after_pipeline": int(panel["date"].nunique()),
        "timings_seconds": timings,
        "warning_summary": summarize_captured_warnings(
            captured_warnings,
            logger_name="factorlab.workflows.config_runner",
        ),
        "outputs": {
            **{k: str(v) for k, v in outputs.items()},
            **adapter_audit_tables,
        },
    }
    meta_path = out / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    manifest_path = out / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(collect_runtime_manifest(repo_root=repo_root), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return ConfigRunResult(
        out_dir=out,
        index_html=Path(outputs["index_html"]),
        summary_csv=Path(outputs["summary_csv"]),
        run_meta_json=meta_path,
        run_manifest_json=manifest_path,
        backtest_summary_csv=backtest_summary_csv,
    )
