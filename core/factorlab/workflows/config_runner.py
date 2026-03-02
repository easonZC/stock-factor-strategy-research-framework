"""Config-driven one-click runner for TS/CS factor research workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    generate_synthetic_panel,
    prepare_sina_panel,
    read_panel,
)
from factorlab.factors import apply_factors, default_factor_registry
from factorlab.research import FactorResearchPipeline, TSResearchConfig, TimeSeriesFactorResearchPipeline
from factorlab.strategies import FlexibleLongShortStrategy, LongShortQuantileStrategy, Strategy, TopKLongStrategy
from factorlab.utils import get_logger
from factorlab.workflows.runtime import collect_runtime_manifest

LOGGER = get_logger("factorlab.workflows.config_runner")


FactorScope = Literal["ts", "cs"]
EvalAxis = Literal["time", "cross_section"]

FACTOR_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "momentum_20": {"close"},
    "volatility_20": {"close"},
    "liquidity_shock": {"volume"},
    "size": {"mkt_cap"},
}


@dataclass(slots=True)
class ConfigRunResult:
    """Result pointers for config-driven run."""

    out_dir: Path
    index_html: Path
    summary_csv: Path
    run_meta_json: Path
    run_manifest_json: Path
    backtest_summary_csv: Path | None


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


def _normalize_data_cfg(cfg: dict[str, Any], scope: FactorScope) -> dict[str, Any]:
    data_cfg = _as_dict(cfg.get("data"))
    adapter = str(data_cfg.get("adapter", "synthetic")).strip().lower()
    if adapter not in {"synthetic", "sina", "parquet", "csv"}:
        LOGGER.warning("Invalid data.adapter='%s'. Use synthetic.", adapter)
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
        "path": data_cfg.get("path"),
        "data_dir": data_cfg.get("data_dir"),
        "asset": data_cfg.get("asset"),
        "sanitize": _to_bool(data_cfg.get("sanitize"), True),
        "duplicate_policy": str(data_cfg.get("duplicate_policy", "last")).strip().lower(),
        "fields_required": fields_required,
        "synthetic": _as_dict(data_cfg.get("synthetic")),
    }


def _normalize_factor_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    fac_cfg = _as_dict(cfg.get("factor"))
    names = [str(x).strip() for x in _as_list(fac_cfg.get("names")) if str(x).strip()]
    if not names:
        names = ["momentum_20"]
    on_missing = str(fac_cfg.get("on_missing", "raise")).strip().lower()
    if on_missing not in {"raise", "warn_skip"}:
        LOGGER.warning("Invalid factor.on_missing='%s'. Use 'raise'.", on_missing)
        on_missing = "raise"
    return {"names": names, "on_missing": on_missing}


def _normalize_research_cfg(cfg: dict[str, Any], scope: FactorScope) -> dict[str, Any]:
    raw = _as_dict(cfg.get("research"))
    horizons_raw = _as_list(raw.get("horizons"))
    horizons = sorted(
        {
            _to_int(x, 0)
            for x in horizons_raw
            if _to_int(x, 0) > 0
        }
    )
    if not horizons:
        horizons = [1, 5, 10, 20]

    missing_policy = str(raw.get("missing_policy", "drop")).strip().lower()
    allowed_missing = {"drop", "fill_zero", "ffill_by_asset", "cs_median_by_date", "keep"}
    if missing_policy not in allowed_missing:
        LOGGER.warning("Invalid research.missing_policy='%s'. Use 'drop'.", missing_policy)
        missing_policy = "drop"

    default_steps = ["winsorize", "standardize", "neutralize"] if scope == "cs" else []
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

    out = {
        "horizons": horizons,
        "quantiles": max(2, _to_int(raw.get("quantiles"), 5)),
        "ic_rolling_window": max(5, _to_int(raw.get("ic_rolling_window"), 20)),
        "ts_standardize_window": max(5, _to_int(raw.get("ts_standardize_window"), 60)),
        "ts_quantile_lookback": max(10, _to_int(raw.get("ts_quantile_lookback"), 60)),
        "missing_policy": missing_policy,
        "preprocess_steps": preprocess_steps,
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
    strategy_mode = str(strategy_cfg.get("mode", default_mode)).strip().lower()
    if strategy_mode not in {"sign", "topk", "longshort", "flex"}:
        LOGGER.warning("Invalid backtest.strategy.mode='%s'. Use '%s'.", strategy_mode, default_mode)
        strategy_mode = default_mode

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
        "sign_threshold": max(0.0, _to_float(strategy_cfg.get("sign_threshold"), 0.0)),
        "commission_bps": _to_float(raw.get("commission_bps"), 3.0),
        "slippage_bps": _to_float(raw.get("slippage_bps"), 2.0),
        "leverage": max(0.1, _to_float(raw.get("leverage"), 1.0)),
        "execution_delay_days": max(0, _to_int(raw.get("execution_delay_days"), 1)),
        "execution_price_col": str(raw.get("execution_price_col", "close")).strip(),
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


def load_run_config(path: str | Path) -> dict[str, Any]:
    """Load yaml run configuration from file path."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    payload = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config must be a YAML object.")
    return payload


def deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dict values from overlay into base and return a new dict."""
    out = dict(base)
    for key, val in overlay.items():
        cur = out.get(key)
        if isinstance(cur, dict) and isinstance(val, dict):
            out[key] = deep_merge_dict(cur, val)
        else:
            out[key] = val
    return out


def apply_config_override(cfg: dict[str, Any], override: str) -> dict[str, Any]:
    """Apply one dotted-path override in form `a.b.c=value` and return a new dict."""
    text = str(override).strip()
    if "=" not in text:
        raise ValueError(f"Invalid override '{override}'. Expected format: key.path=value")
    path_raw, value_raw = text.split("=", 1)
    path = [x.strip() for x in path_raw.split(".") if x.strip()]
    if not path:
        raise ValueError(f"Invalid override '{override}'. Empty key path.")
    try:
        value = yaml.safe_load(value_raw)
    except Exception:
        value = value_raw

    out = dict(cfg)
    node: dict[str, Any] = out
    for seg in path[:-1]:
        nxt = node.get(seg)
        if not isinstance(nxt, dict):
            nxt = {}
            node[seg] = nxt
        node = nxt
    node[path[-1]] = value
    return out


def compose_run_config(
    config_paths: list[str | Path],
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Compose effective run config from multiple yaml files and dotted overrides.

    Merge order:
    - Start from first config file.
    - Deep-merge subsequent files from left to right.
    - Apply overrides in provided order.
    """
    if not config_paths:
        raise ValueError("At least one config path is required.")

    merged = load_run_config(config_paths[0])
    for path in config_paths[1:]:
        merged = deep_merge_dict(merged, load_run_config(path))

    for ov in overrides or []:
        merged = apply_config_override(merged, ov)
    return merged


def _load_data(data_cfg: dict[str, Any], scope_cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    adapter = data_cfg["adapter"]
    sanitize = bool(data_cfg["sanitize"])
    duplicate_policy = str(data_cfg["duplicate_policy"])
    loader_report: dict[str, Any] = {"adapter": adapter, "sanitize": sanitize}

    if adapter == "synthetic":
        s = data_cfg["synthetic"]
        n_assets_default = 1 if data_cfg["mode"] == "single_asset" else 40
        syn_cfg = SyntheticConfig(
            n_assets=max(1, _to_int(s.get("n_assets"), n_assets_default)),
            n_days=max(60, _to_int(s.get("n_days"), 260)),
            seed=_to_int(s.get("seed"), 7),
            start_date=str(s.get("start_date", "2021-01-01")),
        )
        panel = generate_synthetic_panel(syn_cfg)
        loader_report["synthetic"] = syn_cfg.__dict__ if hasattr(syn_cfg, "__dict__") else {
            "n_assets": syn_cfg.n_assets,
            "n_days": syn_cfg.n_days,
            "seed": syn_cfg.seed,
            "start_date": syn_cfg.start_date,
        }
        return panel, loader_report

    if adapter == "sina":
        data_dir = data_cfg.get("data_dir")
        if not data_dir:
            raise ValueError("data.data_dir is required when data.adapter=sina")
        panel = prepare_sina_panel(AdapterConfig(data_dir=str(data_dir)))
        return panel, loader_report

    path = data_cfg.get("path")
    if not path:
        raise ValueError("data.path is required when data.adapter is parquet/csv")
    read_res = read_panel(
        path=str(path),
        sanitize=sanitize,
        sanitization_config=PanelSanitizationConfig(duplicate_policy=duplicate_policy),
        return_report=sanitize,
    )
    if sanitize:
        panel, report = read_res
        loader_report["sanitization_report"] = report.__dict__ if hasattr(report, "__dict__") else str(report)
    else:
        panel = read_res
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
    if not selected:
        raise RuntimeError("No valid factors available after checking required columns.")
    return selected, skipped


def _validate_required_fields(panel: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise KeyError(f"Data missing required fields: {missing}")


def _compute_factors(
    panel: pd.DataFrame,
    factor_names: list[str],
    on_missing: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = panel.copy()
    registry = default_factor_registry()
    missing = [f for f in factor_names if f not in out.columns]
    computable = [f for f in missing if f in registry]
    if computable:
        out = apply_factors(out, computable, inplace=True)
    unresolved = [f for f in factor_names if f not in out.columns]
    if unresolved:
        if on_missing == "warn_skip":
            LOGGER.warning("Skip unresolved factors due to factor.on_missing=warn_skip: %s", unresolved)
        else:
            raise KeyError(f"Factors missing and not computable: {unresolved}")
    selected = [f for f in factor_names if f not in unresolved]
    if not selected:
        raise RuntimeError("No valid factors available after resolving factor names.")
    return out, computable, selected


def _build_strategy(back_cfg: dict[str, Any]) -> Strategy:
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
    return FlexibleLongShortStrategy(
        name="flexible_long_short",
        long_fraction=float(back_cfg["long_fraction"]),
        short_fraction=float(back_cfg["short_fraction"]),
        long_only=bool(back_cfg["long_only"]),
        rebalance_every=int(back_cfg["rebalance_every"]),
        weight_scheme=back_cfg["weight_scheme"],
        max_weight=back_cfg["max_weight"],
    )


def _build_sign_weights(score_df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    tmp = score_df[["date", "asset", "score"]].copy()
    tmp["score"] = pd.to_numeric(tmp["score"], errors="coerce")
    w = np.sign(tmp["score"]).astype(float)
    if threshold > 0:
        w[np.abs(tmp["score"]) <= float(threshold)] = 0.0
    out = tmp[["date", "asset"]].copy()
    out["weight"] = w
    return out


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
    )

    strategy = None
    if back_cfg["strategy_mode"] in {"topk", "longshort", "flex"}:
        strategy = _build_strategy(back_cfg)

    rows: list[dict[str, Any]] = []
    for fac in factors:
        score_df = panel[["date", "asset", fac]].rename(columns={fac: "score"})
        if back_cfg["strategy_mode"] == "sign":
            weights = _build_sign_weights(score_df, threshold=float(back_cfg["sign_threshold"]))
        else:
            assert strategy is not None
            weights = strategy.generate_weights(score_df)

        res = run_backtest(panel=panel, weights=weights, config=bt_cfg)
        fac_dir = bt_dir / fac
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


def run_from_config(
    config: str | Path | dict[str, Any],
    out_dir: str | Path,
    repo_root: str | Path | None = None,
) -> ConfigRunResult:
    """Run TS/CS factor pipeline from yaml/dict configuration."""
    raw_cfg = load_run_config(config) if not isinstance(config, dict) else dict(config)
    scope_cfg = _normalize_factor_scope(raw_cfg)
    data_cfg = _normalize_data_cfg(raw_cfg, scope=scope_cfg["factor_scope"])
    fac_cfg = _normalize_factor_cfg(raw_cfg)
    research_cfg = _normalize_research_cfg(raw_cfg, scope=scope_cfg["factor_scope"])
    back_cfg = _normalize_backtest_cfg(raw_cfg, scope=scope_cfg["factor_scope"])
    universe_cfg = _normalize_universe_cfg(raw_cfg)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    panel, load_report = _load_data(data_cfg, scope_cfg)
    panel, mode_report = _ensure_mode_shape(panel, data_cfg=data_cfg)
    requested_factors = list(fac_cfg["names"])
    candidate_factors, precheck_skipped_factors = _filter_factors_by_available_columns(
        panel=panel,
        factor_names=requested_factors,
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
    )

    universe_report = None
    if scope_cfg["factor_scope"] == "cs" and universe_cfg["enabled"]:
        panel, universe_report = apply_universe_filter(panel, config=universe_cfg["config"])

    panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)
    if panel.empty:
        raise RuntimeError("No data available after preprocessing.")

    if scope_cfg["factor_scope"] == "cs":
        wins = _as_dict(research_cfg.get("winsorize"))
        neu = _as_dict(research_cfg.get("neutralize"))
        neutral_mode = str(neu.get("mode", "both")).strip().lower() if _to_bool(neu.get("enabled"), True) else "none"
        if neutral_mode not in {"none", "size", "industry", "both"}:
            LOGGER.warning("Invalid research.neutralize.mode='%s'. Use 'both'.", neutral_mode)
            neutral_mode = "both"

        cs_cfg = ResearchConfig(
            horizons=research_cfg["horizons"],
            quantiles=int(research_cfg["quantiles"]),
            ic_rolling_window=int(research_cfg["ic_rolling_window"]),
            standardization=scope_cfg["standardization"],  # type: ignore[arg-type]
            winsorize_enabled=_to_bool(wins.get("enabled"), True),
            winsorize_method=str(wins.get("method", "quantile")).strip().lower(),
            lower_q=_to_float(wins.get("lower_q"), 0.01),
            upper_q=_to_float(wins.get("upper_q"), 0.99),
            mad_scale=max(1.0, _to_float(wins.get("mad_scale"), 5.0)),
            missing_policy=str(research_cfg.get("missing_policy", "drop")),
            preprocess_steps=research_cfg.get("preprocess_steps") or ["winsorize", "standardize", "neutralize"],
            neutralization=NeutralizationConfig(mode=neutral_mode),  # type: ignore[arg-type]
        )
        outputs = FactorResearchPipeline(cs_cfg).run(panel=panel, factors=effective_factors, out_dir=out)
    else:
        ts_cfg = TSResearchConfig(
            horizons=research_cfg["horizons"],
            quantiles=int(research_cfg["quantiles"]),
            ic_rolling_window=int(research_cfg["ic_rolling_window"]),
            standardization=scope_cfg["standardization"],  # type: ignore[arg-type]
            ts_standardize_window=int(research_cfg["ts_standardize_window"]),
            ts_quantile_lookback=int(research_cfg["ts_quantile_lookback"]),
        )
        outputs = TimeSeriesFactorResearchPipeline(ts_cfg).run(panel=panel, factors=effective_factors, out_dir=out)

    backtest_summary_csv = _run_optional_backtest(
        panel=panel,
        factors=effective_factors,
        scope_cfg=scope_cfg,
        back_cfg=back_cfg,
        out_dir=out,
    )

    meta = {
        "scope": scope_cfg,
        "data": {
            "config": data_cfg,
            "load_report": load_report,
            "mode_report": mode_report,
            "required_fields": required_fields,
        },
        "factors": {
            "requested": requested_factors,
            "candidate_after_precheck": candidate_factors,
            "skipped_in_precheck": precheck_skipped_factors,
            "effective": effective_factors,
            "computed_factors": computed_factors,
            "on_missing": fac_cfg["on_missing"],
        },
        "research": research_cfg,
        "universe_filter": {
            "enabled": universe_cfg["enabled"],
            "report": universe_report.__dict__ if hasattr(universe_report, "__dict__") else universe_report,
        },
        "backtest": {"config": back_cfg, "summary_csv": str(backtest_summary_csv) if backtest_summary_csv else None},
        "rows_after_pipeline": int(len(panel)),
        "assets_after_pipeline": int(panel["asset"].nunique()),
        "dates_after_pipeline": int(panel["date"].nunique()),
        "outputs": {k: str(v) for k, v in outputs.items()},
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
