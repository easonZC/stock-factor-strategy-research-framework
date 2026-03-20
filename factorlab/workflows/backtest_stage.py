"""Strategy resolution and optional backtest stage helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factorlab.backtest import run_backtest
from factorlab.config import BacktestConfig, CostConfig
from factorlab.strategies import (
    FlexibleLongShortStrategy,
    LongShortQuantileStrategy,
    MeanVarianceOptimizerStrategy,
    Strategy,
    StrategyDefinition,
    TopKLongStrategy,
    build_strategy_registry,
    strategy_definition_from_instance,
)
from factorlab.utils import safe_slug
from factorlab.workflows.config_normalization import to_optional_float as _to_optional_float
from factorlab.workflows.plugin_preflight import preflight_requested_components


def build_strategy(back_cfg: dict[str, Any], strategy_registry: dict[str, Any]) -> Strategy:
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
        raise KeyError(f"Unknown strategy mode '{mode}'. Available strategy plugins: {sorted(strategy_registry.keys())}")
    obj = strategy_registry[mode]()
    if not isinstance(obj, Strategy):
        raise TypeError(f"Registry constructor for '{mode}' did not return Strategy instance.")
    return obj


def resolve_strategy_definition(
    back_cfg: dict[str, Any],
    strategy_registry: dict[str, Any],
) -> StrategyDefinition:
    """Build the effective strategy definition for the current run."""
    mode = str(back_cfg["strategy_mode"])
    if mode == "sign":
        return StrategyDefinition(
            name="sign",
            family="sign_rule",
            description="Sign-based strategy using factor score sign with optional threshold.",
            constraints=("gross_unbounded_raw_sign", "threshold_filter"),
            tags=("baseline", "rule_based"),
            parameters={"sign_threshold": float(back_cfg["sign_threshold"])},
            implementation="factorlab.workflows.backtest_stage.build_sign_weights",
            origin="builtin",
        )
    return strategy_definition_from_instance(build_strategy(back_cfg, strategy_registry=strategy_registry))


def build_sign_weights(score_df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    tmp = score_df[["date", "asset", "score"]].copy()
    tmp["score"] = pd.to_numeric(tmp["score"], errors="coerce")
    w = np.sign(tmp["score"]).astype(float)
    if threshold > 0:
        w[np.abs(tmp["score"]) <= float(threshold)] = 0.0
    out = tmp[["date", "asset"]].copy()
    out["weight"] = w
    return out


def build_unique_slug_map(names: list[str], default: str = "item") -> dict[str, str]:
    """Build deduplicated safe slugs for arbitrary names."""
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


def run_optional_backtest(
    panel: pd.DataFrame,
    factors: list[str],
    scope_cfg: dict[str, Any],
    back_cfg: dict[str, Any],
    out_dir: Path,
    strategy_registry: dict[str, Any] | None = None,
) -> tuple[Path | None, dict[str, Any]]:
    if not back_cfg["enabled"]:
        return None, {
            "kind": "strategy",
            "requested": [],
            "available": [],
            "resolved": [],
            "missing": [],
            "on_missing": "raise",
            "alias_hits": {},
            "skipped": "backtest_disabled",
        }
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

    strategy_registry = strategy_registry or build_strategy_registry(
        plugin_dirs=back_cfg["strategy_plugin_dirs"] if back_cfg["strategy_auto_discover"] else [],
        plugin_specs=back_cfg["strategy_plugins"],
        on_plugin_error=back_cfg["strategy_plugin_on_error"],
        include_defaults=False,
    )
    strategy = None
    mode = str(back_cfg["strategy_mode"])
    builtin_modes = {"sign", "topk", "longshort", "flex", "meanvar"}
    strategy_preflight_report: dict[str, Any]
    if mode in builtin_modes:
        strategy_preflight_report = {
            "kind": "strategy",
            "requested": [mode],
            "available": sorted(builtin_modes),
            "resolved": [mode],
            "missing": [],
            "on_missing": "raise",
            "alias_hits": {},
        }
        if mode != "sign":
            strategy = build_strategy(back_cfg, strategy_registry=strategy_registry)
    else:
        strategy_preflight_report = preflight_requested_components(
            kind="strategy",
            requested=[mode],
            available=sorted(strategy_registry.keys()),
            on_missing="raise",
            logger_name="factorlab.workflows.config_runner",
        ).to_dict()
        strategy = build_strategy(back_cfg, strategy_registry=strategy_registry)

    factor_slug_map = build_unique_slug_map(factors, default="factor")
    rows: list[dict[str, Any]] = []
    for fac in factors:
        score_df = panel[["date", "asset", fac]].rename(columns={fac: "score"})
        if back_cfg["strategy_mode"] == "sign":
            weights = build_sign_weights(score_df, threshold=float(back_cfg["sign_threshold"]))
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
        return None, strategy_preflight_report
    summary_path = bt_dir / "backtest_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path, strategy_preflight_report


__all__ = [
    "build_sign_weights",
    "build_strategy",
    "resolve_strategy_definition",
    "run_optional_backtest",
]
