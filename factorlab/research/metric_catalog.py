"""因子评估指标分层与速览视图。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """指标定义。"""

    name: str
    tier: str
    scope: str
    meaning: str


METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("rank_ic_mean", "core", "cs", "截面秩相关均值，主判断指标"),
    MetricSpec("rank_icir", "core", "cs", "截面秩相关信息比率，稳定性指标"),
    MetricSpec("nw_p_rank_ic", "core", "cs", "秩相关 Newey-West 显著性 p 值"),
    MetricSpec("ls_sharpe", "core", "cs_ts", "多空组合夏普"),
    MetricSpec("ls_max_drawdown", "core", "cs_ts", "多空组合最大回撤"),
    MetricSpec("ic_mean", "core", "ts_cs", "相关系数均值（TS/CS 通用）"),
    MetricSpec("icir", "core", "ts_cs", "相关系数信息比率（TS/CS 通用）"),
    MetricSpec("nw_p_ic", "core", "ts_cs", "相关系数 Newey-West 显著性 p 值"),
    MetricSpec("signal_lag0_ic_mean", "core", "ts", "TS 当期信号 IC 均值"),
    MetricSpec("signal_best_lag_ic_mean", "core", "ts", "TS 最优滞后信号 IC 均值"),
    MetricSpec("fmb_factor_beta_mean", "diagnostic", "cs", "Fama-MacBeth 因子暴露均值"),
    MetricSpec("fmb_factor_nw_p", "diagnostic", "cs", "Fama-MacBeth 因子暴露显著性"),
    MetricSpec("industry_top_group_mean_ls", "diagnostic", "cs", "行业分解中最佳组多空收益"),
    MetricSpec("style_top_group_mean_ls", "diagnostic", "cs", "风格分解中最佳组多空收益"),
    MetricSpec("quantile_monotonicity_mean", "diagnostic", "cs", "分位收益单调性均值"),
    MetricSpec("rank_autocorr_lag1_mean", "diagnostic", "cs", "因子秩自相关"),
    MetricSpec("nw_t_long_short", "diagnostic", "cs_ts", "多空收益显著性 t 值"),
    MetricSpec("nw_p_long_short", "diagnostic", "cs_ts", "多空收益显著性 p 值"),
)


def _coerce_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_metric_inventory(summary: pd.DataFrame) -> pd.DataFrame:
    """生成指标分层清单（不删除任何指标，仅做分级可视化）。"""
    total = int(len(summary))
    rows: list[dict[str, object]] = []
    known = set()

    for spec in METRIC_SPECS:
        known.add(spec.name)
        present = spec.name in summary.columns
        non_null = 0
        if present and total > 0:
            non_null = int(_coerce_num(summary[spec.name]).notna().sum())
        rows.append(
            {
                "metric": spec.name,
                "tier": spec.tier,
                "scope": spec.scope,
                "present": bool(present),
                "non_null_rows": non_null,
                "non_null_ratio": float(non_null / total) if total > 0 else 0.0,
                "meaning": spec.meaning,
            }
        )

    for col in summary.columns:
        if col in known or col in {"factor", "variant", "horizon"}:
            continue
        if summary[col].dtype.kind not in {"i", "u", "f"} and col not in {"industry_top_group", "style_top_group"}:
            continue
        non_null = int(pd.to_numeric(summary[col], errors="coerce").notna().sum())
        rows.append(
            {
                "metric": col,
                "tier": "diagnostic",
                "scope": "unknown",
                "present": True,
                "non_null_rows": non_null,
                "non_null_ratio": float(non_null / total) if total > 0 else 0.0,
                "meaning": "未分类指标（自动归入诊断层）",
            }
        )

    inv = pd.DataFrame(rows)
    if inv.empty:
        return inv
    tier_order = {"core": 0, "diagnostic": 1}
    inv["_tier"] = inv["tier"].map(tier_order).fillna(9)
    inv = inv.sort_values(["_tier", "metric"]).drop(columns=["_tier"]).reset_index(drop=True)
    return inv


def _first_available_metric(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns and _coerce_num(df[col]).notna().any():
            return col
    return ""


def build_factor_scorecard(summary: pd.DataFrame) -> pd.DataFrame:
    """生成每个因子的核心评分卡，便于快速决策。"""
    if summary.empty or "factor" not in summary.columns:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for fac, grp in summary.groupby("factor", dropna=False):
        g = grp.copy()
        primary_metric = _first_available_metric(
            g,
            candidates=["rank_ic_mean", "ic_mean", "signal_lag0_ic_mean"],
        )
        if not primary_metric:
            continue
        vals = _coerce_num(g[primary_metric])
        if not vals.notna().any():
            continue
        idx = vals.abs().idxmax()
        row = g.loc[idx]
        primary_val = float(pd.to_numeric(row.get(primary_metric), errors="coerce"))

        icir_metric = _first_available_metric(pd.DataFrame([row]), candidates=["rank_icir", "icir"])
        p_metric = _first_available_metric(pd.DataFrame([row]), candidates=["nw_p_rank_ic", "nw_p_ic", "nw_p_long_short"])
        icir_val = float(pd.to_numeric(row.get(icir_metric), errors="coerce")) if icir_metric else float("nan")
        p_val = float(pd.to_numeric(row.get(p_metric), errors="coerce")) if p_metric else float("nan")
        ls_sharpe = float(pd.to_numeric(row.get("ls_sharpe"), errors="coerce"))
        ls_dd = float(pd.to_numeric(row.get("ls_max_drawdown"), errors="coerce"))

        if np.isfinite(p_val) and p_val <= 0.10 and abs(primary_val) >= 0.02:
            action = "优先跟踪"
        elif abs(primary_val) >= 0.01:
            action = "继续观察"
        else:
            action = "低优先级"

        rows.append(
            {
                "factor": str(fac),
                "variant": str(row.get("variant", "default")),
                "horizon": row.get("horizon", np.nan),
                "primary_metric": primary_metric,
                "primary_value": primary_val,
                "direction": "正向" if primary_val >= 0 else "反向",
                "icir_metric": icir_metric or "",
                "icir_value": icir_val,
                "pvalue_metric": p_metric or "",
                "pvalue": p_val,
                "ls_sharpe": ls_sharpe,
                "ls_max_drawdown": ls_dd,
                "action": action,
            }
        )

    score = pd.DataFrame(rows)
    if score.empty:
        return score
    score = score.sort_values("primary_value", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    score.insert(0, "rank", np.arange(1, len(score) + 1))
    return score

