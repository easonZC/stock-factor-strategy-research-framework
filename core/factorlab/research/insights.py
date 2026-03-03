"""因子结果自动解读模板。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _strength_label(primary_abs: float) -> str:
    if primary_abs >= 0.05:
        return "强"
    if primary_abs >= 0.02:
        return "中"
    return "弱"


def _confidence_label(pvalue: float) -> str:
    if np.isfinite(pvalue) and pvalue <= 0.01:
        return "高"
    if np.isfinite(pvalue) and pvalue <= 0.10:
        return "中"
    return "低"


def _risk_label(drawdown: float, sharpe: float) -> str:
    if np.isfinite(drawdown) and drawdown <= -0.20:
        return "回撤偏大"
    if np.isfinite(sharpe) and sharpe < 0:
        return "收益风险比偏弱"
    return "可控"


def _format_metric_text(name: str, value: float) -> str:
    if not np.isfinite(value):
        return f"{name}=NaN"
    return f"{name}={value:.4f}"


def build_factor_insights(scorecard: pd.DataFrame) -> pd.DataFrame:
    """基于评分卡生成中文解读模板。"""
    if scorecard.empty:
        return pd.DataFrame(
            columns=[
                "rank",
                "factor",
                "insight_title",
                "strength",
                "confidence",
                "risk",
                "summary_text",
                "action_text",
            ]
        )

    rows: list[dict[str, object]] = []
    for _, row in scorecard.iterrows():
        rank = int(row.get("rank", 0))
        factor = str(row.get("factor", "unknown"))
        variant = str(row.get("variant", "default"))
        horizon = row.get("horizon", np.nan)
        primary_metric = str(row.get("primary_metric", "metric"))
        primary_value = _safe_float(row.get("primary_value"))
        direction = str(row.get("direction", ""))
        p_metric = str(row.get("pvalue_metric", "pvalue"))
        p_value = _safe_float(row.get("pvalue"))
        action = str(row.get("action", "继续观察"))
        ls_sharpe = _safe_float(row.get("ls_sharpe"))
        ls_dd = _safe_float(row.get("ls_max_drawdown"))

        strength = _strength_label(abs(primary_value) if np.isfinite(primary_value) else 0.0)
        confidence = _confidence_label(p_value)
        risk = _risk_label(ls_dd, ls_sharpe)

        h_text = f"h{int(horizon)}" if pd.notna(horizon) else "h?"
        core_text = _format_metric_text(primary_metric, primary_value)
        p_text = _format_metric_text(p_metric, p_value)
        sharpe_text = _format_metric_text("ls_sharpe", ls_sharpe)
        dd_text = _format_metric_text("ls_max_drawdown", ls_dd)

        summary = (
            f"{factor} 在 {variant}/{h_text} 下为{direction}信号，"
            f"{core_text}，显著性 {p_text}；"
            f"信号强度={strength}，置信度={confidence}，风险={risk}。"
        )

        if action == "优先跟踪":
            action_text = "建议纳入主观察池，进入滚动监控与组合权重试验。"
        elif action == "继续观察":
            action_text = "建议保留观察，等待更多样本或与其他因子组合后再决策。"
        else:
            action_text = "建议降权处理，仅作为补充特征或对照项。"

        action_text = f"{action_text}（{sharpe_text}, {dd_text}）"

        rows.append(
            {
                "rank": rank,
                "factor": factor,
                "insight_title": f"#{rank} {factor}",
                "strength": strength,
                "confidence": confidence,
                "risk": risk,
                "summary_text": summary,
                "action_text": action_text,
            }
        )

    out = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)
    return out

