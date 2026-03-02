"""模块说明。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factorlab.config import BacktestConfig


@dataclass(slots=True)
class BacktestResult:
    """中文说明。"""

    daily: pd.DataFrame
    metrics: pd.DataFrame


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    return float(drawdown.min())


def _clip_abs_weight(row: pd.Series, max_abs_weight: float | None) -> pd.Series:
    if max_abs_weight is None:
        return row
    cap = abs(float(max_abs_weight))
    return row.clip(lower=-cap, upper=cap)


def _enforce_net_limit(row: pd.Series, max_net_exposure: float | None) -> pd.Series:
    if max_net_exposure is None:
        return row
    cap = abs(float(max_net_exposure))
    net = float(row.sum())
    if abs(net) <= cap + 1e-12:
        return row
    if len(row) == 0:
        return row
    adjust = (net - np.sign(net) * cap) / len(row)
    return row - adjust


def _scale_gross(row: pd.Series, target_gross: float, max_gross_exposure: float | None) -> pd.Series:
    out = row.copy()
    gross = float(out.abs().sum())
    if gross > 0 and target_gross > 0:
        out = out * (float(target_gross) / gross)
    if max_gross_exposure is not None:
        cap = abs(float(max_gross_exposure))
        gross = float(out.abs().sum())
        if gross > cap and cap > 0:
            out = out * (cap / gross)
    return out


def _finalize_caps(row: pd.Series, config: BacktestConfig) -> pd.Series:
    """中文说明。"""
    out = _clip_abs_weight(row, max_abs_weight=config.max_abs_weight)
    out = _enforce_net_limit(out, max_net_exposure=config.max_net_exposure)
    if config.max_gross_exposure is not None:
        cap = abs(float(config.max_gross_exposure))
        gross = float(out.abs().sum())
        if gross > cap and cap > 0:
            out = out * (cap / gross)
    return out


def _industry_neutralize_row(row: pd.Series, industry_row: pd.Series | None) -> pd.Series:
    if industry_row is None:
        return row
    if row.empty:
        return row
    out = row.copy()
    inds = industry_row.reindex(out.index)
    for ind in inds.dropna().unique():
        mask = inds == ind
        if int(mask.sum()) < 2:
            continue
        out.loc[mask] = out.loc[mask] - float(out.loc[mask].mean())
    return out


def _build_industry_wide(panel: pd.DataFrame, industry_col: str, columns: pd.Index) -> pd.DataFrame | None:
    if industry_col not in panel.columns:
        return None
    base = panel[["date", "asset", industry_col]].dropna(subset=["date", "asset"]).copy()
    if base.empty:
        return None
    base = base.drop_duplicates(subset=["date", "asset"], keep="last")
    wide = base.pivot(index="date", columns="asset", values=industry_col).sort_index()
    return wide.reindex(columns=columns)


def _apply_weight_constraints(
    w_target: pd.DataFrame,
    panel: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    w = w_target.copy().sort_index().fillna(0.0)
    if w.empty:
        return w

    industry_wide = _build_industry_wide(panel, industry_col=config.industry_col, columns=w.columns)
    out = pd.DataFrame(index=w.index, columns=w.columns, dtype=float)

    prev: pd.Series | None = None
    for dt in w.index:
        row = w.loc[dt].astype(float).copy()
        industry_row = None
        if bool(config.enforce_industry_neutral) and industry_wide is not None and dt in industry_wide.index:
            industry_row = industry_wide.loc[dt]

        row = _industry_neutralize_row(row, industry_row=industry_row)
        row = _clip_abs_weight(row, max_abs_weight=config.max_abs_weight)
        row = _enforce_net_limit(row, max_net_exposure=config.max_net_exposure)
        row = _scale_gross(
            row,
            target_gross=float(config.long_short_leverage),
            max_gross_exposure=config.max_gross_exposure,
        )
        row = _finalize_caps(row, config=config)

        if prev is not None and config.max_turnover is not None:
            max_turn = abs(float(config.max_turnover))
            delta = row - prev
            turn = float(delta.abs().sum())
            if turn > max_turn and max_turn >= 0:
                scale = max_turn / turn if turn > 0 else 1.0
                row = prev + delta * scale
                row = _clip_abs_weight(row, max_abs_weight=config.max_abs_weight)
                row = _enforce_net_limit(row, max_net_exposure=config.max_net_exposure)
                row = _scale_gross(
                    row,
                    target_gross=float(config.long_short_leverage),
                    max_gross_exposure=config.max_gross_exposure,
                )
                row = _finalize_caps(row, config=config)

        out.loc[dt] = row
        prev = row

    return out.fillna(0.0)


def _benchmark_returns(panel: pd.DataFrame, ret_wide: pd.DataFrame, config: BacktestConfig) -> pd.Series:
    mode = str(config.benchmark_mode).strip().lower()
    if mode == "none":
        return pd.Series(np.nan, index=ret_wide.index)
    if mode == "cross_sectional_mean":
        return ret_wide.mean(axis=1)
    if mode == "panel_column" and config.benchmark_return_col in panel.columns:
        bmk = (
            panel[["date", config.benchmark_return_col]]
            .dropna()
            .groupby("date", as_index=True)[config.benchmark_return_col]
            .mean()
            .reindex(ret_wide.index)
        )
        return bmk.astype(float)
    return pd.Series(np.nan, index=ret_wide.index)


def _alpha_beta(returns: pd.Series, benchmark: pd.Series, ann_days: int) -> tuple[float, float]:
    df = pd.DataFrame({"r": returns, "b": benchmark}).dropna()
    if len(df) < 8:
        return float("nan"), float("nan")
    x = df["b"].to_numpy(dtype=float)
    y = df["r"].to_numpy(dtype=float)
    x_var = float(np.var(x))
    if x_var <= 0:
        return float("nan"), float("nan")
    beta = float(np.mean((x - x.mean()) * (y - y.mean())) / x_var)
    alpha_daily = float(y.mean() - beta * x.mean())
    alpha_ann = float((1.0 + alpha_daily) ** ann_days - 1.0)
    return alpha_ann, beta


def run_backtest(panel: pd.DataFrame, weights: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    """中文说明。"""
    df = panel[["date", "asset", "close"]].copy().sort_values(["asset", "date"])
    df["asset_ret"] = df.groupby("asset")["close"].pct_change()
    ret_wide = df.pivot(index="date", columns="asset", values="asset_ret").sort_index().fillna(0.0)

    w = weights[["date", "asset", "weight"]].copy()
    w_wide = w.pivot(index="date", columns="asset", values="weight").sort_index().fillna(0.0)
    w_wide = w_wide.reindex(ret_wide.index).fillna(0.0)
    w_wide = _apply_weight_constraints(w_wide, panel=panel, config=config)

    # 当日信号从下一交易日收益开始执行，避免前视偏差。
    w_exec = w_wide.shift(1).fillna(0.0)
    gross_ret = (w_exec * ret_wide).sum(axis=1)

    turnover = (w_wide.diff().abs().sum(axis=1)).fillna(0.0)
    one_way_cost = (config.cost.commission_bps + config.cost.slippage_bps) / 10000.0
    cost = turnover * one_way_cost
    net_ret = gross_ret - cost
    equity = (1.0 + net_ret).cumprod()

    gross_exposure = w_wide.abs().sum(axis=1)
    net_exposure = w_wide.sum(axis=1)
    benchmark_ret = _benchmark_returns(panel, ret_wide=ret_wide, config=config)

    daily = pd.DataFrame(
        {
            "date": ret_wide.index,
            "gross_ret": gross_ret.values,
            "cost": cost.values,
            "net_ret": net_ret.values,
            "turnover": turnover.values,
            "gross_exposure": gross_exposure.values,
            "net_exposure": net_exposure.values,
            "benchmark_ret": benchmark_ret.values,
            "equity": equity.values,
        }
    )

    ann = config.cost.annualization_days
    ann_ret = float((1.0 + net_ret.mean()) ** ann - 1.0)
    ann_vol = float(net_ret.std(ddof=0) * np.sqrt(ann))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    mdd = _max_drawdown(equity)

    downside = net_ret[net_ret < 0]
    downside_vol = float(downside.std(ddof=0) * np.sqrt(ann)) if len(downside) > 0 else float("nan")
    sortino = ann_ret / downside_vol if downside_vol and np.isfinite(downside_vol) and downside_vol > 0 else np.nan
    calmar = ann_ret / abs(mdd) if np.isfinite(mdd) and mdd < 0 else np.nan
    alpha_ann, beta = _alpha_beta(net_ret, benchmark_ret, ann_days=ann)

    metrics = pd.DataFrame(
        {
            "cum_return": [float(equity.iloc[-1] - 1.0)],
            "ann_return": [ann_ret],
            "ann_vol": [ann_vol],
            "sharpe": [float(sharpe)],
            "sortino": [float(sortino)],
            "calmar": [float(calmar)],
            "max_drawdown": [mdd],
            "avg_turnover": [float(turnover.mean())],
            "turnover_p95": [float(turnover.quantile(0.95))],
            "avg_gross_exposure": [float(gross_exposure.mean())],
            "avg_net_exposure": [float(net_exposure.mean())],
            "alpha_ann": [alpha_ann],
            "beta": [beta],
        }
    )
    return BacktestResult(daily=daily, metrics=metrics)
