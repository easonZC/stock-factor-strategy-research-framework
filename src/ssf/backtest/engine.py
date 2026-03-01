"""Backtest engine for long-only and long-short weight streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ssf.config import BacktestConfig


@dataclass(slots=True)
class BacktestResult:
    """Container for backtest outputs."""

    daily: pd.DataFrame
    metrics: pd.DataFrame


def _max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    return float(drawdown.min())


def run_backtest(panel: pd.DataFrame, weights: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    """Run a simple daily backtest with turnover-based trading costs."""
    df = panel[["date", "asset", "close"]].copy().sort_values(["asset", "date"])
    df["asset_ret"] = df.groupby("asset")["close"].pct_change()
    ret_wide = df.pivot(index="date", columns="asset", values="asset_ret").sort_index().fillna(0.0)

    w = weights[["date", "asset", "weight"]].copy()
    w_wide = w.pivot(index="date", columns="asset", values="weight").sort_index().fillna(0.0)
    w_wide = w_wide.reindex(ret_wide.index).fillna(0.0)

    # Signal at t is executed from t+1 return to avoid look-ahead.
    w_exec = w_wide.shift(1).fillna(0.0)
    gross_ret = (w_exec * ret_wide).sum(axis=1)

    turnover = (w_wide.diff().abs().sum(axis=1)).fillna(0.0)
    one_way_cost = (config.cost.commission_bps + config.cost.slippage_bps) / 10000.0
    cost = turnover * one_way_cost
    net_ret = gross_ret - cost
    equity = (1.0 + net_ret).cumprod()

    daily = pd.DataFrame(
        {
            "date": ret_wide.index,
            "gross_ret": gross_ret.values,
            "cost": cost.values,
            "net_ret": net_ret.values,
            "turnover": turnover.values,
            "equity": equity.values,
        }
    )

    ann = config.cost.annualization_days
    ann_ret = float((1.0 + net_ret.mean()) ** ann - 1.0)
    ann_vol = float(net_ret.std(ddof=0) * np.sqrt(ann))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    mdd = _max_drawdown(equity)

    metrics = pd.DataFrame(
        {
            "cum_return": [float(equity.iloc[-1] - 1.0)],
            "ann_return": [ann_ret],
            "ann_vol": [ann_vol],
            "sharpe": [float(sharpe)],
            "max_drawdown": [mdd],
            "avg_turnover": [float(turnover.mean())],
        }
    )
    return BacktestResult(daily=daily, metrics=metrics)
