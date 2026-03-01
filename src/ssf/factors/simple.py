"""Built-in example factors for demo and baseline research."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ssf.factors.base import Factor


@dataclass(slots=True)
class MomentumFactor(Factor):
    """Price momentum over a lookback window."""

    lookback: int = 20

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        df = panel.sort_values(["asset", "date"])
        ret = df.groupby("asset")["close"].pct_change(self.lookback)
        return ret.reindex(panel.index)


@dataclass(slots=True)
class VolatilityFactor(Factor):
    """Rolling return volatility factor."""

    window: int = 20

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        df = panel.sort_values(["asset", "date"]).copy()
        ret = df.groupby("asset")["close"].pct_change()
        vol = ret.groupby(df["asset"]).rolling(self.window, min_periods=max(5, self.window // 4)).std().reset_index(level=0, drop=True)
        return (-vol).reindex(panel.index)


@dataclass(slots=True)
class LiquidityShockFactor(Factor):
    """Volume-vs-history shock proxy factor."""

    window: int = 20

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        df = panel.sort_values(["asset", "date"]).copy()
        vol = df["volume"].replace(0, np.nan)
        mean_vol = vol.groupby(df["asset"]).rolling(self.window, min_periods=max(5, self.window // 4)).mean().reset_index(level=0, drop=True)
        ratio = vol / mean_vol
        return ratio.reindex(panel.index)


@dataclass(slots=True)
class SizeFactor(Factor):
    """Log market-cap factor (small-cap tilt as negative sign)."""

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        return -np.log1p(panel["mkt_cap"].astype(float))
