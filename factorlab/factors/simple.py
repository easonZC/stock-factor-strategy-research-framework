"""用于演示与基线研究的内置示例因子。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factorlab.factors.base import Factor


@dataclass(slots=True)
class MomentumFactor(Factor):
    FACTOR_FAMILY = "price_trend"
    FACTOR_DESCRIPTION = "Classic lookback momentum measured as multi-day percentage return."
    FACTOR_FORMULA = "close_t / close_{t-lookback} - 1"
    REQUIRED_COLUMNS = ("close",)
    FACTOR_TAGS = ("price", "trend", "baseline")

    lookback: int = 20

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        df = panel.sort_values(["asset", "date"])
        ret = df.groupby("asset")["close"].pct_change(self.lookback)
        return ret.reindex(panel.index)


@dataclass(slots=True)
class VolatilityFactor(Factor):
    FACTOR_FAMILY = "risk"
    FACTOR_DESCRIPTION = "Negative rolling return volatility so lower realized volatility ranks higher."
    FACTOR_FORMULA = "- rolling_std(pct_change(close, 1), window)"
    REQUIRED_COLUMNS = ("close",)
    FACTOR_TAGS = ("price", "volatility", "defensive")

    window: int = 20

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        df = panel.sort_values(["asset", "date"]).copy()
        ret = df.groupby("asset")["close"].pct_change()
        vol = ret.groupby(df["asset"]).rolling(self.window, min_periods=max(5, self.window // 4)).std().reset_index(level=0, drop=True)
        return (-vol).reindex(panel.index)


@dataclass(slots=True)
class LiquidityShockFactor(Factor):
    FACTOR_FAMILY = "liquidity"
    FACTOR_DESCRIPTION = "Relative volume spike versus rolling average volume."
    FACTOR_FORMULA = "volume / rolling_mean(volume, window)"
    REQUIRED_COLUMNS = ("volume",)
    FACTOR_TAGS = ("volume", "liquidity", "flow")

    window: int = 20

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        df = panel.sort_values(["asset", "date"]).copy()
        vol = df["volume"].replace(0, np.nan)
        mean_vol = vol.groupby(df["asset"]).rolling(self.window, min_periods=max(5, self.window // 4)).mean().reset_index(level=0, drop=True)
        ratio = vol / mean_vol
        return ratio.reindex(panel.index)


@dataclass(slots=True)
class SizeFactor(Factor):
    FACTOR_FAMILY = "size"
    FACTOR_DESCRIPTION = "Negative log market capitalization as a small-cap tilt proxy."
    FACTOR_FORMULA = "-log(1 + mkt_cap)"
    REQUIRED_COLUMNS = ("mkt_cap",)
    FACTOR_TAGS = ("size", "cross_sectional", "baseline")

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        return -np.log1p(panel["mkt_cap"].astype(float))


@dataclass(slots=True)
class VolumePricePressureFactor(Factor):
    FACTOR_FAMILY = "price_volume"
    FACTOR_DESCRIPTION = (
        "Signed rolling price pressure using 1-day returns scaled by relative volume shock. "
        "Useful as a simple microstructure-aware factor for baseline research."
    )
    FACTOR_FORMULA = (
        "rolling_mean(sign(ret_1) * abs(ret_1) * log(1 + volume / rolling_mean(volume, window)), window)"
    )
    REQUIRED_COLUMNS = ("close", "volume")
    FACTOR_TAGS = ("price", "volume", "flow", "microstructure")

    window: int = 20

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        df = panel.sort_values(["asset", "date"]).copy()
        ret_1 = df.groupby("asset")["close"].pct_change()
        volume = df["volume"].astype(float).replace(0.0, np.nan)
        avg_volume = (
            volume.groupby(df["asset"])
            .rolling(self.window, min_periods=max(5, self.window // 4))
            .mean()
            .reset_index(level=0, drop=True)
        )
        rel_volume = (volume / avg_volume).replace([np.inf, -np.inf], np.nan)
        signed_pressure = np.sign(ret_1) * np.abs(ret_1) * np.log1p(rel_volume.clip(lower=0.0))
        factor = (
            signed_pressure.groupby(df["asset"])
            .rolling(self.window, min_periods=max(5, self.window // 4))
            .mean()
            .reset_index(level=0, drop=True)
        )
        return factor.reindex(panel.index)
