"""可复用的实战因子插件示例。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factorlab.factors.base import Factor


def _missing_required(panel: pd.DataFrame, required: list[str]) -> bool:
    return any(col not in panel.columns for col in required)


def _sorted(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.sort_values(["asset", "date"]).copy()


@dataclass(slots=True)
class TrendBreakoutFactor(Factor):
    """趋势突破强度：收盘价相对滚动高低区间的位置。"""

    lookback: int = 30

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        if _missing_required(panel, ["asset", "date", "close", "high", "low"]):
            return pd.Series(np.nan, index=panel.index, name=self.name)
        df = _sorted(panel)
        rolling_high = (
            df.groupby("asset")["high"]
            .rolling(self.lookback, min_periods=max(5, self.lookback // 3))
            .max()
            .reset_index(level=0, drop=True)
        )
        rolling_low = (
            df.groupby("asset")["low"]
            .rolling(self.lookback, min_periods=max(5, self.lookback // 3))
            .min()
            .reset_index(level=0, drop=True)
        )
        width = (rolling_high - rolling_low).replace(0.0, np.nan)
        signal = ((df["close"] - rolling_low) / width) - 0.5
        return signal.reindex(panel.index)


@dataclass(slots=True)
class VolumePricePressureFactor(Factor):
    """量价压力：短期收益与成交量脉冲的耦合。"""

    window: int = 15

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        if _missing_required(panel, ["asset", "date", "close", "volume"]):
            return pd.Series(np.nan, index=panel.index, name=self.name)
        df = _sorted(panel)
        ret_1 = df.groupby("asset")["close"].pct_change()
        amount = (df["close"].astype(float) * df["volume"].astype(float)).replace([np.inf, -np.inf], np.nan)
        amount_mean = (
            amount.groupby(df["asset"])
            .rolling(self.window, min_periods=max(5, self.window // 3))
            .mean()
            .reset_index(level=0, drop=True)
        )
        pulse = amount / amount_mean.replace(0.0, np.nan)
        signal = ret_1 * pulse
        return signal.reindex(panel.index)


@dataclass(slots=True)
class RangeReversalFactor(Factor):
    """区间反转：日内涨跌在波动区间中的归一化反转强度。"""

    window: int = 5

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        if _missing_required(panel, ["asset", "date", "open", "high", "low", "close"]):
            return pd.Series(np.nan, index=panel.index, name=self.name)
        df = _sorted(panel)
        intraday_move = (df["close"] - df["open"]).astype(float)
        intraday_range = (df["high"] - df["low"]).astype(float).replace(0.0, np.nan)
        raw = -(intraday_move / intraday_range)
        signal = (
            raw.groupby(df["asset"])
            .rolling(self.window, min_periods=max(3, self.window // 2))
            .mean()
            .reset_index(level=0, drop=True)
        )
        return signal.reindex(panel.index)


@dataclass(slots=True)
class VolatilityRegimeShiftFactor(Factor):
    """波动率状态切换：短波动与长波动比值。"""

    short_window: int = 10
    long_window: int = 40

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        if _missing_required(panel, ["asset", "date", "close"]):
            return pd.Series(np.nan, index=panel.index, name=self.name)
        df = _sorted(panel)
        ret = df.groupby("asset")["close"].pct_change()
        vol_short = (
            ret.groupby(df["asset"])
            .rolling(self.short_window, min_periods=max(5, self.short_window // 2))
            .std()
            .reset_index(level=0, drop=True)
        )
        vol_long = (
            ret.groupby(df["asset"])
            .rolling(self.long_window, min_periods=max(10, self.long_window // 3))
            .std()
            .reset_index(level=0, drop=True)
        )
        signal = vol_short / vol_long.replace(0.0, np.nan)
        return signal.reindex(panel.index)


def get_factor_registry():
    """返回插件注册表。"""
    return {
        "trend_breakout_30": lambda: TrendBreakoutFactor(name="trend_breakout_30", lookback=30),
        "volume_price_pressure_15": lambda: VolumePricePressureFactor(name="volume_price_pressure_15", window=15),
        "range_reversal_5": lambda: RangeReversalFactor(name="range_reversal_5", window=5),
        "volatility_regime_shift_10_40": lambda: VolatilityRegimeShiftFactor(
            name="volatility_regime_shift_10_40",
            short_window=10,
            long_window=40,
        ),
    }

