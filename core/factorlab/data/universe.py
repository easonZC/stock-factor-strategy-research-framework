"""Tradable-universe filtering utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factorlab.config import UniverseFilterConfig


@dataclass(slots=True)
class UniverseFilterReport:
    """Summary of universe-filter effects."""

    rows_before: int
    rows_after: int
    assets_before: int
    assets_after: int
    dates_before: int
    dates_after: int
    removed_ratio: float


def apply_universe_filter(
    panel: pd.DataFrame,
    config: UniverseFilterConfig,
) -> tuple[pd.DataFrame, UniverseFilterReport]:
    """Apply conservative tradability filters without future information."""
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["asset", "date"]).reset_index(drop=True)

    rows_before = int(len(df))
    assets_before = int(df["asset"].nunique()) if "asset" in df.columns else 0
    dates_before = int(df["date"].nunique()) if "date" in df.columns else 0

    keep = pd.Series(True, index=df.index)

    if "close" in df.columns and float(config.min_close) > 0:
        close = pd.to_numeric(df["close"], errors="coerce")
        keep &= close >= float(config.min_close)

    if int(config.min_history_days) > 1 and "asset" in df.columns:
        history_n = df.groupby("asset").cumcount() + 1
        keep &= history_n >= int(config.min_history_days)

    min_dv = float(config.min_median_dollar_volume)
    if min_dv > 0 and {"close", "volume", "asset"}.issubset(df.columns):
        close = pd.to_numeric(df["close"], errors="coerce")
        volume = pd.to_numeric(df["volume"], errors="coerce")
        dollar_volume = close * volume
        lookback = max(2, int(config.liquidity_lookback))
        rolling_median = (
            dollar_volume.groupby(df["asset"])
            .rolling(window=lookback, min_periods=max(2, lookback // 3))
            .median()
            .reset_index(level=0, drop=True)
        )
        keep &= rolling_median >= min_dv

    filtered = df.loc[keep].copy()
    filtered = filtered.sort_values(["date", "asset"]).reset_index(drop=True)

    rows_after = int(len(filtered))
    assets_after = int(filtered["asset"].nunique()) if "asset" in filtered.columns else 0
    dates_after = int(filtered["date"].nunique()) if "date" in filtered.columns else 0
    removed_ratio = float(0.0 if rows_before == 0 else (rows_before - rows_after) / rows_before)

    report = UniverseFilterReport(
        rows_before=rows_before,
        rows_after=rows_after,
        assets_before=assets_before,
        assets_after=assets_after,
        dates_before=dates_before,
        dates_after=dates_after,
        removed_ratio=removed_ratio,
    )
    return filtered, report
