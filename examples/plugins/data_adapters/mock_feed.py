"""Example custom data adapter plugin.

Auto-discovered adapter name from function:
  prepare_mock_feed_panel -> "mock_feed"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.config import AdapterConfig


def prepare_mock_feed_panel(config: AdapterConfig) -> pd.DataFrame:
    """Generate a synthetic-like custom feed for plugin demonstration."""
    assets = [f"MOCK{i:02d}" for i in range(1, 7)]
    dates = pd.date_range(config.start_date or "2022-01-03", periods=220, freq="B")
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(123)

    for asset in assets:
        drift = float(rng.normal(0.0005, 0.0002))
        vol = float(rng.uniform(0.005, 0.02))
        price = 50.0 + float(rng.uniform(0, 30))
        for dt in dates:
            ret = float(rng.normal(drift, vol))
            open_px = price
            close_px = max(1.0, open_px * (1.0 + ret))
            high_px = max(open_px, close_px) * (1.0 + float(rng.uniform(0.0, 0.01)))
            low_px = min(open_px, close_px) * (1.0 - float(rng.uniform(0.0, 0.01)))
            volume = float(rng.integers(100_000, 2_000_000))
            rows.append(
                {
                    "date": dt,
                    "asset": asset,
                    "open": open_px,
                    "high": high_px,
                    "low": low_px,
                    "close": close_px,
                    "volume": volume,
                    "mkt_cap": close_px * volume * 100.0,
                    "industry": "Demo",
                }
            )
            price = close_px

    panel = pd.DataFrame(rows).sort_values(["date", "asset"]).reset_index(drop=True)
    if config.min_rows_per_asset > 0:
        counts = panel.groupby("asset")["date"].count()
        keep = counts[counts >= int(config.min_rows_per_asset)].index
        panel = panel[panel["asset"].isin(set(keep))].copy()
    return panel.reset_index(drop=True)

