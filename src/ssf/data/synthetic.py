"""Synthetic panel generation for clean-machine demos and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ssf.config import SyntheticConfig


def generate_synthetic_panel(config: SyntheticConfig) -> pd.DataFrame:
    """Generate a leakage-safe synthetic OHLCV panel with size/industry fields."""
    rng = np.random.default_rng(int(config.seed))
    dates = pd.bdate_range(config.start_date, periods=int(config.n_days))
    assets = [f"A{i:04d}" for i in range(int(config.n_assets))]
    industries = np.array(["Tech", "Finance", "Industry", "Consumer", "Healthcare"])

    market_factor = rng.normal(0.0002, 0.008, size=len(dates))
    rows: list[pd.DataFrame] = []

    for asset in assets:
        beta = rng.normal(1.0, 0.2)
        idio_vol = float(np.clip(rng.normal(0.012, 0.003), 0.004, 0.035))
        trend = rng.normal(0.00015, 0.00025)
        base_price = float(rng.uniform(8.0, 120.0))
        shares = float(rng.uniform(5e7, 2e9))
        industry = str(rng.choice(industries))

        noise = rng.normal(0.0, idio_vol, size=len(dates))
        ret = trend + beta * market_factor + noise
        close = base_price * np.cumprod(1.0 + ret)
        close = np.maximum(close, 1.0)

        open_noise = rng.normal(0.0, 0.0035, size=len(dates))
        open_px = close * (1.0 + open_noise)
        spread = np.abs(rng.normal(0.004, 0.0015, size=len(dates)))
        high = np.maximum(open_px, close) * (1.0 + spread)
        low = np.minimum(open_px, close) * (1.0 - spread)

        base_vol = float(rng.uniform(3e5, 4e6))
        vol_shock = rng.lognormal(mean=0.0, sigma=0.35, size=len(dates))
        volume = np.maximum(base_vol * vol_shock, 1000.0)

        mkt_cap = close * shares

        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "asset": asset,
                    "open": open_px.astype(float),
                    "high": high.astype(float),
                    "low": low.astype(float),
                    "close": close.astype(float),
                    "volume": volume.astype(float),
                    "mkt_cap": mkt_cap.astype(float),
                    "industry": industry,
                }
            )
        )

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)
    return panel
