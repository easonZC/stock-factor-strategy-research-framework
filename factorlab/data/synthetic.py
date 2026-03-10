"""合成面板数据生成工具。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.config import SyntheticConfig


def _cs_zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    std = float(np.nanstd(arr))
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(arr)
    return (arr - float(np.nanmean(arr))) / std


def generate_synthetic_panel(config: SyntheticConfig) -> pd.DataFrame:
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


def generate_model_factor_benchmark_panel(
    config: SyntheticConfig,
    tier: str = "engineering_demo",
) -> pd.DataFrame:
    """生成模型因子基准面板。

    `engineering_demo`:
        高信噪比、强非线性，用于工程回归与 NN 因子能力展示。
    `research_realistic`:
        更弱、更接近真实研究量级的截面信号，用于现实感更强的基准。
    """
    if tier not in {"engineering_demo", "research_realistic"}:
        raise ValueError(f"Unsupported benchmark tier: {tier}")
    rng = np.random.default_rng(int(config.seed))
    n_assets, n_days = int(config.n_assets), int(config.n_days)
    dates = pd.bdate_range(config.start_date, periods=n_days)
    assets = [f"A{i:04d}" for i in range(n_assets)]
    industry = rng.choice(np.array(["Tech", "Finance", "Industry", "Consumer"]), size=n_assets)
    shares = np.exp(rng.normal(18.7, 0.8, size=n_assets))
    base_price = rng.uniform(12.0, 90.0, size=n_assets)
    base_volume = np.exp(rng.normal(13.4, 0.35, size=n_assets))

    close = np.zeros((n_days, n_assets), dtype=float)
    open_px = np.zeros_like(close)
    high = np.zeros_like(close)
    low = np.zeros_like(close)
    volume = np.zeros_like(close)
    close[0] = base_price
    open_px[0] = base_price * (1.0 + rng.normal(0.0, 0.001, size=n_assets))
    spread0 = np.abs(rng.normal(0.003, 0.001, size=n_assets))
    high[0] = np.maximum(open_px[0], close[0]) * (1.0 + spread0)
    low[0] = np.minimum(open_px[0], close[0]) * (1.0 - spread0)
    volume[0] = base_volume

    alpha = np.zeros(n_assets, dtype=float)
    market = rng.normal(0.0, 0.0018, size=n_days)
    regime = np.sin(np.linspace(0.0, 5.0 * np.pi, n_days)) * 0.0005

    for t in range(1, n_days):
        start = max(0, t - 20)
        mom = np.zeros(n_assets, dtype=float)
        vol = np.zeros(n_assets, dtype=float)
        liq = np.zeros(n_assets, dtype=float)
        size = -np.log1p(close[t - 1] * shares)
        for idx in range(n_assets):
            hist = close[start:t, idx]
            if len(hist) >= 5:
                mom[idx] = close[t - 1, idx] / hist[0] - 1.0
            ret_hist = pd.Series(hist).pct_change().dropna().to_numpy()
            if len(ret_hist) >= 5:
                vol[idx] = -float(np.std(ret_hist, ddof=1))
            avg_vol = float(np.mean(volume[start:t, idx])) if (t - start) >= 5 else float(base_volume[idx])
            liq[idx] = volume[t - 1, idx] / max(avg_vol, 1.0)

        mom_z = _cs_zscore(mom)
        vol_z = _cs_zscore(vol)
        liq_z = _cs_zscore(np.log1p(liq))
        size_z = _cs_zscore(size)
        raw_signal = (
            0.8 * np.tanh(2.0 * mom_z * liq_z)
            + 0.7 * np.sin(2.2 * mom_z)
            + 0.5 * np.sign(liq_z) * np.sqrt(np.abs(liq_z) + 1e-6)
            + 0.35 * np.where(size_z > 0.0, size_z**2, -0.5 * size_z**2)
            - 0.3 * np.abs(vol_z)
        )
        if tier == "research_realistic":
            signal = 0.45 * raw_signal
            alpha = 0.72 * alpha + 0.00055 * signal + rng.normal(0.0, 0.00065, size=n_assets)
            ret = market[t] + regime[t] + alpha + rng.normal(0.0, 0.0085, size=n_assets)
            volume_noise = 0.18
            volume_signal_scale = 0.08
        else:
            signal = raw_signal
            alpha = 0.68 * alpha + 0.0026 * signal + rng.normal(0.0, 0.00045, size=n_assets)
            ret = market[t] + regime[t] + alpha + rng.normal(0.0, 0.0020, size=n_assets)
            volume_noise = 0.08
            volume_signal_scale = 0.22
        close[t] = np.maximum(1.0, close[t - 1] * (1.0 + ret))
        volume[t] = np.maximum(
            1000.0,
            base_volume
            * np.exp(
                0.18 * np.abs(ret)
                + volume_signal_scale * np.maximum(signal, 0.0)
                + rng.normal(0.0, volume_noise, size=n_assets)
            ),
        )
        open_px[t] = close[t - 1] * (1.0 + rng.normal(0.0, 0.0014, size=n_assets))
        spread = np.abs(rng.normal(0.0038, 0.001, size=n_assets))
        high[t] = np.maximum(open_px[t], close[t]) * (1.0 + spread)
        low[t] = np.minimum(open_px[t], close[t]) * (1.0 - spread)

    return pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "asset": asset,
                    "open": open_px[:, idx],
                    "high": high[:, idx],
                    "low": low[:, idx],
                    "close": close[:, idx],
                    "volume": volume[:, idx],
                    "mkt_cap": close[:, idx] * shares[idx],
                    "industry": industry[idx],
                }
            )
            for idx, asset in enumerate(assets)
        ],
        ignore_index=True,
    )
