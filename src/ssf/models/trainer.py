"""Training utility for model-based factors using synthetic panel data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ssf.config import SyntheticConfig
from ssf.data.synthetic import generate_synthetic_panel
from ssf.factors.simple import LiquidityShockFactor, MomentumFactor, VolatilityFactor
from ssf.models.registry import ModelRegistry
from ssf.research.forward_returns import add_forward_returns


def train_model_factor(
    model_name: str = "ridge",
    model_out: str = "artifacts/models/model_factor.joblib",
    synthetic_config: SyntheticConfig | None = None,
) -> Path:
    """Train and persist a simple predictive model for factor inference."""
    cfg = synthetic_config or SyntheticConfig(n_assets=30, n_days=260, seed=17)
    panel = generate_synthetic_panel(cfg)

    factors = [
        MomentumFactor(name="momentum_20", lookback=20),
        VolatilityFactor(name="volatility_20", window=20),
        LiquidityShockFactor(name="liquidity_shock", window=20),
    ]
    for fac in factors:
        panel[fac.name] = fac.compute(panel)

    panel = add_forward_returns(panel, horizons=[5])
    panel = panel.dropna(subset=["momentum_20", "volatility_20", "liquidity_shock", "fwd_ret_5"])

    train = panel.sort_values("date").copy()
    split = int(len(train) * 0.8)
    x_train = train[["momentum_20", "volatility_20", "liquidity_shock"]].iloc[:split]
    y_train = train["fwd_ret_5"].iloc[:split]

    model = ModelRegistry.create(model_name)
    model.fit(x_train, y_train)

    out = Path(model_out)
    return ModelRegistry.save(model, out, model_name=model_name)
