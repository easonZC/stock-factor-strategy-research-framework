"""Factor factory and registration helpers."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from ssf.factors.base import Factor
from ssf.factors.simple import LiquidityShockFactor, MomentumFactor, SizeFactor, VolatilityFactor


def default_factor_registry() -> dict[str, Callable[[], Factor]]:
    """Built-in factor constructors."""
    return {
        "momentum_20": lambda: MomentumFactor(name="momentum_20", lookback=20),
        "volatility_20": lambda: VolatilityFactor(name="volatility_20", window=20),
        "liquidity_shock": lambda: LiquidityShockFactor(name="liquidity_shock", window=20),
        "size": lambda: SizeFactor(name="size"),
    }


def apply_factors(panel: pd.DataFrame, factor_names: list[str], inplace: bool = True) -> pd.DataFrame:
    """Compute configured factor names and append columns to panel.

    Args:
        panel: Input panel dataframe.
        factor_names: Factor names to compute.
        inplace: If True, mutate input dataframe to avoid large memory copies.
    """
    out = panel if inplace else panel.copy()
    reg = default_factor_registry()
    for name in factor_names:
        if name not in reg:
            raise KeyError(f"Unknown factor '{name}'. Available: {list(reg)}")
        factor = reg[name]()
        out[factor.name] = factor.compute(out)
    return out
