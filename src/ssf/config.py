"""Dataclass-based configuration objects for framework modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


NeutralizationMode = Literal["none", "size", "industry", "both"]


@dataclass(slots=True)
class CostConfig:
    """Transaction-cost assumptions in basis points."""

    commission_bps: float = 1.0
    slippage_bps: float = 1.0
    annualization_days: int = 252


@dataclass(slots=True)
class BacktestConfig:
    """Backtest engine parameters."""

    cost: CostConfig = field(default_factory=CostConfig)
    long_short_leverage: float = 1.0


@dataclass(slots=True)
class NeutralizationConfig:
    """Cross-sectional neutralization settings."""

    mode: NeutralizationMode = "both"
    size_col: str = "mkt_cap"
    industry_col: str = "industry"


@dataclass(slots=True)
class ResearchConfig:
    """Factor research pipeline settings."""

    horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    quantiles: int = 5
    ic_rolling_window: int = 20
    winsorize_method: Literal["quantile", "mad"] = "quantile"
    lower_q: float = 0.01
    upper_q: float = 0.99
    mad_scale: float = 5.0
    missing_policy: Literal["drop"] = "drop"
    neutralization: NeutralizationConfig = field(default_factory=NeutralizationConfig)


@dataclass(slots=True)
class SyntheticConfig:
    """Synthetic panel generator settings."""

    n_assets: int = 40
    n_days: int = 260
    seed: int = 7
    start_date: str = "2021-01-01"


@dataclass(slots=True)
class AdapterConfig:
    """Sina CSV adapter settings."""

    data_dir: str
    required_cols: tuple[str, ...] = ("date", "close")
    min_rows_per_asset: int = 30
