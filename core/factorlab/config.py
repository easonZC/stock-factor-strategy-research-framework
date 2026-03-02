"""Dataclass-based configuration objects for framework modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


NeutralizationMode = Literal["none", "size", "industry", "both"]
MissingReturnPolicy = Literal["zero"]
BenchmarkMode = Literal["none", "cross_sectional_mean", "panel_column"]
CSStandardizeMode = Literal["cs_zscore", "cs_rank", "cs_robust_zscore", "none"]
MissingFactorPolicy = Literal["drop", "fill_zero", "ffill_by_asset", "cs_median_by_date", "keep"]
PreprocessStep = Literal["winsorize", "standardize", "neutralize"]


@dataclass(slots=True)
class CostConfig:
    """Transaction-cost assumptions in basis points."""

    commission_bps: float = 1.0
    slippage_bps: float = 1.0
    annualization_days: int = 252


@dataclass(slots=True)
class BacktestConfig:
    """Backtest engine parameters.

    `long_short_leverage` is interpreted as target gross exposure after
    cross-sectional normalization on each date.
    """

    cost: CostConfig = field(default_factory=CostConfig)
    long_short_leverage: float = 1.0
    validate_inputs: bool = True
    missing_return_policy: MissingReturnPolicy = "zero"
    execution_price_col: str = "close"
    execution_delay_days: int = 1
    is_tradable_col: str = "is_tradable"
    can_buy_col: str = "can_buy"
    can_sell_col: str = "can_sell"
    volume_col: str = "volume"
    enable_tradability_constraints: bool = False
    max_participation_rate: float | None = None
    benchmark_mode: BenchmarkMode = "none"
    benchmark_return_col: str = "benchmark_ret"


@dataclass(slots=True)
class UniverseFilterConfig:
    """Tradable-universe filters applied on panel rows."""

    min_close: float = 0.0
    min_history_days: int = 1
    min_median_dollar_volume: float = 0.0
    liquidity_lookback: int = 20


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
    standardization: CSStandardizeMode = "cs_zscore"
    winsorize_enabled: bool = True
    winsorize_method: Literal["quantile", "mad"] = "quantile"
    lower_q: float = 0.01
    upper_q: float = 0.99
    mad_scale: float = 5.0
    missing_policy: MissingFactorPolicy = "drop"
    preprocess_steps: list[PreprocessStep] = field(
        default_factory=lambda: ["winsorize", "standardize", "neutralize"]
    )
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
