"""框架级数据类配置定义。"""

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
    """交易成本假设（单位：bp）。"""

    commission_bps: float = 1.0
    slippage_bps: float = 1.0
    annualization_days: int = 252


@dataclass(slots=True)
class BacktestConfig:
    """回测引擎参数。

    `long_short_leverage` 表示在逐日截面归一化后目标总敞口。
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
    # 风险约束
    max_turnover: float | None = None
    max_abs_weight: float | None = None
    max_gross_exposure: float | None = None
    max_net_exposure: float | None = None
    enforce_industry_neutral: bool = False
    industry_col: str = "industry"


@dataclass(slots=True)
class UniverseFilterConfig:
    """可交易股票池过滤配置。"""

    min_close: float = 0.0
    min_history_days: int = 1
    min_median_dollar_volume: float = 0.0
    liquidity_lookback: int = 20


@dataclass(slots=True)
class NeutralizationConfig:
    """截面中性化配置。"""

    mode: NeutralizationMode = "both"
    size_col: str = "mkt_cap"
    industry_col: str = "industry"


@dataclass(slots=True)
class ResearchConfig:
    """因子研究流水线配置。"""

    horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    quantiles: int = 5
    ic_rolling_window: int = 20
    annualization_days: int = 252
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
    """合成面板数据生成配置。"""

    n_assets: int = 40
    n_days: int = 260
    seed: int = 7
    start_date: str = "2021-01-01"


@dataclass(slots=True)
class AdapterConfig:
    """外部数据适配器配置。"""

    data_dir: str = ""
    required_cols: tuple[str, ...] = ("date", "close")
    min_rows_per_asset: int = 30
    symbols: tuple[str, ...] = ()
    start_date: str | None = None
    end_date: str | None = None
    request_timeout_sec: int = 20
