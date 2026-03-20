"""可复用的股票因子与策略研究框架。"""

from .config import (
    AdapterConfig,
    BacktestConfig,
    CostConfig,
    NeutralizationConfig,
    ResearchConfig,
    SyntheticConfig,
)
from .runtime import OutputContext, RunContext

__all__ = [
    "AdapterConfig",
    "BacktestConfig",
    "CostConfig",
    "NeutralizationConfig",
    "OutputContext",
    "ResearchConfig",
    "RunContext",
    "SyntheticConfig",
]
