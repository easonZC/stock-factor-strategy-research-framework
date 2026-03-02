"""模块说明。"""

from .forward_returns import add_forward_returns
from .pipeline import FactorResearchPipeline
from .ts_pipeline import TSResearchConfig, TimeSeriesFactorResearchPipeline
from .walkforward import WalkForwardConfig, WalkForwardResult, run_walkforward_strategy

__all__ = [
    "FactorResearchPipeline",
    "TimeSeriesFactorResearchPipeline",
    "TSResearchConfig",
    "add_forward_returns",
    "WalkForwardConfig",
    "WalkForwardResult",
    "run_walkforward_strategy",
]
