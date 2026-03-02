"""研究流水线能力聚合导出。"""

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
