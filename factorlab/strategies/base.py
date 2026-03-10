"""策略接口定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True, frozen=True)
class StrategyDefinition:
    """策略定义与治理元数据。"""

    name: str
    family: str = "custom"
    description: str = ""
    constraints: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    implementation: str = ""
    origin: str = "builtin"


@dataclass(slots=True)
class Strategy(ABC):
    """策略基类，约束权重生成接口。"""

    name: str

    @abstractmethod
    def generate_weights(self, score_df: pd.DataFrame) -> pd.DataFrame:
        """根据打分面板生成组合权重。"""
        raise NotImplementedError
