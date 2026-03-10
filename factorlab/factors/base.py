"""因子接口定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True, frozen=True)
class FactorDefinition:
    """因子定义与治理元数据。"""

    name: str
    family: str = "custom"
    description: str = ""
    formula: str = ""
    required_columns: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    implementation: str = ""
    origin: str = "builtin"


@dataclass(slots=True)
class Factor(ABC):
    """因子基类，约束统一计算接口。"""

    name: str

    @abstractmethod
    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """基于输入面板计算因子值。"""
        raise NotImplementedError


@dataclass(slots=True)
class PanelFactorResult:
    """因子计算结果封装。"""

    name: str
    values: pd.Series
