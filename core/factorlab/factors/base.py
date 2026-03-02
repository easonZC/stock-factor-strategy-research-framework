"""因子接口定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


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
