"""因子接口定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class Factor(ABC):
    """中文说明。"""

    name: str

    @abstractmethod
    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """中文说明。"""


@dataclass(slots=True)
class PanelFactorResult:
    """中文说明。"""

    name: str
    values: pd.Series
