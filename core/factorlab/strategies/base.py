"""策略接口定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class Strategy(ABC):
    """中文说明。"""

    name: str

    @abstractmethod
    def generate_weights(self, score_df: pd.DataFrame) -> pd.DataFrame:
        """中文说明。"""
