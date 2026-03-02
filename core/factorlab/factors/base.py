"""Factor interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class Factor(ABC):
    """Abstract factor interface."""

    name: str

    @abstractmethod
    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Return factor values aligned to panel index."""


@dataclass(slots=True)
class PanelFactorResult:
    """Container for a factor column generated on a panel."""

    name: str
    values: pd.Series
