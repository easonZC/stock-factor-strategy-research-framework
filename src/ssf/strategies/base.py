"""Strategy interface definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class Strategy(ABC):
    """Abstract strategy interface producing date/asset weights."""

    name: str

    @abstractmethod
    def generate_weights(self, score_df: pd.DataFrame) -> pd.DataFrame:
        """Input columns: date, asset, score. Output columns: date, asset, weight."""
