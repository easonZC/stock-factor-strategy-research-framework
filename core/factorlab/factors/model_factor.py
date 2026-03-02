"""模型因子实现。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from factorlab.factors.base import Factor
from factorlab.models.registry import ModelRegistry


@dataclass(slots=True)
class ModelFactor(Factor):

    model_name: str
    model_path: str
    feature_cols: list[str] = field(default_factory=lambda: ["momentum_20", "volatility_20", "liquidity_shock"])

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        model = ModelRegistry.load(Path(self.model_path), self.model_name)
        feats = panel[self.feature_cols].copy()
        preds = model.predict(feats.fillna(0.0))
        return pd.Series(preds, index=panel.index, name=self.name)
