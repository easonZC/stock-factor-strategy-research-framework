"""模块说明。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factorlab.strategies.base import Strategy
from factorlab.strategies.implementations import _normalize_long


@dataclass(slots=True)
class MeanVarianceOptimizerStrategy(Strategy):
    """中文说明。"""

    risk_aversion: float = 5.0
    long_only: bool = False
    gross_target: float = 1.0
    net_target: float = 0.0
    rebalance_every: int = 1
    max_weight: float | None = None

    def _optimize_long_only(self, scores: pd.Series) -> pd.Series:
        mu = scores.astype(float).clip(lower=0.0)
        if float(mu.sum()) <= 0:
            mu = pd.Series(1.0, index=scores.index, dtype=float)
        w = _normalize_long(mu, max_weight=self.max_weight)
        gross = float(w.abs().sum())
        if gross > 0 and self.gross_target > 0:
            w = w * (float(self.gross_target) / gross)
        return w

    def _optimize_long_short(self, scores: pd.Series) -> pd.Series:
        s = scores.astype(float)
        if s.std(ddof=0) > 0:
            mu = (s - s.mean()) / s.std(ddof=0)
        else:
            mu = s - s.mean()

        lam = max(float(self.risk_aversion), 1e-6)
        w = mu / lam
        w = w - float(w.mean()) + float(self.net_target) / max(len(w), 1)
        if self.max_weight is not None:
            cap = abs(float(self.max_weight))
            w = w.clip(lower=-cap, upper=cap)
        gross = float(w.abs().sum())
        if gross > 0 and self.gross_target > 0:
            w = w * (float(self.gross_target) / gross)
        return w

    def generate_weights(self, score_df: pd.DataFrame) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        prev: pd.Series | None = None
        dates = sorted(score_df["date"].dropna().unique())
        for i, dt in enumerate(dates):
            grp = score_df[score_df["date"] == dt].copy()
            if i % max(1, int(self.rebalance_every)) == 0 or prev is None:
                g = grp.dropna(subset=["score"]).copy()
                if g.empty:
                    continue
                g["asset"] = g["asset"].astype(str)
                scores = g.set_index("asset")["score"].astype(float)
                if self.long_only:
                    prev = self._optimize_long_only(scores)
                else:
                    prev = self._optimize_long_short(scores)

            if prev is None or prev.empty:
                continue
            available_assets = set(grp["asset"].astype(str))
            cur = prev[prev.index.isin(available_assets)]
            if cur.empty:
                continue
            chunk = pd.DataFrame({"date": dt, "asset": cur.index, "weight": cur.values})
            frames.append(chunk)
        if not frames:
            return pd.DataFrame(columns=["date", "asset", "weight"])
        return pd.concat(frames, ignore_index=True)

