"""Built-in strategy implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ssf.strategies.base import Strategy


@dataclass(slots=True)
class TopKLongStrategy(Strategy):
    """Equal-weight long-only top-k strategy."""

    top_k: int = 20

    def generate_weights(self, score_df: pd.DataFrame) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for dt, grp in score_df.groupby("date"):
            g = grp.dropna(subset=["score"]).sort_values("score", ascending=False).head(self.top_k)
            if g.empty:
                continue
            g = g[["date", "asset"]].copy()
            g["weight"] = 1.0 / len(g)
            frames.append(g)
        if not frames:
            return pd.DataFrame(columns=["date", "asset", "weight"])
        return pd.concat(frames, ignore_index=True)


@dataclass(slots=True)
class LongShortQuantileStrategy(Strategy):
    """Long top quantile, short bottom quantile with equal absolute weights."""

    quantile: float = 0.2

    def generate_weights(self, score_df: pd.DataFrame) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for dt, grp in score_df.groupby("date"):
            g = grp.dropna(subset=["score"]).sort_values("score", ascending=False)
            if g.empty:
                continue
            k = max(1, int(np.floor(len(g) * self.quantile)))
            long_leg = g.head(k)[["date", "asset"]].copy()
            short_leg = g.tail(k)[["date", "asset"]].copy()
            long_leg["weight"] = 1.0 / k
            short_leg["weight"] = -1.0 / k
            frames.append(pd.concat([long_leg, short_leg], ignore_index=True))
        if not frames:
            return pd.DataFrame(columns=["date", "asset", "weight"])
        return pd.concat(frames, ignore_index=True)
