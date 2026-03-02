"""Built-in strategy implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ssf.strategies.base import Strategy


def _normalize_long(weights: pd.Series, max_weight: float | None = None) -> pd.Series:
    if weights.empty:
        return weights
    w = weights.astype(float).copy()
    w = w.clip(lower=0.0)
    total = float(w.sum())
    if total <= 0:
        return pd.Series(0.0, index=w.index, dtype=float)
    w = w / total
    if max_weight is not None:
        cap = max(0.01, float(max_weight))
        w = w.clip(upper=cap)
        if float(w.sum()) > 0:
            w = w / float(w.sum())
    return w


def _normalize_long_short(
    long_weights: pd.Series,
    short_weights: pd.Series,
    max_weight: float | None = None,
) -> tuple[pd.Series, pd.Series]:
    w_long = _normalize_long(long_weights, max_weight=max_weight)
    w_short = _normalize_long(short_weights, max_weight=max_weight)
    return w_long, -w_short


def _weights_from_scores(scores: pd.Series, scheme: str) -> pd.Series:
    s = scores.astype(float)
    if scheme == "rank":
        ranks = s.rank(method="average")
        return ranks / float(ranks.sum()) if float(ranks.sum()) > 0 else pd.Series(1.0, index=s.index)
    return pd.Series(1.0, index=s.index, dtype=float)


@dataclass(slots=True)
class TopKLongStrategy(Strategy):
    """Equal-weight long-only top-k strategy."""

    top_k: int = 20
    rebalance_every: int = 1
    weight_scheme: str = "equal"
    max_weight: float | None = None

    def generate_weights(self, score_df: pd.DataFrame) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        prev: pd.Series | None = None
        dates = sorted(score_df["date"].dropna().unique())
        for i, dt in enumerate(dates):
            grp = score_df[score_df["date"] == dt].copy()
            if i % max(1, int(self.rebalance_every)) == 0 or prev is None:
                g = grp.dropna(subset=["score"]).sort_values("score", ascending=False).head(self.top_k)
                if g.empty:
                    continue
                base = _weights_from_scores(g["score"], scheme=str(self.weight_scheme).lower())
                prev = _normalize_long(base, max_weight=self.max_weight)
                prev.index = g["asset"].astype(str).values

            if prev is None or prev.empty:
                continue
            available_assets = set(grp["asset"].astype(str))
            cur = prev[prev.index.isin(available_assets)]
            if cur.empty:
                continue
            cur = _normalize_long(cur, max_weight=self.max_weight)
            chunk = pd.DataFrame({"date": dt, "asset": cur.index, "weight": cur.values})
            frames.append(chunk)
        if not frames:
            return pd.DataFrame(columns=["date", "asset", "weight"])
        return pd.concat(frames, ignore_index=True)


@dataclass(slots=True)
class LongShortQuantileStrategy(Strategy):
    """Long top quantile, short bottom quantile with equal absolute weights."""

    quantile: float = 0.2
    rebalance_every: int = 1
    weight_scheme: str = "equal"
    max_weight: float | None = None

    def generate_weights(self, score_df: pd.DataFrame) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        prev: pd.Series | None = None
        dates = sorted(score_df["date"].dropna().unique())
        for i, dt in enumerate(dates):
            grp = score_df[score_df["date"] == dt].copy()
            if i % max(1, int(self.rebalance_every)) == 0 or prev is None:
                g = grp.dropna(subset=["score"]).sort_values("score", ascending=False)
                if g.empty:
                    continue
                k = max(1, int(np.floor(len(g) * self.quantile)))
                long_leg = g.head(k).copy()
                short_leg = g.tail(k).copy()

                long_base = _weights_from_scores(long_leg["score"], scheme=str(self.weight_scheme).lower())
                short_base = _weights_from_scores(short_leg["score"].abs(), scheme=str(self.weight_scheme).lower())
                w_long, w_short = _normalize_long_short(
                    long_base,
                    short_base,
                    max_weight=self.max_weight,
                )
                prev = pd.concat(
                    [
                        pd.Series(w_long.values, index=long_leg["asset"].astype(str).values),
                        pd.Series(w_short.values, index=short_leg["asset"].astype(str).values),
                    ]
                )

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


@dataclass(slots=True)
class FlexibleLongShortStrategy(Strategy):
    """Configurable long-short strategy with optional long-only mode."""

    long_fraction: float = 0.2
    short_fraction: float = 0.2
    long_only: bool = False
    rebalance_every: int = 1
    weight_scheme: str = "equal"
    max_weight: float | None = None

    def generate_weights(self, score_df: pd.DataFrame) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        prev: pd.Series | None = None
        dates = sorted(score_df["date"].dropna().unique())
        for i, dt in enumerate(dates):
            grp = score_df[score_df["date"] == dt].copy()
            if i % max(1, int(self.rebalance_every)) == 0 or prev is None:
                g = grp.dropna(subset=["score"]).sort_values("score", ascending=False)
                if g.empty:
                    continue
                n = len(g)
                k_long = max(1, int(np.floor(n * float(self.long_fraction))))
                long_leg = g.head(k_long).copy()
                long_base = _weights_from_scores(long_leg["score"], scheme=str(self.weight_scheme).lower())
                long_w = _normalize_long(long_base, max_weight=self.max_weight)

                if bool(self.long_only):
                    prev = pd.Series(long_w.values, index=long_leg["asset"].astype(str).values)
                else:
                    k_short = max(1, int(np.floor(n * float(self.short_fraction))))
                    short_leg = g.tail(k_short).copy()
                    short_base = _weights_from_scores(
                        short_leg["score"].abs(),
                        scheme=str(self.weight_scheme).lower(),
                    )
                    norm_long, norm_short = _normalize_long_short(
                        long_w,
                        short_base,
                        max_weight=self.max_weight,
                    )
                    prev = pd.concat(
                        [
                            pd.Series(norm_long.values, index=long_leg["asset"].astype(str).values),
                            pd.Series(norm_short.values, index=short_leg["asset"].astype(str).values),
                        ]
                    )

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
