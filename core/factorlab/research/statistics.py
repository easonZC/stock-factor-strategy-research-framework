"""Statistical utilities: IC, ICIR, decay, and Newey-West significance."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats
from factorlab.utils import safe_corr


def compute_daily_ic(df: pd.DataFrame, factor_col: str, ret_col: str) -> pd.DataFrame:
    """Compute daily Pearson IC and Spearman RankIC."""
    rows: list[dict[str, float]] = []
    for dt, grp in df.groupby("date"):
        g = grp[[factor_col, ret_col]].dropna()
        if len(g) < 5:
            continue
        ic = safe_corr(g[factor_col], g[ret_col], method="pearson", min_obs=5)
        rank_ic = safe_corr(g[factor_col], g[ret_col], method="spearman", min_obs=5)
        rows.append({"date": dt, "ic": float(ic), "rank_ic": float(rank_ic)})
    if not rows:
        return pd.DataFrame(columns=["date", "ic", "rank_ic"])
    out = pd.DataFrame(rows).sort_values("date")
    return out


def newey_west_tstat(series: pd.Series, lags: int | None = None) -> tuple[float, float]:
    """Compute Newey-West t-stat and p-value for mean(series)."""
    x = series.dropna().astype(float).to_numpy()
    n = len(x)
    if n < 8:
        return float("nan"), float("nan")

    mu = x.mean()
    u = x - mu
    if lags is None:
        lags = int(math.floor(4 * (n / 100) ** (2 / 9)))
        lags = max(1, lags)

    gamma0 = np.dot(u, u) / n
    long_var = gamma0
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        gamma = np.dot(u[l:], u[:-l]) / n
        long_var += 2 * w * gamma

    var_mean = long_var / n
    if var_mean <= 0:
        return float("nan"), float("nan")

    t_stat = mu / np.sqrt(var_mean)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
    return float(t_stat), float(p_value)


def summarize_ic(ic_df: pd.DataFrame) -> dict[str, float]:
    """Summarize daily IC/RankIC series."""
    ic_mean = float(ic_df["ic"].mean())
    ic_std = float(ic_df["ic"].std(ddof=0))
    rank_ic_mean = float(ic_df["rank_ic"].mean())
    rank_ic_std = float(ic_df["rank_ic"].std(ddof=0))

    icir = ic_mean / ic_std if ic_std > 0 else float("nan")
    rank_icir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else float("nan")

    nw_t, nw_p = newey_west_tstat(ic_df["ic"])
    nw_rank_t, nw_rank_p = newey_west_tstat(ic_df["rank_ic"])

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_std": rank_ic_std,
        "icir": float(icir),
        "rank_icir": float(rank_icir),
        "nw_t_ic": nw_t,
        "nw_p_ic": nw_p,
        "nw_t_rank_ic": nw_rank_t,
        "nw_p_rank_ic": nw_rank_p,
    }


def build_ic_decay(summary_rows: list[dict[str, float]]) -> pd.DataFrame:
    """Build IC-decay table across horizons from summary rows."""
    df = pd.DataFrame(summary_rows)
    cols = ["horizon", "ic_mean", "rank_ic_mean"]
    return df[cols].sort_values("horizon").reset_index(drop=True)
