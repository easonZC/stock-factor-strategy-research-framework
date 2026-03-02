"""分位组合分析：含换手与多空腿。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_quantiles(df: pd.DataFrame, factor_col: str, quantiles: int) -> pd.Series:
    """中文说明。"""

    def _assign(grp: pd.DataFrame) -> pd.Series:
        if grp[factor_col].notna().sum() < quantiles:
            return pd.Series(np.nan, index=grp.index)
        ranks = grp[factor_col].rank(method="first")
        q = pd.qcut(ranks, quantiles, labels=False, duplicates="drop")
        return q + 1

    return df.groupby("date", group_keys=False).apply(_assign)


def quantile_returns(
    df: pd.DataFrame,
    factor_col: str,
    ret_col: str,
    quantiles: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """中文说明。"""
    tmp = df[["date", "asset", factor_col, ret_col]].copy()
    tmp["quantile"] = assign_quantiles(tmp, factor_col, quantiles)

    daily = (
        tmp.dropna(subset=["quantile", ret_col])
        .groupby(["date", "quantile"], as_index=False)[ret_col]
        .mean()
    )
    piv = daily.pivot(index="date", columns="quantile", values=ret_col).sort_index()

    for q in range(1, quantiles + 1):
        if q not in piv.columns:
            piv[q] = np.nan
    piv = piv[[q for q in range(1, quantiles + 1)]]
    piv.columns = [f"Q{q}" for q in range(1, quantiles + 1)]
    piv["long_short"] = piv[f"Q{quantiles}"] - piv["Q1"]

    nav = (1 + piv.fillna(0.0)).cumprod().reset_index()
    daily_ret = piv.reset_index()

    turnover_rows: list[dict[str, float]] = []
    prev_sets: dict[int, set[str]] = {}
    for dt, grp in tmp.dropna(subset=["quantile"]).groupby("date"):
        row: dict[str, float] = {"date": dt}
        for q in range(1, quantiles + 1):
            cur = set(grp.loc[grp["quantile"] == q, "asset"].astype(str).tolist())
            prev = prev_sets.get(q, set())
            if not prev:
                row[f"Q{q}"] = np.nan
            else:
                overlap = len(cur & prev) / max(len(prev), 1)
                row[f"Q{q}"] = 1.0 - overlap
            prev_sets[q] = cur
        ls_inputs = [row.get("Q1", np.nan), row.get(f"Q{quantiles}", np.nan)]
        if np.isnan(ls_inputs).all():
            row["long_short"] = np.nan
        else:
            row["long_short"] = float(np.nanmean(ls_inputs))
        turnover_rows.append(row)

    turnover = pd.DataFrame(turnover_rows).sort_values("date").reset_index(drop=True)
    return daily_ret, nav, turnover
