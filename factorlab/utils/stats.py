"""统计计算辅助函数。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_corr(
    left: pd.Series,
    right: pd.Series,
    method: str = "pearson",
    min_obs: int = 3,
) -> float:
    a = pd.to_numeric(pd.Series(left), errors="coerce")
    b = pd.to_numeric(pd.Series(right), errors="coerce")
    mask = a.notna() & b.notna()
    if int(mask.sum()) < int(min_obs):
        return float("nan")
    x = a[mask]
    y = b[mask]
    if int(x.nunique(dropna=True)) < 2 or int(y.nunique(dropna=True)) < 2:
        return float("nan")
    val = x.corr(y, method=method)
    return float(val) if np.isfinite(val) else float("nan")

