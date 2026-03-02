"""Sample custom preprocess transforms for config-driven runs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def transform_robust_clip(
    panel: pd.DataFrame,
    factor_col: str,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
) -> pd.Series:
    """Cross-sectional quantile clip by date."""
    tmp = panel[["date", factor_col]].copy()

    def _clip(group: pd.Series) -> pd.Series:
        lo = group.quantile(float(lower_q))
        hi = group.quantile(float(upper_q))
        return group.clip(lower=lo, upper=hi)

    return tmp.groupby("date", group_keys=False)[factor_col].apply(_clip)


def transform_signed_log(panel: pd.DataFrame, factor_col: str) -> pd.Series:
    """Signed log transform to compress heavy tails while preserving direction."""
    s = panel[factor_col].astype(float)
    return np.sign(s) * np.log1p(np.abs(s))
