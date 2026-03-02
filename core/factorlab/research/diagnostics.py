"""Coverage, missingness, outlier, and stability diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def coverage_by_date(df: pd.DataFrame, factor_col: str) -> pd.DataFrame:
    """Coverage ratio per date for a factor."""
    total = df.groupby("date")["asset"].count()
    usable = df.groupby("date")[factor_col].apply(lambda s: s.notna().sum())
    cov = (usable / total).rename("coverage").reset_index()
    return cov


def missing_rates(df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """Missing ratio per factor."""
    rows = []
    for col in factor_cols:
        rows.append({"factor": col, "missing_rate": float(df[col].isna().mean())})
    return pd.DataFrame(rows)


def outlier_monitor(before: pd.Series, after: pd.Series, factor_name: str) -> pd.DataFrame:
    """Before/after stats after winsorization for report diagnostics."""
    return pd.DataFrame(
        {
            "factor": [factor_name],
            "before_mean": [float(before.mean())],
            "before_std": [float(before.std(ddof=0))],
            "after_mean": [float(after.mean())],
            "after_std": [float(after.std(ddof=0))],
        }
    )


def factor_stability(df: pd.DataFrame, factor_col: str) -> pd.DataFrame:
    """Compute lag1/lag5 autocorrelation and rolling drift metrics."""
    wide = df.pivot(index="date", columns="asset", values=factor_col).sort_index()
    dates = wide.index

    ac1 = []
    ac5 = []
    for i, dt in enumerate(dates):
        row_now = wide.iloc[i]

        if i >= 1:
            row_l1 = wide.iloc[i - 1]
            mask = row_now.notna() & row_l1.notna()
            ac1.append(row_now[mask].corr(row_l1[mask]) if mask.sum() > 5 else np.nan)
        else:
            ac1.append(np.nan)

        if i >= 5:
            row_l5 = wide.iloc[i - 5]
            mask = row_now.notna() & row_l5.notna()
            ac5.append(row_now[mask].corr(row_l5[mask]) if mask.sum() > 5 else np.nan)
        else:
            ac5.append(np.nan)

    cs_mean = wide.mean(axis=1)
    cs_std = wide.std(axis=1)
    st = pd.DataFrame(
        {
            "date": dates,
            "autocorr_lag1": ac1,
            "autocorr_lag5": ac5,
            "rolling_mean_20": cs_mean.rolling(20, min_periods=5).mean().values,
            "rolling_std_20": cs_std.rolling(20, min_periods=5).mean().values,
        }
    )
    return st


def factor_corr_matrix(df: pd.DataFrame, factors: list[str], method: str = "spearman") -> pd.DataFrame:
    """Multi-factor correlation matrix."""
    if len(factors) < 2:
        return pd.DataFrame()
    mat = df[factors].corr(method=method)
    return mat
