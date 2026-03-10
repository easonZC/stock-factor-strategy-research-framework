"""截面因子研究的无泄露预处理变换。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.config import CSStandardizeMode, NeutralizationConfig


def winsorize_series_quantile(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    lo = series.quantile(lower_q)
    hi = series.quantile(upper_q)
    return series.clip(lower=lo, upper=hi)


def winsorize_series_mad(series: pd.Series, scale: float = 5.0) -> pd.Series:
    med = series.median()
    mad = (series - med).abs().median()
    if pd.isna(mad) or mad == 0:
        return series.copy()
    lo = med - scale * mad
    hi = med + scale * mad
    return series.clip(lower=lo, upper=hi)


def apply_winsorize(
    df: pd.DataFrame,
    factor_col: str,
    method: str,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    mad_scale: float = 5.0,
) -> pd.Series:
    if method not in {"quantile", "mad"}:
        raise ValueError("winsorize method must be one of: quantile, mad")

    def _f(s: pd.Series) -> pd.Series:
        if method == "quantile":
            return winsorize_series_quantile(s, lower_q=lower_q, upper_q=upper_q)
        return winsorize_series_mad(s, scale=mad_scale)

    return df.groupby("date", group_keys=False)[factor_col].apply(_f)


def cs_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    grouped = df.groupby("date")[col]
    mu = grouped.transform("mean")
    sigma = grouped.transform("std").replace(0, np.nan)
    return (df[col] - mu) / sigma


def cs_rank(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("date")[col].rank(pct=True)


def cs_robust_zscore(df: pd.DataFrame, col: str, mad_scale: float = 1.4826) -> pd.Series:
    grouped = df.groupby("date")[col]
    median = grouped.transform("median")
    mad = grouped.transform(lambda s: (s - s.median()).abs().median())
    denom = (mad_scale * mad).where((mad_scale * mad).abs() > 1e-12, np.nan)
    return (df[col] - median) / denom


def apply_cs_standardize(df: pd.DataFrame, col: str, method: CSStandardizeMode) -> pd.Series:
    if method == "none":
        return df[col].astype(float)
    if method == "cs_rank":
        return cs_rank(df, col).astype(float)
    if method == "cs_robust_zscore":
        return cs_robust_zscore(df, col).astype(float)
    if method == "cs_zscore":
        return cs_zscore(df, col).astype(float)
    raise ValueError(f"Unsupported CS standardization method: {method}")


def ts_rolling_zscore(df: pd.DataFrame, col: str, window: int = 20) -> pd.Series:
    def _roll(s: pd.Series) -> pd.Series:
        mu = s.rolling(window=window, min_periods=max(5, window // 4)).mean()
        sigma = s.rolling(window=window, min_periods=max(5, window // 4)).std().replace(0, np.nan)
        return (s - mu) / sigma

    return df.sort_values(["asset", "date"]).groupby("asset", group_keys=False)[col].apply(_roll)


def handle_missing(df: pd.DataFrame, cols: list[str], policy: str = "drop") -> pd.DataFrame:
    """按显式策略处理缺失值。

    支持策略：
    - drop: 删除 `cols` 存在缺失的行。
    - fill_zero: 将 `cols` 缺失值填充为 0。
    - ffill_by_asset: 按 (asset, date) 排序后逐资产前向填充，再删除残余缺失。
    - cs_median_by_date: 按同日截面中位数填充，再删除残余缺失。
    - keep: 保持原样不处理。
    """
    out = df.copy()
    policy_norm = str(policy).strip().lower()
    if policy_norm == "drop":
        return out.dropna(subset=cols)
    if policy_norm == "fill_zero":
        out.loc[:, cols] = out[cols].fillna(0.0)
        return out
    if policy_norm == "ffill_by_asset":
        required = {"asset", "date"}
        if not required.issubset(out.columns):
            missing = sorted(required - set(out.columns))
            raise KeyError(f"ffill_by_asset requires columns: {missing}")
        order = out.index
        tmp = out.sort_values(["asset", "date"]).copy()
        tmp.loc[:, cols] = tmp.groupby("asset", group_keys=False)[cols].ffill()
        tmp = tmp.reindex(order)
        return tmp.dropna(subset=cols)
    if policy_norm == "cs_median_by_date":
        if "date" not in out.columns:
            raise KeyError("cs_median_by_date requires 'date' column")
        for col in cols:
            med = out.groupby("date")[col].transform("median")
            out[col] = out[col].fillna(med)
        return out.dropna(subset=cols)
    if policy_norm == "keep":
        return out
    raise ValueError(
        "Unsupported missing policy. Use one of: drop, fill_zero, "
        "ffill_by_asset, cs_median_by_date, keep."
    )


def neutralize_factor(
    df: pd.DataFrame,
    factor_col: str,
    config: NeutralizationConfig,
) -> pd.Series:
    """逐日截面中性化（规模/行业/二者）。

    严格无前视：
    - 仅使用同日截面暴露（`mkt_cap`、`industry`）。
    - 拟合残差回归时不引用未来日期样本。
    """

    mode = config.mode
    if mode == "none":
        return df[factor_col].copy()

    out = pd.Series(index=df.index, dtype=float)

    for dt, grp in df.groupby("date"):
        y = grp[factor_col].astype(float)
        design = []

        if mode in {"size", "both"} and config.size_col in grp.columns:
            design.append(np.log1p(grp[config.size_col].astype(float).replace([np.inf, -np.inf], np.nan)))

        if mode in {"industry", "both"} and config.industry_col in grp.columns:
            dummies = pd.get_dummies(grp[config.industry_col].astype(str), drop_first=True)
            if not dummies.empty:
                for c in dummies.columns:
                    design.append(dummies[c].astype(float))

        if not design:
            out.loc[grp.index] = y
            continue

        X = pd.concat(design, axis=1)
        valid = y.notna() & X.notna().all(axis=1)
        if valid.sum() < max(5, X.shape[1] + 2):
            out.loc[grp.index] = np.nan
            continue

        yv = y[valid].to_numpy(dtype=float)
        Xv = X[valid].to_numpy(dtype=float)
        Xv = np.column_stack([np.ones(len(Xv)), Xv])
        beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
        resid = yv - Xv @ beta

        tmp = pd.Series(np.nan, index=grp.index)
        tmp.loc[valid[valid].index] = resid
        out.loc[grp.index] = tmp

    return out
