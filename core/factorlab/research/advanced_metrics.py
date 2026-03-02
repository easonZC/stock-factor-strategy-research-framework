"""高级研究指标与风险收益统计。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from factorlab.utils import safe_corr


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def _sortino_ratio(returns: pd.Series, annualization_days: int = 252) -> float:
    if returns.empty:
        return float("nan")
    s = returns.dropna().astype(float)
    if s.empty:
        return float("nan")
    downside = s[s < 0]
    downside_std = float(np.sqrt(np.mean(np.square(downside.values)))) if not downside.empty else 0.0
    if downside_std <= 0:
        return float("nan")
    mean_ret = float(s.mean())
    return float((mean_ret / downside_std) * np.sqrt(annualization_days))


def summarize_quantile_profile(
    q_daily: pd.DataFrame,
    annualization_days: int = 252,
) -> pd.DataFrame:
    q_cols = [c for c in q_daily.columns if c.startswith("Q")]
    rows: list[dict[str, float | str]] = []
    for col in q_cols + ["long_short"]:
        s = q_daily[col].dropna().astype(float)
        if s.empty:
            rows.append(
                {
                    "bucket": col,
                    "mean_ret": np.nan,
                    "std_ret": np.nan,
                    "ann_ret": np.nan,
                    "ann_vol": np.nan,
                    "sharpe": np.nan,
                    "hit_rate": np.nan,
                    "max_drawdown": np.nan,
                }
            )
            continue
        mean_ret = float(s.mean())
        std_ret = float(s.std(ddof=0))
        ann_ret = float((1.0 + mean_ret) ** annualization_days - 1.0)
        ann_vol = float(std_ret * np.sqrt(annualization_days))
        sharpe = float((mean_ret / std_ret) * np.sqrt(annualization_days)) if std_ret > 0 else np.nan
        mdd = _max_drawdown_from_returns(s)
        calmar = float(ann_ret / abs(mdd)) if np.isfinite(mdd) and mdd < 0 else np.nan
        rows.append(
            {
                "bucket": col,
                "mean_ret": mean_ret,
                "std_ret": std_ret,
                "ann_ret": ann_ret,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "sortino": _sortino_ratio(s, annualization_days=annualization_days),
                "calmar": calmar,
                "hit_rate": float((s > 0).mean()),
                "max_drawdown": mdd,
            }
        )
    return pd.DataFrame(rows)


def summarize_quantile_monotonicity(q_daily: pd.DataFrame) -> dict[str, float]:
    q_cols = sorted([c for c in q_daily.columns if c.startswith("Q")], key=lambda x: int(x[1:]))
    if len(q_cols) < 3:
        return {
            "quantile_monotonicity_mean": np.nan,
            "quantile_monotonicity_pos_ratio": np.nan,
        }

    q_idx = np.arange(1, len(q_cols) + 1, dtype=float)
    rho_vals: list[float] = []
    for _, row in q_daily[q_cols].iterrows():
        vals = row.to_numpy(dtype=float)
        mask = np.isfinite(vals)
        if mask.sum() < 3:
            continue
        rho = safe_corr(pd.Series(q_idx[mask]), pd.Series(vals[mask]), method="spearman", min_obs=3)
        rho_vals.append(float(rho))

    if not rho_vals:
        return {
            "quantile_monotonicity_mean": np.nan,
            "quantile_monotonicity_pos_ratio": np.nan,
        }
    arr = np.array(rho_vals, dtype=float)
    return {
        "quantile_monotonicity_mean": float(np.nanmean(arr)),
        "quantile_monotonicity_pos_ratio": float(np.nanmean(arr > 0)),
    }


def compute_long_short_alpha_beta(
    long_short_returns: pd.Series,
    market_returns: pd.Series,
    annualization_days: int = 252,
) -> dict[str, float]:
    df = pd.DataFrame({"ls": long_short_returns, "mkt": market_returns}).dropna()
    if len(df) < 10:
        return {"ls_alpha_ann": np.nan, "ls_beta": np.nan, "ls_r2": np.nan}

    x = df["mkt"].to_numpy(dtype=float)
    y = df["ls"].to_numpy(dtype=float)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_var = float(np.var(x))
    if x_var <= 0:
        return {"ls_alpha_ann": np.nan, "ls_beta": np.nan, "ls_r2": np.nan}

    cov = float(np.mean((x - x_mean) * (y - y_mean)))
    beta = cov / x_var
    alpha_daily = y_mean - beta * x_mean
    y_hat = alpha_daily + beta * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    alpha_ann = float((1.0 + alpha_daily) ** annualization_days - 1.0)
    return {"ls_alpha_ann": alpha_ann, "ls_beta": float(beta), "ls_r2": float(r2)}


def compute_factor_rank_autocorr(
    df: pd.DataFrame,
    factor_col: str,
    lag: int = 1,
) -> pd.DataFrame:
    if lag < 1:
        raise ValueError("lag must be >= 1")
    wide = df.pivot(index="date", columns="asset", values=factor_col).sort_index()
    rank_wide = wide.rank(axis=1, method="average", pct=True)
    rows: list[dict[str, float | pd.Timestamp]] = []
    for i in range(lag, len(rank_wide)):
        cur = rank_wide.iloc[i]
        prev = rank_wide.iloc[i - lag]
        mask = cur.notna() & prev.notna()
        rho = safe_corr(cur[mask], prev[mask], method="pearson", min_obs=5)
        rows.append({"date": rank_wide.index[i], f"rank_autocorr_lag{lag}": rho})
    return pd.DataFrame(rows)
