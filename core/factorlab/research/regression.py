"""截面回归与分解分析工具。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.research.statistics import newey_west_tstat


def run_fama_macbeth(
    df: pd.DataFrame,
    ret_col: str,
    factor_col: str,
    size_col: str | None = "mkt_cap",
    industry_col: str | None = "industry",
    min_obs: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["date", ret_col, factor_col]
    if size_col and size_col in df.columns:
        cols.append(size_col)
    if industry_col and industry_col in df.columns:
        cols.append(industry_col)
    base = df[cols].copy()

    x_cols: list[str] = [factor_col]
    if size_col and size_col in base.columns:
        base["log_size"] = np.log1p(pd.to_numeric(base[size_col], errors="coerce"))
        x_cols.append("log_size")
    if industry_col and industry_col in base.columns:
        dummies = pd.get_dummies(base[industry_col].astype(str), prefix="ind", drop_first=True)
        base = pd.concat([base, dummies], axis=1)
        x_cols.extend(list(dummies.columns))

    rows: list[dict[str, float | pd.Timestamp]] = []
    for dt, grp in base.groupby("date"):
        g = grp.dropna(subset=[ret_col, *x_cols]).copy()
        if len(g) < max(int(min_obs), len(x_cols) + 2):
            continue
        y = g[ret_col].to_numpy(dtype=float)
        X = g[x_cols].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)

        row: dict[str, float | pd.Timestamp] = {"date": dt, "intercept": float(beta[0])}
        for i, c in enumerate(x_cols):
            row[c] = float(beta[i + 1])
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["date", "intercept", *x_cols]), pd.DataFrame(
            columns=["coef", "mean_beta", "std_beta", "nw_t", "nw_p", "n_obs"]
        )

    coef_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    summary_rows: list[dict[str, float | str | int]] = []
    for c in ["intercept", *x_cols]:
        s = coef_df[c].astype(float)
        nw_t, nw_p = newey_west_tstat(s)
        summary_rows.append(
            {
                "coef": c,
                "mean_beta": float(s.mean()),
                "std_beta": float(s.std(ddof=0)),
                "nw_t": nw_t,
                "nw_p": nw_p,
                "n_obs": int(s.notna().sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return coef_df, summary


def quantile_group_decomposition(
    df: pd.DataFrame,
    factor_col: str,
    ret_col: str,
    group_col: str,
    quantiles: int = 5,
    min_group_size: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp = df[["date", factor_col, ret_col, group_col]].copy()
    tmp[group_col] = tmp[group_col].astype(str)
    rows: list[dict[str, float | str | pd.Timestamp | int]] = []
    for (dt, grp_name), grp in tmp.groupby(["date", group_col]):
        g = grp.dropna(subset=[factor_col, ret_col]).copy()
        if len(g) < max(int(min_group_size), int(quantiles)):
            continue
        ranks = g[factor_col].rank(method="first")
        q = pd.qcut(ranks, quantiles, labels=False, duplicates="drop")
        if q.isna().all():
            continue
        q_int = q.astype("float").dropna()
        if q_int.empty:
            continue
        lo = int(q_int.min())
        hi = int(q_int.max())
        g = g.loc[q.notna()].copy()
        g["q"] = q.dropna().astype(int).values
        top = float(g.loc[g["q"] == hi, ret_col].mean())
        bot = float(g.loc[g["q"] == lo, ret_col].mean())
        rows.append(
            {
                "date": dt,
                "group": str(grp_name),
                "long_short": top - bot,
                "top_ret": top,
                "bottom_ret": bot,
                "n_assets": int(len(g)),
            }
        )

    if not rows:
        detail_cols = ["date", "group", "long_short", "top_ret", "bottom_ret", "n_assets"]
        summary_cols = ["group", "mean_long_short", "std_long_short", "hit_rate", "n_dates", "contribution_share"]
        return pd.DataFrame(columns=detail_cols), pd.DataFrame(columns=summary_cols)

    detail = pd.DataFrame(rows).sort_values(["date", "group"]).reset_index(drop=True)
    summary = (
        detail.groupby("group", as_index=False)
        .agg(
            mean_long_short=("long_short", "mean"),
            std_long_short=("long_short", "std"),
            hit_rate=("long_short", lambda s: float((s > 0).mean())),
            n_dates=("long_short", "count"),
        )
        .sort_values("mean_long_short", ascending=False)
        .reset_index(drop=True)
    )
    denom = float(summary["mean_long_short"].abs().sum())
    summary["contribution_share"] = summary["mean_long_short"] / denom if denom > 0 else np.nan
    return detail, summary


def make_size_style_bucket(
    df: pd.DataFrame,
    size_col: str = "mkt_cap",
) -> pd.Series:
    if size_col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    labels = {1: "Small", 2: "Mid", 3: "Large"}

    def _assign(grp: pd.DataFrame) -> pd.Series:
        s = pd.to_numeric(grp[size_col], errors="coerce")
        valid = s.dropna()
        if len(valid) < 6:
            return pd.Series(np.nan, index=grp.index)
        ranks = valid.rank(method="first")
        q = pd.qcut(ranks, 3, labels=False, duplicates="drop")
        out = pd.Series(np.nan, index=grp.index, dtype=object)
        if q.notna().any():
            out.loc[valid.index] = q.astype(int).map(lambda x: labels.get(int(x) + 1, "Unknown")).values
        return out

    return df.groupby("date", group_keys=False).apply(_assign)

