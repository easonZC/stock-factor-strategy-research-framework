"""时序因子研究流水线实现。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from factorlab.plotting import (
    plot_coverage,
    plot_ic_series,
    plot_lag_profile,
    plot_quantile_nav,
    plot_turnover,
)
from factorlab.preprocess import ts_rolling_zscore
from factorlab.reporting import render_report
from factorlab.research.advanced_metrics import summarize_quantile_profile
from factorlab.research.diagnostics import coverage_by_date, outlier_monitor
from factorlab.research.forward_returns import add_forward_returns
from factorlab.research.statistics import newey_west_tstat, summarize_ic
from factorlab.utils import get_logger, safe_slug

LOGGER = get_logger("factorlab.research.ts")


TSStandardizeMode = Literal["ts_rolling_zscore", "zscore", "none"]


def _build_factor_slug_map(factors: list[str]) -> dict[str, str]:
    """为 TS 输出目录生成稳定且不冲突的因子 slug。"""
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for fac in factors:
        base = safe_slug(fac, default="factor")
        candidate = base
        seq = 2
        while candidate in used:
            candidate = f"{base}_{seq}"
            seq += 1
        used.add(candidate)
        mapping[fac] = candidate
    return mapping


@dataclass(slots=True)
class TSResearchConfig:
    """时序因子分析配置。"""

    horizons: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    quantiles: int = 5
    ic_rolling_window: int = 20
    annualization_days: int = 252
    standardization: TSStandardizeMode = "ts_rolling_zscore"
    ts_standardize_window: int = 60
    ts_quantile_lookback: int = 60
    min_obs_per_asset_for_ic: int = 40
    ts_signal_lags: list[int] = field(default_factory=lambda: [0, 1, 2, 5, 10])


def _zscore_by_asset(df: pd.DataFrame, col: str) -> pd.Series:
    mu = df.groupby("asset")[col].transform("mean")
    sigma = df.groupby("asset")[col].transform("std").replace(0, np.nan)
    return (df[col] - mu) / sigma


def _standardize_ts_factor(panel: pd.DataFrame, factor_col: str, cfg: TSResearchConfig) -> pd.Series:
    if cfg.standardization == "none":
        return panel[factor_col].astype(float)
    if cfg.standardization == "zscore":
        return _zscore_by_asset(panel, factor_col).astype(float)
    return ts_rolling_zscore(
        panel[["asset", "date", factor_col]].copy(),
        col=factor_col,
        window=max(5, int(cfg.ts_standardize_window)),
    ).astype(float)


def _rolling_pct_rank(series: pd.Series, lookback: int, min_periods: int) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    for i, x in enumerate(arr):
        if not np.isfinite(x):
            continue
        start = max(0, i - lookback + 1)
        win = arr[start : i + 1]
        win = win[np.isfinite(win)]
        if len(win) < min_periods:
            continue
        out[i] = float(np.mean(win <= x))
    return pd.Series(out, index=series.index, dtype=float)


def _assign_time_quantiles(
    df: pd.DataFrame,
    factor_col: str,
    quantiles: int,
    lookback: int,
) -> pd.Series:
    min_periods = max(quantiles, lookback // 3, 8)
    tmp = df[["asset", "date", factor_col]].sort_values(["asset", "date"]).copy()
    tmp["_pct"] = tmp.groupby("asset", group_keys=False)[factor_col].apply(
        lambda s: _rolling_pct_rank(s, lookback=lookback, min_periods=min_periods)
    )
    q = np.ceil(tmp["_pct"] * quantiles)
    q = q.clip(lower=1, upper=quantiles)
    return q.reindex(df.index)


def _time_quantile_returns(
    df: pd.DataFrame,
    factor_col: str,
    ret_col: str,
    quantiles: int,
    lookback: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tmp = df[["date", "asset", factor_col, ret_col]].copy()
    tmp["quantile"] = _assign_time_quantiles(
        tmp,
        factor_col=factor_col,
        quantiles=quantiles,
        lookback=lookback,
    )
    tmp = tmp.dropna(subset=["quantile", ret_col]).copy()
    tmp["quantile"] = tmp["quantile"].astype(int)

    daily = tmp.groupby(["date", "quantile"], as_index=False)[ret_col].mean()
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
    for dt, grp in tmp.groupby("date"):
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
        row["long_short"] = float(np.nanmean(ls_inputs)) if not np.isnan(ls_inputs).all() else np.nan
        turnover_rows.append(row)

    turnover = pd.DataFrame(turnover_rows).sort_values("date").reset_index(drop=True)
    return daily_ret, nav, turnover


def _compute_time_ic_series(
    df: pd.DataFrame,
    factor_col: str,
    ret_col: str,
    window: int,
    min_obs_per_asset: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    minp = max(8, window // 3)
    for _, grp in df.groupby("asset"):
        g = grp.sort_values("date")[[factor_col, ret_col, "date"]].dropna().copy()
        if len(g) < max(min_obs_per_asset, minp):
            continue
        g["ic"] = g[factor_col].rolling(window=window, min_periods=minp).corr(g[ret_col])
        # 用资产内排序近似滚动秩信息系数，保持实现轻量且稳定。
        ranked_factor = g[factor_col].rank(method="average")
        ranked_ret = g[ret_col].rank(method="average")
        g["rank_ic"] = ranked_factor.rolling(window=window, min_periods=minp).corr(ranked_ret)
        parts.append(g[["date", "ic", "rank_ic"]])
    if not parts:
        return pd.DataFrame(columns=["date", "ic", "rank_ic"])
    out = pd.concat(parts, ignore_index=True)
    out = out.groupby("date", as_index=False)[["ic", "rank_ic"]].mean().sort_values("date")
    return out


def _compute_signal_lag_profile(
    df: pd.DataFrame,
    factor_col: str,
    ret_col: str,
    lags: list[int],
    window: int,
    min_obs_per_asset: int,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    base = df[["date", "asset", factor_col, ret_col]].sort_values(["asset", "date"]).copy()
    for lag in sorted({max(0, int(x)) for x in lags}):
        tmp = base.copy()
        shifted_col = ret_col
        if lag > 0:
            shifted_col = f"{ret_col}_lag_{lag}"
            tmp[shifted_col] = tmp.groupby("asset")[ret_col].shift(-lag)
        ic_daily = _compute_time_ic_series(
            tmp.dropna(subset=[factor_col, shifted_col]),
            factor_col=factor_col,
            ret_col=shifted_col,
            window=window,
            min_obs_per_asset=min_obs_per_asset,
        )
        if ic_daily.empty:
            rows.append(
                {
                    "lag": lag,
                    "ic_mean": np.nan,
                    "rank_ic_mean": np.nan,
                    "icir": np.nan,
                    "rank_icir": np.nan,
                    "nw_t_ic": np.nan,
                    "nw_p_ic": np.nan,
                    "n_dates": 0,
                }
            )
            continue
        stats = summarize_ic(ic_daily)
        nw_t_ic, nw_p_ic = newey_west_tstat(ic_daily["ic"])
        rows.append(
            {
                "lag": lag,
                "ic_mean": float(stats.get("ic_mean", np.nan)),
                "rank_ic_mean": float(stats.get("rank_ic_mean", np.nan)),
                "icir": float(stats.get("icir", np.nan)),
                "rank_icir": float(stats.get("rank_icir", np.nan)),
                "nw_t_ic": nw_t_ic,
                "nw_p_ic": nw_p_ic,
                "n_dates": int(len(ic_daily)),
            }
        )
    return pd.DataFrame(rows).sort_values("lag").reset_index(drop=True)


class TimeSeriesFactorResearchPipeline:
    """生成 TS 因子研究输出（表格、图形、HTML 报告）。"""

    def __init__(self, config: TSResearchConfig):
        self.config = config

    def run(
        self,
        panel: pd.DataFrame,
        factors: list[str],
        out_dir: str | Path,
        overview_files: dict[str, Path] | None = None,
    ) -> dict[str, Path]:
        out = Path(out_dir)
        assets_dir = out / "assets"
        tables_dir = out / "tables"
        detail_assets_dir = assets_dir / "detail"
        detail_tables_dir = tables_dir / "detail"
        assets_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        detail_assets_dir.mkdir(parents=True, exist_ok=True)
        detail_tables_dir.mkdir(parents=True, exist_ok=True)

        panel = panel.sort_values(["asset", "date"]).reset_index(drop=True)
        panel = add_forward_returns(panel, horizons=self.config.horizons)
        factor_slug_map = _build_factor_slug_map(factors)

        summary_rows: list[dict[str, float | str | int]] = []
        figure_map: dict[str, list[Path]] = {}
        table_map: dict[str, list[Path]] = {}
        if overview_files and overview_files.get("factor_definitions") is not None:
            table_map["overview"] = [overview_files["factor_definitions"]]
        figure_sources: dict[str, dict[str, Any]] = {}
        outlier_rows: list[pd.DataFrame] = []

        def register_figure(
            path: Path,
            *,
            factor: str | None,
            variant: str,
            label: str,
            source_tables: list[Path],
            description: str,
            scope: str = "detail",
        ) -> Path:
            figure_sources[str(path.resolve())] = {
                "factor": factor,
                "variant": variant,
                "chart": path.name,
                "label": label,
                "source_tables": [str(item) for item in source_tables],
                "description": description,
                "scope": scope,
            }
            return path

        for fac in factors:
            LOGGER.info("Running TS factor research for: %s", fac)
            fac_raw = panel[fac].astype(float)
            fac_std = _standardize_ts_factor(panel, factor_col=fac, cfg=self.config)
            panel[f"{fac}_ts"] = fac_std
            outlier_rows.append(outlier_monitor(fac_raw, fac_std, fac))
            fac_slug = factor_slug_map[fac]
            fac_asset_dir = detail_assets_dir / f"{fac_slug}__ts"
            fac_table_dir = detail_tables_dir / f"{fac_slug}__ts"
            fac_asset_dir.mkdir(parents=True, exist_ok=True)
            fac_table_dir.mkdir(parents=True, exist_ok=True)

            fac_tables: list[Path] = []
            fac_figs: list[Path] = []
            ic_primary: pd.DataFrame | None = None
            primary_ic_path: Path | None = None

            for h in self.config.horizons:
                ret_col = f"fwd_ret_{h}"
                tmp = panel[["date", "asset", f"{fac}_ts", ret_col]].dropna()
                ic_daily = _compute_time_ic_series(
                    tmp.rename(columns={f"{fac}_ts": "factor"}),
                    factor_col="factor",
                    ret_col=ret_col,
                    window=max(10, int(self.config.ic_rolling_window)),
                    min_obs_per_asset=int(self.config.min_obs_per_asset_for_ic),
                )
                if ic_daily.empty:
                    continue
                ic_daily["ic_roll"] = ic_daily["ic"].rolling(
                    max(5, int(self.config.ic_rolling_window)),
                    min_periods=5,
                ).mean()
                stats = summarize_ic(ic_daily)
                stats_row = {"factor": fac, "variant": "ts", "horizon": int(h)}
                stats_row.update(stats)
                summary_rows.append(stats_row)

                ic_path = fac_table_dir / f"ic_daily_h{h}.csv"
                ic_daily.to_csv(ic_path, index=False)
                fac_tables.append(ic_path)
                if h == self.config.horizons[0]:
                    ic_primary = ic_daily
                    primary_ic_path = ic_path

            if ic_primary is None:
                continue

            primary_ret_col = f"fwd_ret_{self.config.horizons[0]}"
            tmpq = panel[["date", "asset", f"{fac}_ts", primary_ret_col]].dropna()
            q_daily, q_nav, q_turn = _time_quantile_returns(
                tmpq.rename(columns={f"{fac}_ts": "factor"}),
                factor_col="factor",
                ret_col=primary_ret_col,
                quantiles=int(self.config.quantiles),
                lookback=max(10, int(self.config.ts_quantile_lookback)),
            )
            q_daily_path = fac_table_dir / "quantile_daily.csv"
            q_nav_path = fac_table_dir / "quantile_nav.csv"
            q_turn_path = fac_table_dir / "turnover.csv"
            q_daily.to_csv(q_daily_path, index=False)
            q_nav.to_csv(q_nav_path, index=False)
            q_turn.to_csv(q_turn_path, index=False)
            fac_tables.extend([q_daily_path, q_nav_path, q_turn_path])

            q_profile = summarize_quantile_profile(
                q_daily,
                annualization_days=int(self.config.annualization_days),
            )
            q_profile_path = fac_table_dir / "quantile_profile.csv"
            q_profile.to_csv(q_profile_path, index=False)
            fac_tables.append(q_profile_path)

            lag_profile = _compute_signal_lag_profile(
                tmpq.rename(columns={f"{fac}_ts": "factor"}),
                factor_col="factor",
                ret_col=primary_ret_col,
                lags=self.config.ts_signal_lags,
                window=max(10, int(self.config.ic_rolling_window)),
                min_obs_per_asset=int(self.config.min_obs_per_asset_for_ic),
            )
            lag_profile_path = fac_table_dir / "signal_lag_ic.csv"
            lag_profile.to_csv(lag_profile_path, index=False)
            fac_tables.append(lag_profile_path)

            ls_t, ls_p = newey_west_tstat(q_daily["long_short"])
            ls_profile_row = q_profile[q_profile["bucket"] == "long_short"]
            ls_profile = ls_profile_row.iloc[0].to_dict() if not ls_profile_row.empty else {}
            best_lag_row = {}
            lag_valid = lag_profile[np.isfinite(lag_profile["ic_mean"])] if not lag_profile.empty else pd.DataFrame()
            if not lag_valid.empty:
                best_lag_row = lag_valid.sort_values("ic_mean", ascending=False).iloc[0].to_dict()
            lag0_row = (
                lag_profile[lag_profile["lag"] == 0].iloc[0].to_dict()
                if (not lag_profile.empty and (lag_profile["lag"] == 0).any())
                else {}
            )
            summary_rows.append(
                {
                    "factor": fac,
                    "variant": "ts",
                    "horizon": int(self.config.horizons[0]),
                    "ic_mean": np.nan,
                    "ic_std": np.nan,
                    "rank_ic_mean": np.nan,
                    "rank_ic_std": np.nan,
                    "icir": np.nan,
                    "rank_icir": np.nan,
                    "nw_t_ic": np.nan,
                    "nw_p_ic": np.nan,
                    "nw_t_rank_ic": np.nan,
                    "nw_p_rank_ic": np.nan,
                    "nw_t_long_short": ls_t,
                    "nw_p_long_short": ls_p,
                    "ls_mean_ret": ls_profile.get("mean_ret", np.nan),
                    "ls_sharpe": ls_profile.get("sharpe", np.nan),
                    "ls_sortino": ls_profile.get("sortino", np.nan),
                    "ls_calmar": ls_profile.get("calmar", np.nan),
                    "ls_hit_rate": ls_profile.get("hit_rate", np.nan),
                    "ls_max_drawdown": ls_profile.get("max_drawdown", np.nan),
                    "signal_lag0_ic_mean": lag0_row.get("ic_mean", np.nan),
                    "signal_lag0_rank_ic_mean": lag0_row.get("rank_ic_mean", np.nan),
                    "signal_best_lag": best_lag_row.get("lag", np.nan),
                    "signal_best_lag_ic_mean": best_lag_row.get("ic_mean", np.nan),
                }
            )

            cov = coverage_by_date(
                panel[["date", "asset", f"{fac}_ts"]].rename(columns={f"{fac}_ts": "factor"}),
                "factor",
            )
            cov_path = fac_table_dir / "coverage.csv"
            cov.to_csv(cov_path, index=False)
            fac_tables.append(cov_path)

            fac_figs.extend(
                [
                    register_figure(
                        plot_ic_series(ic_primary, fac_asset_dir / "ic.png", title=f"{fac} [ts] IC"),
                        factor=fac,
                        variant="ts",
                        label="IC Series",
                        source_tables=[primary_ic_path] if primary_ic_path is not None else [],
                        description="Primary IC / RankIC series generated from same-run time-series IC tables.",
                    ),
                    register_figure(
                        plot_quantile_nav(
                            q_nav,
                            fac_asset_dir / "quantile_nav.png",
                            title=f"{fac} [ts] Time-Quantile NAV",
                        ),
                        factor=fac,
                        variant="ts",
                        label="Quantile NAV",
                        source_tables=[q_nav_path, q_profile_path, q_daily_path],
                        description="Time-quantile NAV chart generated from same-run quantile return tables.",
                    ),
                    register_figure(
                        plot_turnover(
                            q_turn,
                            fac_asset_dir / "turnover.png",
                            title=f"{fac} [ts] Time-Quantile Turnover",
                        ),
                        factor=fac,
                        variant="ts",
                        label="Turnover",
                        source_tables=[q_turn_path],
                        description="Time-quantile turnover chart generated from same-run turnover tables.",
                    ),
                    register_figure(
                        plot_coverage(cov, fac_asset_dir / "coverage.png", title=f"{fac} [ts] Coverage"),
                        factor=fac,
                        variant="ts",
                        label="Coverage",
                        source_tables=[cov_path],
                        description="Coverage chart generated from same-run coverage tables.",
                    ),
                    register_figure(
                        plot_lag_profile(
                            lag_profile,
                            fac_asset_dir / "signal_lag_ic.png",
                            title=f"{fac} [ts] Signal Lag IC",
                        ),
                        factor=fac,
                        variant="ts",
                        label="Signal Lag IC",
                        source_tables=[lag_profile_path],
                        description="Signal lag IC chart generated from same-run lag evaluation tables.",
                    ),
                ]
            )

            figure_map[fac] = fac_figs
            table_map[fac] = fac_tables

        if outlier_rows:
            outlier_df = pd.concat(outlier_rows, ignore_index=True)
            outlier_path = tables_dir / "outlier_before_after.csv"
            outlier_df.to_csv(outlier_path, index=False)
            table_map.setdefault("global", []).append(outlier_path)

        summary = pd.DataFrame(summary_rows)
        summary_path = tables_dir / "summary.csv"
        summary.to_csv(summary_path, index=False)
        table_map.setdefault("global", []).append(summary_path)

        config_path = out / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "horizons": self.config.horizons,
                    "quantiles": self.config.quantiles,
                    "annualization_days": self.config.annualization_days,
                    "standardization": self.config.standardization,
                    "ts_standardize_window": self.config.ts_standardize_window,
                    "ts_quantile_lookback": self.config.ts_quantile_lookback,
                    "ts_signal_lags": self.config.ts_signal_lags,
                    "factors": factors,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        index_html = render_report(
            out,
            summary=summary,
            figure_map=figure_map,
            table_map=table_map,
            figure_sources=figure_sources,
            overview_files=overview_files,
        )
        LOGGER.info("TS report generated at: %s", index_html)
        return {
            "index_html": index_html,
            "summary_csv": summary_path,
            "config_json": config_path,
            "assets_dir": assets_dir,
            "tables_dir": tables_dir,
        }
