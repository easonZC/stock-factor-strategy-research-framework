"""End-to-end factor research pipeline (report-grade, configurable)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from factorlab.config import ResearchConfig
from factorlab.plotting import (
    plot_corr_heatmap,
    plot_coverage,
    plot_ic_decay,
    plot_ic_series,
    plot_outlier_before_after,
    plot_quantile_nav,
    plot_stability,
    plot_turnover,
)
from factorlab.preprocess import apply_cs_standardize, apply_winsorize, handle_missing, neutralize_factor
from factorlab.research.diagnostics import (
    coverage_by_date,
    factor_corr_matrix,
    factor_stability,
    missing_rates,
    outlier_monitor,
)
from factorlab.research.advanced_metrics import (
    compute_factor_rank_autocorr,
    compute_long_short_alpha_beta,
    summarize_quantile_monotonicity,
    summarize_quantile_profile,
)
from factorlab.research.forward_returns import add_forward_returns
from factorlab.research.quantile import quantile_returns
from factorlab.research.report import render_report
from factorlab.research.statistics import build_ic_decay, compute_daily_ic, newey_west_tstat, summarize_ic
from factorlab.utils import get_logger

LOGGER = get_logger("factorlab.research")
ALLOWED_PREPROCESS_STEPS = {"winsorize", "standardize", "neutralize"}


class FactorResearchPipeline:
    """Generate factor research outputs (tables + figures + HTML)."""

    def __init__(self, config: ResearchConfig):
        self.config = config

    def run(self, panel: pd.DataFrame, factors: list[str], out_dir: str | Path) -> dict[str, Path]:
        out = Path(out_dir)
        assets_dir = out / "assets"
        tables_dir = out / "tables"
        assets_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)
        for col in panel.select_dtypes(include=["float64"]).columns:
            panel[col] = panel[col].astype("float32")
        panel = add_forward_returns(panel, horizons=self.config.horizons)

        summary_rows: list[dict[str, float | str | int]] = []
        figure_map: dict[str, list[Path]] = {}
        table_map: dict[str, list[Path]] = {}

        missing_tbl = missing_rates(panel, factors)
        miss_path = tables_dir / "missing_rates.csv"
        missing_tbl.to_csv(miss_path, index=False)
        table_map["global"] = [miss_path]

        outlier_rows = []

        raw_steps = [str(x).strip().lower() for x in self.config.preprocess_steps if str(x).strip()]
        preprocess_steps = [s for s in raw_steps if s in ALLOWED_PREPROCESS_STEPS]
        if not preprocess_steps:
            preprocess_steps = ["winsorize", "standardize", "neutralize"]
            LOGGER.warning(
                "Invalid preprocess_steps=%s. Fallback to default %s",
                self.config.preprocess_steps,
                preprocess_steps,
            )

        for fac in factors:
            LOGGER.info("Running factor research for: %s", fac)
            fac_raw = panel[fac].astype(float)
            fac_stage = fac_raw.copy()

            if "winsorize" in preprocess_steps and self.config.winsorize_enabled:
                fac_stage = apply_winsorize(
                    panel,
                    factor_col=fac,
                    method=self.config.winsorize_method,
                    lower_q=self.config.lower_q,
                    upper_q=self.config.upper_q,
                    mad_scale=self.config.mad_scale,
                )
            fac_outlier_after = fac_stage.copy()

            if "standardize" in preprocess_steps:
                fac_stage = apply_cs_standardize(
                    pd.DataFrame({"date": panel["date"], fac: fac_stage}),
                    col=fac,
                    method=self.config.standardization,
                )

            outlier_rows.append(outlier_monitor(fac_raw, fac_outlier_after, fac))

            variants: dict[str, pd.Series] = {"raw": fac_stage}
            if "neutralize" in preprocess_steps and self.config.neutralization.mode != "none":
                variants["neutralized"] = neutralize_factor(
                    panel.assign(**{fac: fac_stage}),
                    fac,
                    self.config.neutralization,
                )

            for variant_name, series in variants.items():
                panel[f"{fac}_{variant_name}"] = series

            fac_table_paths: list[Path] = []
            fac_fig_paths: list[Path] = []

            for variant in variants:
                col = f"{fac}_{variant}"
                fac_asset_dir = assets_dir / "factors" / fac / variant
                fac_table_dir = tables_dir / "factors" / fac / variant
                fac_asset_dir.mkdir(parents=True, exist_ok=True)
                fac_table_dir.mkdir(parents=True, exist_ok=True)

                # IC/RankIC across horizons
                horizon_rows = []
                daily_ic_primary: pd.DataFrame | None = None
                for h in self.config.horizons:
                    ret_col = f"fwd_ret_{h}"
                    tmp = handle_missing(panel[["date", "asset", col, ret_col]], cols=[col, ret_col], policy=self.config.missing_policy)
                    ic_daily = compute_daily_ic(tmp, factor_col=col, ret_col=ret_col)
                    if ic_daily.empty:
                        continue
                    ic_daily["ic_roll"] = ic_daily["ic"].rolling(self.config.ic_rolling_window, min_periods=5).mean()
                    if h == self.config.horizons[0]:
                        daily_ic_primary = ic_daily
                    stats = summarize_ic(ic_daily)
                    stats_row = {"factor": fac, "variant": variant, "horizon": h}
                    stats_row.update(stats)
                    horizon_rows.append(stats_row)
                    summary_rows.append(stats_row)

                    path_ic_csv = fac_table_dir / f"ic_daily_h{h}.csv"
                    ic_daily.to_csv(path_ic_csv, index=False)
                    fac_table_paths.append(path_ic_csv)

                if not horizon_rows or daily_ic_primary is None:
                    continue

                # decay table + plot
                decay_df = build_ic_decay(horizon_rows)
                decay_csv = fac_table_dir / "ic_decay.csv"
                decay_df.to_csv(decay_csv, index=False)
                fac_table_paths.append(decay_csv)
                fac_fig_paths.append(
                    plot_ic_decay(
                        decay_df,
                        fac_asset_dir / "ic_decay.png",
                        title=f"{fac} [{variant}] IC decay",
                    )
                )

                # primary horizon quantile analysis
                primary_ret_col = f"fwd_ret_{self.config.horizons[0]}"
                tmpq = handle_missing(
                    panel[["date", "asset", col, primary_ret_col]],
                    cols=[col, primary_ret_col],
                    policy=self.config.missing_policy,
                )
                q_daily, q_nav, q_turn = quantile_returns(
                    tmpq,
                    factor_col=col,
                    ret_col=primary_ret_col,
                    quantiles=self.config.quantiles,
                )

                q_daily_csv = fac_table_dir / "quantile_daily.csv"
                q_nav_csv = fac_table_dir / "quantile_nav.csv"
                q_turn_csv = fac_table_dir / "turnover.csv"
                q_daily.to_csv(q_daily_csv, index=False)
                q_nav.to_csv(q_nav_csv, index=False)
                q_turn.to_csv(q_turn_csv, index=False)
                fac_table_paths.extend([q_daily_csv, q_nav_csv, q_turn_csv])

                q_profile = summarize_quantile_profile(q_daily, annualization_days=252)
                q_profile_csv = fac_table_dir / "quantile_profile.csv"
                q_profile.to_csv(q_profile_csv, index=False)
                fac_table_paths.append(q_profile_csv)

                mono_stats = summarize_quantile_monotonicity(q_daily)
                ls_series = q_daily.set_index("date")["long_short"]
                market_series = tmpq.groupby("date")[primary_ret_col].mean().sort_index()
                ls_reg = compute_long_short_alpha_beta(ls_series, market_series, annualization_days=252)
                rank_ac = compute_factor_rank_autocorr(
                    panel[["date", "asset", col]].rename(columns={col: "factor"}),
                    factor_col="factor",
                    lag=1,
                )
                rank_ac_mean = float(rank_ac["rank_autocorr_lag1"].mean()) if not rank_ac.empty else float("nan")
                rank_ac_csv = fac_table_dir / "rank_autocorr_lag1.csv"
                rank_ac.to_csv(rank_ac_csv, index=False)
                fac_table_paths.append(rank_ac_csv)
                ls_diagnostics = {
                    **mono_stats,
                    **ls_reg,
                    "rank_autocorr_lag1_mean": rank_ac_mean,
                }
                ls_diag_csv = fac_table_dir / "long_short_diagnostics.csv"
                pd.DataFrame([ls_diagnostics]).to_csv(ls_diag_csv, index=False)
                fac_table_paths.append(ls_diag_csv)

                # Newey-West on long-short daily returns
                nw_t_ls, nw_p_ls = newey_west_tstat(q_daily["long_short"])
                ls_profile_row = q_profile[q_profile["bucket"] == "long_short"]
                ls_profile = ls_profile_row.iloc[0].to_dict() if not ls_profile_row.empty else {}
                summary_rows.append(
                    {
                        "factor": fac,
                        "variant": variant,
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
                        "nw_t_long_short": nw_t_ls,
                        "nw_p_long_short": nw_p_ls,
                        "ls_mean_ret": ls_profile.get("mean_ret", np.nan),
                        "ls_sharpe": ls_profile.get("sharpe", np.nan),
                        "ls_hit_rate": ls_profile.get("hit_rate", np.nan),
                        "ls_max_drawdown": ls_profile.get("max_drawdown", np.nan),
                        "quantile_monotonicity_mean": mono_stats.get("quantile_monotonicity_mean", np.nan),
                        "quantile_monotonicity_pos_ratio": mono_stats.get(
                            "quantile_monotonicity_pos_ratio", np.nan
                        ),
                        "ls_alpha_ann": ls_reg.get("ls_alpha_ann", np.nan),
                        "ls_beta": ls_reg.get("ls_beta", np.nan),
                        "ls_r2": ls_reg.get("ls_r2", np.nan),
                        "rank_autocorr_lag1_mean": rank_ac_mean,
                    }
                )

                # diagnostics
                cov = coverage_by_date(panel[["date", "asset", col]].rename(columns={col: "factor"}), "factor")
                cov_csv = fac_table_dir / "coverage.csv"
                cov.to_csv(cov_csv, index=False)
                fac_table_paths.append(cov_csv)

                st = factor_stability(panel[["date", "asset", col]].rename(columns={col: "factor"}), "factor")
                st_csv = fac_table_dir / "stability.csv"
                st.to_csv(st_csv, index=False)
                fac_table_paths.append(st_csv)

                # plots (key)
                fac_fig_paths.extend(
                    [
                        plot_ic_series(
                            daily_ic_primary,
                            fac_asset_dir / "ic.png",
                            title=f"{fac} [{variant}] IC",
                        ),
                        plot_quantile_nav(
                            q_nav,
                            fac_asset_dir / "quantile_nav.png",
                            title=f"{fac} [{variant}] Quantile NAV",
                        ),
                        plot_turnover(
                            q_turn,
                            fac_asset_dir / "turnover.png",
                            title=f"{fac} [{variant}] Turnover",
                        ),
                        plot_coverage(
                            cov,
                            fac_asset_dir / "coverage.png",
                            title=f"{fac} [{variant}] Coverage",
                        ),
                        plot_stability(
                            st,
                            fac_asset_dir / "stability.png",
                            title=f"{fac} [{variant}] Stability",
                        ),
                    ]
                )

            figure_map[fac] = fac_fig_paths
            table_map[fac] = fac_table_paths

        outlier_df = pd.concat(outlier_rows, ignore_index=True) if outlier_rows else pd.DataFrame()
        if not outlier_df.empty:
            outlier_csv = tables_dir / "outlier_before_after.csv"
            outlier_df.to_csv(outlier_csv, index=False)
            table_map.setdefault("global", []).append(outlier_csv)
            figure_map.setdefault("global", []).append(
                plot_outlier_before_after(outlier_df, assets_dir / "outlier_before_after.png")
            )

        # factor correlation (when >=2)
        if len(factors) >= 2:
            corr_s = factor_corr_matrix(panel, factors, method="spearman")
            corr_p = factor_corr_matrix(panel, factors, method="pearson")
            if not corr_s.empty:
                corr_s_csv = tables_dir / "factor_corr_spearman.csv"
                corr_p_csv = tables_dir / "factor_corr_pearson.csv"
                corr_s.to_csv(corr_s_csv)
                corr_p.to_csv(corr_p_csv)
                table_map.setdefault("global", []).extend([corr_s_csv, corr_p_csv])
                figure_map.setdefault("global", []).append(
                    plot_corr_heatmap(corr_s, assets_dir / "factor_corr_spearman.png", title="Factor Spearman Correlation")
                )

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
                    "standardization": self.config.standardization,
                    "winsorize_enabled": self.config.winsorize_enabled,
                    "winsorize_method": self.config.winsorize_method,
                    "missing_policy": self.config.missing_policy,
                    "preprocess_steps": preprocess_steps,
                    "neutralization_mode": self.config.neutralization.mode,
                    "factors": factors,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        index_html = render_report(out, summary=summary, figure_map=figure_map, table_map=table_map)
        LOGGER.info("Report generated at: %s", index_html)
        return {
            "index_html": index_html,
            "summary_csv": summary_path,
            "config_json": config_path,
            "assets_dir": assets_dir,
            "tables_dir": tables_dir,
        }
