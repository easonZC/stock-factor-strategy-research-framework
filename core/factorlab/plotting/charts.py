"""Chart utilities used by the HTML report generator."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from factorlab.plotting.style import apply_style


def _save(fig: plt.Figure, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_ic_series(ic_df: pd.DataFrame, out_path: Path, title: str = "IC / RankIC") -> Path:
    apply_style()
    fig, ax = plt.subplots()
    ax.plot(ic_df["date"], ic_df["ic"], label="IC", alpha=0.75)
    ax.plot(ic_df["date"], ic_df["rank_ic"], label="RankIC", alpha=0.75)
    if "ic_roll" in ic_df:
        ax.plot(ic_df["date"], ic_df["ic_roll"], label="IC Rolling", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation")
    ax.legend()
    return _save(fig, out_path)


def plot_ic_decay(decay: pd.DataFrame, out_path: Path, title: str = "IC Decay") -> Path:
    apply_style()
    fig, ax = plt.subplots()
    ax.plot(decay["horizon"], decay["ic_mean"], marker="o", label="IC")
    ax.plot(decay["horizon"], decay["rank_ic_mean"], marker="o", label="RankIC")
    ax.set_title(title)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Mean IC")
    ax.legend()
    return _save(fig, out_path)


def plot_quantile_nav(nav_df: pd.DataFrame, out_path: Path, title: str = "Quantile NAV") -> Path:
    apply_style()
    fig, ax = plt.subplots()
    for col in nav_df.columns:
        if col == "date":
            continue
        ax.plot(nav_df["date"], nav_df[col], label=col)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative NAV")
    ax.legend(ncol=3)
    return _save(fig, out_path)


def plot_turnover(turnover_df: pd.DataFrame, out_path: Path, title: str = "Quantile Turnover") -> Path:
    apply_style()
    fig, ax = plt.subplots()
    for col in turnover_df.columns:
        if col == "date":
            continue
        ax.plot(turnover_df["date"], turnover_df[col], label=col)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover")
    ax.legend(ncol=3)
    return _save(fig, out_path)


def plot_coverage(coverage_df: pd.DataFrame, out_path: Path, title: str = "Coverage") -> Path:
    apply_style()
    fig, ax = plt.subplots()
    ax.plot(coverage_df["date"], coverage_df["coverage"], label="Coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Coverage ratio")
    ax.legend()
    return _save(fig, out_path)


def plot_outlier_before_after(stats_df: pd.DataFrame, out_path: Path, title: str = "Outlier Monitoring") -> Path:
    apply_style()
    fig, ax = plt.subplots()
    x = np.arange(len(stats_df))
    w = 0.36
    ax.bar(x - w / 2, stats_df["before_std"], width=w, label="Before std")
    ax.bar(x + w / 2, stats_df["after_std"], width=w, label="After std")
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df["factor"], rotation=20)
    ax.set_title(title)
    ax.legend()
    return _save(fig, out_path)


def plot_stability(stability_df: pd.DataFrame, out_path: Path, title: str = "Factor Stability") -> Path:
    apply_style()
    fig, ax = plt.subplots()
    ax.plot(stability_df["date"], stability_df["autocorr_lag1"], label="lag1")
    ax.plot(stability_df["date"], stability_df["autocorr_lag5"], label="lag5")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Autocorr")
    ax.legend()
    return _save(fig, out_path)


def plot_corr_heatmap(corr: pd.DataFrame, out_path: Path, title: str = "Factor Correlation") -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, out_path)


def plot_group_bar(
    df: pd.DataFrame,
    out_path: Path,
    label_col: str,
    value_col: str,
    title: str,
) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    tmp = df.copy()
    if tmp.empty or label_col not in tmp.columns or value_col not in tmp.columns:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return _save(fig, out_path)

    tmp = tmp.sort_values(value_col, ascending=False)
    x = np.arange(len(tmp))
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in tmp[value_col].astype(float).values]
    ax.bar(x, tmp[value_col].astype(float).values, color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(tmp[label_col].astype(str).tolist(), rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(value_col)
    return _save(fig, out_path)


def plot_lag_profile(
    lag_df: pd.DataFrame,
    out_path: Path,
    title: str = "Signal Lag IC",
) -> Path:
    apply_style()
    fig, ax = plt.subplots()
    if lag_df.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return _save(fig, out_path)
    ax.plot(lag_df["lag"], lag_df["ic_mean"], marker="o", label="IC")
    if "rank_ic_mean" in lag_df.columns:
        ax.plot(lag_df["lag"], lag_df["rank_ic_mean"], marker="o", label="RankIC")
    ax.axhline(0.0, color="#777777", linewidth=1.0, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Execution Lag (days)")
    ax.set_ylabel("Mean IC")
    ax.legend()
    return _save(fig, out_path)
