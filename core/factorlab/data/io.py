"""Panel IO and sanitization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


DuplicatePolicy = Literal["last", "first", "raise"]


@dataclass(slots=True)
class PanelSanitizationConfig:
    """Controls panel sanitization behavior for file inputs."""

    duplicate_policy: DuplicatePolicy = "last"
    sort_values: bool = True


@dataclass(slots=True)
class PanelSanitizationReport:
    """Summary of sanitization effects."""

    original_rows: int
    rows_after_sanitization: int
    dropped_invalid_date_rows: int
    dropped_missing_asset_rows: int
    duplicate_rows_detected: int
    duplicate_rows_dropped: int
    duplicate_policy: DuplicatePolicy


def _ensure_asset_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "asset" in out.columns:
        return out
    candidates = ["symbol", "ticker", "code", "ts_code", "sec_code", "stock_code"]
    for col in candidates:
        if col in out.columns:
            out = out.rename(columns={col: "asset"})
            return out
    out["asset"] = "SINGLE_ASSET"
    return out


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        return out
    candidates = ["trade_date", "datetime", "dt", "day", "日期", "交易日期"]
    for col in candidates:
        if col in out.columns:
            out = out.rename(columns={col: "date"})
            return out
    raise KeyError("Input panel is missing required date column ('date' or aliases).")


def _sanitize_panel(df: pd.DataFrame, cfg: PanelSanitizationConfig) -> tuple[pd.DataFrame, PanelSanitizationReport]:
    out = _ensure_date_column(df)
    out = _ensure_asset_column(out)
    original_rows = int(len(out))

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    invalid_date_mask = out["date"].isna()
    dropped_invalid_date_rows = int(invalid_date_mask.sum())
    if dropped_invalid_date_rows > 0:
        out = out[~invalid_date_mask].copy()

    out["asset"] = out["asset"].astype(str).str.strip()
    missing_asset_mask = out["asset"].isin({"", "nan", "None"})
    dropped_missing_asset_rows = int(missing_asset_mask.sum())
    if dropped_missing_asset_rows > 0:
        out = out[~missing_asset_mask].copy()

    dup_mask = out.duplicated(subset=["date", "asset"], keep=False)
    duplicate_rows_detected = int(dup_mask.sum())
    duplicate_rows_dropped = 0
    if duplicate_rows_detected > 0:
        if cfg.duplicate_policy == "raise":
            dup_rows = out.loc[dup_mask, ["date", "asset"]].head(10)
            raise ValueError(
                "Duplicate (date, asset) rows found with duplicate_policy='raise'. "
                f"Sample:\n{dup_rows}"
            )
        keep = cfg.duplicate_policy
        before = len(out)
        out = out.drop_duplicates(subset=["date", "asset"], keep=keep).copy()
        duplicate_rows_dropped = int(before - len(out))

    if cfg.sort_values:
        out = out.sort_values(["date", "asset"]).reset_index(drop=True)

    report = PanelSanitizationReport(
        original_rows=original_rows,
        rows_after_sanitization=int(len(out)),
        dropped_invalid_date_rows=dropped_invalid_date_rows,
        dropped_missing_asset_rows=dropped_missing_asset_rows,
        duplicate_rows_detected=duplicate_rows_detected,
        duplicate_rows_dropped=duplicate_rows_dropped,
        duplicate_policy=cfg.duplicate_policy,
    )
    return out, report


def read_panel(
    path: str | Path,
    sanitize: bool = True,
    sanitization_config: PanelSanitizationConfig | None = None,
    return_report: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, PanelSanitizationReport]:
    """Read panel from parquet/csv with optional sanitization."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Panel file not found: {p}")

    if p.suffix.lower() == ".parquet":
        panel = pd.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        panel = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported panel format: {p.suffix}. Use .parquet or .csv")

    if not sanitize:
        panel = _ensure_date_column(panel)
        panel = _ensure_asset_column(panel)
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
        panel["asset"] = panel["asset"].astype(str)
        panel = panel.dropna(subset=["date"]).sort_values(["date", "asset"]).reset_index(drop=True)
        if return_report:
            report = PanelSanitizationReport(
                original_rows=int(len(panel)),
                rows_after_sanitization=int(len(panel)),
                dropped_invalid_date_rows=0,
                dropped_missing_asset_rows=0,
                duplicate_rows_detected=0,
                duplicate_rows_dropped=0,
                duplicate_policy="last",
            )
            return panel, report
        return panel

    cfg = sanitization_config or PanelSanitizationConfig()
    sanitized, report = _sanitize_panel(panel, cfg=cfg)
    if return_report:
        return sanitized, report
    return sanitized


def write_panel(panel: pd.DataFrame, path: str | Path) -> Path:
    """Write panel to parquet/csv inferred by extension."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".parquet":
        panel.to_parquet(p, index=False)
    elif p.suffix.lower() == ".csv":
        panel.to_csv(p, index=False)
    else:
        raise ValueError(f"Unsupported output format: {p.suffix}. Use .parquet or .csv")
    return p
