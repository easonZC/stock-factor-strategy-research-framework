"""本地面板数据读写与清洗工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


DuplicatePolicy = Literal["last", "first", "raise"]


@dataclass(slots=True)
class PanelSanitizationConfig:
    """面板清洗配置。"""

    duplicate_policy: DuplicatePolicy = "last"
    sort_values: bool = True


@dataclass(slots=True)
class PanelSanitizationReport:
    """面板清洗结果摘要。"""

    original_rows: int
    rows_after_sanitization: int
    dropped_invalid_date_rows: int
    dropped_missing_asset_rows: int
    duplicate_rows_detected: int
    duplicate_rows_dropped: int
    duplicate_policy: DuplicatePolicy


@dataclass(slots=True)
class PanelDirectoryReadReport:
    """目录读取结果摘要。"""

    directory: str
    pattern: str
    files_total: int
    files_loaded: int
    files_skipped: int
    rows_before_sanitization: int
    rows_after_sanitization: int
    asset_from_filename: bool


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


def _read_frame_by_suffix(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported panel format: {path.suffix}. Use .parquet or .csv")


def _parse_glob_patterns(pattern: str) -> list[str]:
    parts = [str(x).strip() for x in str(pattern).split(",") if str(x).strip()]
    return parts if parts else ["*.parquet", "*.csv"]


def _collect_input_files(directory: Path, pattern: str) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    for glob_pat in _parse_glob_patterns(pattern):
        for path in sorted(directory.glob(glob_pat)):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in {".csv", ".parquet"}:
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            files.append(path)
            seen.add(key)
    return files


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
    """读取单文件面板（CSV/Parquet）。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Panel file not found: {p}")

    panel = _read_frame_by_suffix(p)

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


def read_panel_directory(
    directory: str | Path,
    pattern: str = "*.parquet,*.csv",
    sanitize: bool = True,
    sanitization_config: PanelSanitizationConfig | None = None,
    return_report: bool = False,
    asset_from_filename: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, PanelDirectoryReadReport]:
    """读取目录下多个 CSV/Parquet 文件并合并成标准面板。"""
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"Panel directory not found or not a directory: {dir_path}")

    files = _collect_input_files(dir_path, pattern=pattern)
    if not files:
        raise FileNotFoundError(
            f"No parquet/csv files found under {dir_path} with pattern '{pattern}'."
        )

    parts: list[pd.DataFrame] = []
    skipped = 0
    for file in files:
        try:
            raw = _read_frame_by_suffix(file)
            had_asset_col = "asset" in raw.columns
            frame = _ensure_date_column(raw)
            frame = _ensure_asset_column(frame)
            if asset_from_filename and not had_asset_col:
                frame["asset"] = str(file.stem)
            parts.append(frame)
        except Exception:
            skipped += 1
            continue

    if not parts:
        raise ValueError(f"No valid files could be parsed from directory: {dir_path}")

    merged = pd.concat(parts, ignore_index=True)
    rows_before = int(len(merged))
    if sanitize:
        cfg = sanitization_config or PanelSanitizationConfig()
        cleaned, _ = _sanitize_panel(merged, cfg=cfg)
    else:
        cleaned = _ensure_date_column(merged)
        cleaned = _ensure_asset_column(cleaned)
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
        cleaned["asset"] = cleaned["asset"].astype(str)
        cleaned = cleaned.dropna(subset=["date"]).sort_values(["date", "asset"]).reset_index(drop=True)

    report = PanelDirectoryReadReport(
        directory=str(dir_path),
        pattern=pattern,
        files_total=len(files),
        files_loaded=len(files) - skipped,
        files_skipped=skipped,
        rows_before_sanitization=rows_before,
        rows_after_sanitization=int(len(cleaned)),
        asset_from_filename=bool(asset_from_filename),
    )
    if return_report:
        return cleaned, report
    return cleaned


def write_panel(panel: pd.DataFrame, path: str | Path) -> Path:
    """写出单文件面板（CSV/Parquet）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".parquet":
        panel.to_parquet(p, index=False)
    elif p.suffix.lower() == ".csv":
        panel.to_csv(p, index=False)
    else:
        raise ValueError(f"Unsupported output format: {p.suffix}. Use .parquet or .csv")
    return p
