"""External data adapters with explicit warnings and schema mapping."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from factorlab.config import AdapterConfig
from factorlab.utils import get_logger

LOGGER = get_logger("factorlab.data.adapters")


def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", str(name).strip().lower())


def _column_aliases() -> dict[str, set[str]]:
    aliases = {
        "date": {"date", "trade_date", "datetime", "dt", "day", "日期", "交易日期"},
        "asset": {"asset", "symbol", "ticker", "code", "ts_code", "stock_code", "股票代码"},
        "open": {"open", "open_price", "开盘", "开盘价"},
        "high": {"high", "high_price", "最高", "最高价"},
        "low": {"low", "low_price", "最低", "最低价"},
        "close": {"close", "close_price", "收盘", "收盘价"},
        "volume": {"volume", "vol", "成交量"},
        "mkt_cap": {"mkt_cap", "market_cap", "总市值", "市值", "流通市值"},
        "industry": {"industry", "行业", "行业名称"},
    }
    return {k: {_norm_col(x) for x in v} for k, v in aliases.items()}


def _auto_map_columns(columns: list[str]) -> dict[str, str]:
    alias_map = _column_aliases()
    mapped: dict[str, str] = {}
    normalized_to_raw = {_norm_col(c): c for c in columns}
    for canonical, aliases in alias_map.items():
        for norm_name, raw_name in normalized_to_raw.items():
            if norm_name in aliases:
                mapped[canonical] = raw_name
                break
    return mapped


def _read_csv_robust(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]
    last_error: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # pragma: no cover - exercised by environment variance
            last_error = exc
    assert last_error is not None
    raise last_error


def prepare_sina_panel(config: AdapterConfig) -> pd.DataFrame:
    """Best-effort Sina folder adapter with explicit warnings on schema mismatch."""
    data_dir = Path(config.data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Sina data-dir not found or not a directory: {data_dir}")

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {data_dir}")

    required = [str(c).strip() for c in config.required_cols if str(c).strip()]
    required = required if required else ["date", "close"]

    parts: list[pd.DataFrame] = []
    skipped = 0
    for file in csv_files:
        try:
            raw = _read_csv_robust(file)
        except Exception as exc:
            LOGGER.warning("Skip file %s: failed to read csv (%s)", file, exc)
            skipped += 1
            continue

        if raw.empty:
            LOGGER.warning("Skip file %s: empty csv", file)
            skipped += 1
            continue

        mapping = _auto_map_columns(list(raw.columns))
        missing_required = [c for c in required if c not in mapping]
        if missing_required:
            LOGGER.warning(
                "Skip file %s: required columns missing after auto-mapping: %s. "
                "Detected mapping keys=%s raw columns=%s",
                file,
                missing_required,
                sorted(mapping.keys()),
                list(raw.columns),
            )
            skipped += 1
            continue

        out = pd.DataFrame(index=raw.index)
        out["date"] = pd.to_datetime(raw[mapping["date"]], errors="coerce")
        if "asset" in mapping:
            out["asset"] = raw[mapping["asset"]].astype(str).str.strip()
            out["asset"] = out["asset"].replace({"": file.stem, "nan": file.stem, "None": file.stem})
        else:
            out["asset"] = file.stem

        for col in ["open", "high", "low", "close", "volume", "mkt_cap"]:
            if col in mapping:
                out[col] = pd.to_numeric(raw[mapping[col]], errors="coerce")
            else:
                out[col] = np.nan
                LOGGER.warning("File %s missing optional column '%s' after auto-mapping.", file, col)

        if "industry" in mapping:
            out["industry"] = raw[mapping["industry"]].astype(str).replace({"nan": "Unknown"})
        else:
            out["industry"] = "Unknown"
            LOGGER.warning("File %s missing optional column 'industry' after auto-mapping.", file)

        out = out.dropna(subset=["date", "close"]).copy()
        if len(out) < int(config.min_rows_per_asset):
            LOGGER.warning(
                "Skip file %s: rows after clean (%s) < min_rows_per_asset (%s)",
                file,
                len(out),
                config.min_rows_per_asset,
            )
            skipped += 1
            continue

        parts.append(out)

    if not parts:
        raise ValueError(
            "Sina adapter produced no valid panel rows. "
            "Check folder schema and warnings for missing required columns."
        )

    panel = pd.concat(parts, ignore_index=True)
    panel = panel.drop_duplicates(subset=["date", "asset"], keep="last")
    panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)

    LOGGER.info(
        "Sina adapter completed: files_total=%s files_used=%s files_skipped=%s rows=%s assets=%s",
        len(csv_files),
        len(csv_files) - skipped,
        skipped,
        len(panel),
        panel["asset"].nunique(),
    )
    return panel
