"""Adapter behavior tests for Sina CSV ingestion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from factorlab.config import AdapterConfig
from factorlab.data.adapters import prepare_sina_panel


def test_sina_adapter_basic(tmp_path: Path) -> None:
    data_dir = tmp_path / "sina"
    data_dir.mkdir(parents=True)

    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=40, freq="B"),
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
            "volume": 1000,
            "mkt_cap": 1e9,
            "industry": "Tech",
        }
    )
    df.to_csv(data_dir / "asset_a.csv", index=False)

    panel = prepare_sina_panel(AdapterConfig(data_dir=str(data_dir), min_rows_per_asset=10))
    assert not panel.empty
    assert {"date", "asset", "close"}.issubset(panel.columns)
