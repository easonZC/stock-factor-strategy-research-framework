"""目录型本地面板读取测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from factorlab.data import read_panel_directory


def _write_single_asset_csv(path: Path, asset_seed: int) -> None:
    dates = pd.date_range("2021-01-01", periods=80, freq="B")
    close = 10.0 + asset_seed + pd.Series(range(len(dates)), dtype=float) * 0.1
    df = pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "volume": 1000 + asset_seed,
            "mkt_cap": 1_000_000 + asset_seed * 10_000,
            "industry": "Demo",
        }
    )
    df.to_csv(path, index=False)


def test_read_panel_directory_sets_asset_from_filename(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    _write_single_asset_csv(raw_dir / "asset_a.csv", asset_seed=1)
    _write_single_asset_csv(raw_dir / "asset_b.csv", asset_seed=2)

    panel, report = read_panel_directory(
        directory=raw_dir,
        return_report=True,
        asset_from_filename=True,
    )
    assert panel["asset"].nunique() == 2
    assert set(panel["asset"].unique()) == {"asset_a", "asset_b"}
    assert report.files_total == 2
    assert report.files_loaded == 2
    assert report.files_skipped == 0
