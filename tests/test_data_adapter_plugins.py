"""Tests for data adapter plugin discovery and config integration."""

from __future__ import annotations

import json
from pathlib import Path

from factorlab.data import build_data_adapter_registry
from factorlab.workflows import run_from_config


def _write_data_adapter_plugin(plugin_dir: Path) -> Path:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    path = plugin_dir / "mock_adapter.py"
    path.write_text(
        """
import pandas as pd
from factorlab.config import AdapterConfig

def prepare_mock_panel(config: AdapterConfig) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=160, freq="B")
    assets = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    rows = []
    for dt in dates:
        for i, asset in enumerate(assets):
            close = 50.0 + i * 0.7 + float((dt - dates[0]).days) * 0.03
            rows.append(
                {
                    "date": dt,
                    "asset": asset,
                    "open": close - 0.1,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1000000 + i * 1000,
                    "mkt_cap": close * 1e8,
                    "industry": "Demo",
                }
            )
    return pd.DataFrame(rows)
""".strip(),
        encoding="utf-8",
    )
    return path


def test_build_data_adapter_registry_with_plugin_dir(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    _write_data_adapter_plugin(plugin_dir)
    reg = build_data_adapter_registry(plugin_dirs=[plugin_dir], include_defaults=False, on_plugin_error="raise")
    assert "mock" in reg


def test_run_from_config_with_custom_data_adapter_plugin(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    _write_data_adapter_plugin(plugin_dir)

    cfg = {
        "run": {
            "factor_scope": "cs",
            "eval_axis": "cross_section",
            "standardization": "cs_rank",
        },
        "data": {
            "mode": "panel",
            "adapter": "mock",
            "adapter_auto_discover": True,
            "adapter_plugin_dirs": [str(plugin_dir)],
            "adapter_plugin_on_error": "raise",
            "fields_required": ["date", "asset", "close", "volume", "mkt_cap", "industry"],
        },
        "factor": {"names": ["momentum_20", "volatility_20"]},
        "research": {
            "horizons": [1, 5],
            "quantiles": 5,
            "ic_rolling_window": 20,
            "winsorize": {"enabled": True, "method": "quantile"},
            "neutralize": {"enabled": True, "mode": "both"},
        },
        "backtest": {"enabled": False},
    }
    out_dir = tmp_path / "out_adapter_plugin"
    res = run_from_config(cfg, out_dir=out_dir)
    assert res.index_html.exists()
    meta = json.loads(res.run_meta_json.read_text(encoding="utf-8"))
    assert meta["data"]["config"]["adapter"] == "mock"
    assert "mock" in meta["data"]["adapter_plugin_config"]["registry_adapters"]

