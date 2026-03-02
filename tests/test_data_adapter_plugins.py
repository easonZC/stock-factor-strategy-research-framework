"""模块说明。"""

from __future__ import annotations

import json
from pathlib import Path

from factorlab.data import build_data_adapter_registry, build_data_adapter_validator_registry
from factorlab.data.adapters import prepare_sina_panel, validate_sina_config
from factorlab.workflows import run_from_config


def _write_data_adapter_plugin(plugin_dir: Path) -> Path:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    path = plugin_dir / "mock_adapter.py"
    path.write_text(
        """
import pandas as pd
from factorlab.config import AdapterConfig

def validate_mock_config(config: AdapterConfig) -> None:
    if not config.symbols:
        raise ValueError("mock adapter requires at least one symbol")
    if int(config.min_rows_per_asset) <= 0:
        raise ValueError("mock adapter requires min_rows_per_asset > 0")

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


def _write_duplicate_data_adapter_plugin(plugin_dir: Path) -> Path:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    path = plugin_dir / "dup_adapter.py"
    path.write_text(
        """
import pandas as pd
from factorlab.config import AdapterConfig

def validate_sina_config(config: AdapterConfig) -> None:
    return None

def prepare_sina_panel(config: AdapterConfig) -> pd.DataFrame:
    return pd.DataFrame({"date": [], "asset": [], "close": []})
""".strip(),
        encoding="utf-8",
    )
    return path


def test_build_data_adapter_registry_with_plugin_dir(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    _write_data_adapter_plugin(plugin_dir)
    reg = build_data_adapter_registry(plugin_dirs=[plugin_dir], include_defaults=False, on_plugin_error="raise")
    assert "mock" in reg
    vreg = build_data_adapter_validator_registry(
        plugin_dirs=[plugin_dir],
        include_defaults=False,
        on_plugin_error="raise",
    )
    assert "mock" in vreg


def test_build_data_adapter_registry_warn_skip_keeps_builtin_on_duplicate(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    _write_duplicate_data_adapter_plugin(plugin_dir)

    reg = build_data_adapter_registry(plugin_dirs=[plugin_dir], include_defaults=True, on_plugin_error="warn_skip")
    vreg = build_data_adapter_validator_registry(
        plugin_dirs=[plugin_dir],
        include_defaults=True,
        on_plugin_error="warn_skip",
    )
    assert reg["sina"] is prepare_sina_panel
    assert vreg["sina"] is validate_sina_config


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
            "symbols": ["AAA"],
            "min_rows_per_asset": 20,
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
    assert "mock" in meta["data"]["adapter_validator_plugin_config"]["registry_validators"]
    assert meta["data"]["adapter_validation_report"]["validated"] is True
    assert meta["data"]["load_report"]["panel_profile"]["source"] == "adapter"
    assert float(meta["data"]["load_report"]["adapter_load_seconds"]) >= 0.0
    assert Path(meta["outputs"]["adapter_quality_audit_csv"]).exists()


def test_run_from_config_custom_adapter_validation_hook_blocks_bad_config(tmp_path: Path) -> None:
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
            "symbols": [],
            "fields_required": ["date", "asset", "close", "volume", "mkt_cap", "industry"],
        },
        "factor": {"names": ["momentum_20"]},
        "research": {"horizons": [1, 5], "quantiles": 5, "ic_rolling_window": 20},
        "backtest": {"enabled": False},
    }

    try:
        run_from_config(cfg, out_dir=tmp_path / "out_bad_adapter")
    except ValueError as exc:
        assert "requires at least one symbol" in str(exc)
    else:
        raise AssertionError("Expected adapter validation hook to block invalid config")
