"""Smoke tests for config-driven TS/CS one-click runner."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from factorlab.workflows import run_from_config


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_run_from_config_cs_smoke(tmp_path) -> None:
    cfg = {
        "run": {
            "factor_scope": "cs",
            "eval_axis": "cross_section",
            "standardization": "cs_rank",
        },
        "data": {
            "mode": "panel",
            "adapter": "synthetic",
            "fields_required": ["date", "asset", "close", "volume", "mkt_cap", "industry"],
            "synthetic": {
                "n_assets": 14,
                "n_days": 220,
                "seed": 110,
                "start_date": "2020-01-01",
            },
        },
        "factor": {"names": ["momentum_20", "volatility_20", "liquidity_shock", "size"]},
        "research": {
            "horizons": [1, 5, 10],
            "quantiles": 5,
            "ic_rolling_window": 20,
            "winsorize": {"enabled": True, "method": "quantile"},
            "neutralize": {"enabled": True, "mode": "both"},
        },
        "backtest": {"enabled": True, "strategy": {"mode": "longshort"}},
    }
    cfg_path = tmp_path / "cs.yaml"
    _write_yaml(cfg_path, cfg)

    out_dir = tmp_path / "out_cs"
    result = run_from_config(config=cfg_path, out_dir=out_dir)

    assert result.index_html.exists()
    assert result.summary_csv.exists()
    assert result.run_meta_json.exists()
    assert result.run_manifest_json.exists()
    assert result.backtest_summary_csv is not None and result.backtest_summary_csv.exists()

    meta = json.loads(result.run_meta_json.read_text(encoding="utf-8"))
    assert meta["scope"]["factor_scope"] == "cs"
    assert meta["scope"]["eval_axis"] == "cross_section"


def test_run_from_config_ts_smoke(tmp_path) -> None:
    cfg = {
        "run": {
            "factor_scope": "ts",
            "eval_axis": "time",
            "standardization": "ts_rolling_zscore",
        },
        "data": {
            "mode": "single_asset",
            "adapter": "synthetic",
            "fields_required": ["date", "close"],
            "synthetic": {
                "n_assets": 1,
                "n_days": 280,
                "seed": 123,
                "start_date": "2020-01-01",
            },
        },
        "factor": {"names": ["momentum_20", "volatility_20"]},
        "research": {
            "horizons": [1, 5, 10],
            "quantiles": 5,
            "ic_rolling_window": 20,
            "ts_standardize_window": 50,
            "ts_quantile_lookback": 70,
        },
        "backtest": {"enabled": True, "strategy": {"mode": "sign", "sign_threshold": 0.0}},
    }
    cfg_path = tmp_path / "ts.yaml"
    _write_yaml(cfg_path, cfg)

    out_dir = tmp_path / "out_ts"
    result = run_from_config(config=cfg_path, out_dir=out_dir)

    assert result.index_html.exists()
    assert result.summary_csv.exists()
    assert result.run_meta_json.exists()
    assert result.run_manifest_json.exists()
    assert result.backtest_summary_csv is not None and result.backtest_summary_csv.exists()

    meta = json.loads(result.run_meta_json.read_text(encoding="utf-8"))
    assert meta["scope"]["factor_scope"] == "ts"
    assert meta["scope"]["eval_axis"] == "time"
