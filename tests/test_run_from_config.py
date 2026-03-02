"""配置驱动 TS/CS 一键运行冒烟测试。"""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse
from pathlib import Path

import pandas as pd
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
    assert "Data Adapter Audit" in result.index_html.read_text(encoding="utf-8")

    meta = json.loads(result.run_meta_json.read_text(encoding="utf-8"))
    assert meta["scope"]["factor_scope"] == "cs"
    assert meta["scope"]["eval_axis"] == "cross_section"
    assert "timings_seconds" in meta and "research" in meta["timings_seconds"]
    assert "warning_summary" in meta
    assert "panel_profile" in meta["data"]["load_report"]
    assert "adapter_load_seconds" in meta["data"]["load_report"]
    assert "adapter_validation_report" in meta["data"]
    assert Path(meta["outputs"]["adapter_quality_audit_csv"]).exists()
    assert "transform_plugin_config" in meta["research"]


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
    assert "timings_seconds" in meta and "backtest" in meta["timings_seconds"]
    assert "warning_summary" in meta
    assert "panel_profile" in meta["data"]["load_report"]
    assert "adapter_load_seconds" in meta["data"]["load_report"]
    assert "adapter_quality_audit_csv" in meta["outputs"]
    assert Path(meta["outputs"]["adapter_quality_audit_csv"]).exists()
    assert "custom_transform_report" in meta["research"]


def test_run_from_config_warn_skip_factor_and_flexible_preprocess(tmp_path) -> None:
    cfg = {
        "run": {
            "factor_scope": "cs",
            "eval_axis": "cross_section",
            "standardization": "cs_robust_zscore",
        },
        "data": {
            "mode": "panel",
            "adapter": "synthetic",
            "fields_required": ["date", "asset", "close", "volume", "mkt_cap", "industry"],
            "synthetic": {
                "n_assets": 10,
                "n_days": 180,
                "seed": 101,
                "start_date": "2021-01-01",
            },
        },
        "factor": {"names": ["momentum_20", "not_exists_factor"], "on_missing": "warn_skip"},
        "research": {
            "horizons": [1, 5],
            "quantiles": 5,
            "ic_rolling_window": 20,
            "missing_policy": "fill_zero",
            "preprocess_steps": ["standardize"],
            "neutralize": {"enabled": False},
        },
        "backtest": {"enabled": False},
    }
    cfg_path = tmp_path / "flex.yaml"
    _write_yaml(cfg_path, cfg)

    out_dir = tmp_path / "out_flex"
    result = run_from_config(config=cfg_path, out_dir=out_dir)
    assert result.index_html.exists()
    assert result.summary_csv.exists()

    meta = json.loads(result.run_meta_json.read_text(encoding="utf-8"))
    assert meta["factors"]["requested"] == ["momentum_20", "not_exists_factor"]
    assert meta["factors"]["effective"] == ["momentum_20"]
    assert meta["factors"]["on_missing"] == "warn_skip"


def test_run_from_config_cs_meanvar_and_regression_outputs(tmp_path) -> None:
    cfg = {
        "run": {
            "factor_scope": "cs",
            "eval_axis": "cross_section",
            "standardization": "cs_zscore",
        },
        "data": {
            "mode": "panel",
            "adapter": "synthetic",
            "fields_required": ["date", "asset", "close", "volume", "mkt_cap", "industry"],
            "synthetic": {
                "n_assets": 12,
                "n_days": 180,
                "seed": 907,
                "start_date": "2020-01-01",
            },
        },
        "factor": {"names": ["momentum_20", "volatility_20"]},
        "research": {
            "horizons": [1, 5],
            "quantiles": 5,
            "ic_rolling_window": 20,
            "winsorize": {"enabled": True, "method": "quantile"},
            "neutralize": {"enabled": True, "mode": "both"},
        },
        "backtest": {
            "enabled": True,
            "strategy": {
                "mode": "meanvar",
                "risk_aversion": 4.0,
                "gross_target": 1.2,
                "net_target": 0.0,
                "max_weight": 0.2,
            },
            "leverage": 1.2,
            "max_turnover": 0.8,
            "max_abs_weight": 0.2,
            "max_gross_exposure": 1.2,
            "max_net_exposure": 0.3,
            "benchmark_mode": "cross_sectional_mean",
        },
    }
    cfg_path = tmp_path / "cs_meanvar.yaml"
    _write_yaml(cfg_path, cfg)

    out_dir = tmp_path / "out_cs_meanvar"
    result = run_from_config(config=cfg_path, out_dir=out_dir)
    assert result.index_html.exists()
    assert result.backtest_summary_csv is not None and result.backtest_summary_csv.exists()
    bt_summary = pd.read_csv(Path(result.backtest_summary_csv))
    assert set(bt_summary["strategy_mode"]) == {"meanvar"}

    fmb_summary = out_dir / "tables" / "factors" / "momentum_20" / "raw" / "fama_macbeth_summary.csv"
    industry_summary = out_dir / "tables" / "factors" / "momentum_20" / "raw" / "industry_decomposition_summary.csv"
    style_summary = out_dir / "tables" / "factors" / "momentum_20" / "raw" / "style_decomposition_summary.csv"
    assert fmb_summary.exists()
    assert industry_summary.exists()
    assert style_summary.exists()


def test_run_from_config_with_factor_combinations(tmp_path) -> None:
    cfg = {
        "run": {
            "factor_scope": "cs",
            "eval_axis": "cross_section",
            "standardization": "cs_zscore",
        },
        "data": {
            "mode": "panel",
            "adapter": "synthetic",
            "fields_required": ["date", "asset", "close", "volume", "mkt_cap", "industry"],
            "synthetic": {
                "n_assets": 10,
                "n_days": 160,
                "seed": 517,
                "start_date": "2020-01-01",
            },
        },
        "factor": {
            "names": ["combo_mom_vol"],
            "combinations": [
                {
                    "name": "combo_mom_vol",
                    "weights": {"momentum_20": 1.0, "volatility_20": -0.5},
                    "standardization": "cs_zscore",
                    "orthogonalize_to": ["size"],
                }
            ],
        },
        "research": {
            "horizons": [1, 5],
            "quantiles": 5,
            "ic_rolling_window": 20,
            "winsorize": {"enabled": True, "method": "quantile"},
            "neutralize": {"enabled": True, "mode": "both"},
        },
        "backtest": {"enabled": False},
    }
    cfg_path = tmp_path / "combo.yaml"
    _write_yaml(cfg_path, cfg)
    out_dir = tmp_path / "out_combo"
    result = run_from_config(config=cfg_path, out_dir=out_dir)
    assert result.index_html.exists()
    summary = pd.read_csv(result.summary_csv)
    assert "combo_mom_vol" in set(summary["factor"].astype(str))

    meta = json.loads(result.run_meta_json.read_text(encoding="utf-8"))
    assert "combo_mom_vol" in meta["factors"]["computed_combination_factors"]
    assert "combo_mom_vol" in meta["factors"]["effective"]


def test_run_from_config_stooq_adapter_smoke(tmp_path, monkeypatch) -> None:
    def _make_payload(symbol_raw: str) -> str:
        symbol = str(symbol_raw).split(".")[0].upper()
        base = {"AAPL": 100.0, "MSFT": 200.0, "GOOGL": 150.0, "AMZN": 120.0}.get(symbol, 90.0)
        dates = pd.date_range("2023-01-02", periods=220, freq="B")
        close = base + pd.Series(range(len(dates)), dtype=float) * 0.05
        df = pd.DataFrame(
            {
                "Date": dates.strftime("%Y-%m-%d"),
                "Open": close - 0.1,
                "High": close + 0.2,
                "Low": close - 0.2,
                "Close": close,
                "Volume": 1000000,
            }
        )
        return df.to_csv(index=False)

    class _Resp:
        def __init__(self, payload: str) -> None:
            self._payload = payload.encode("utf-8")

        def read(self) -> bytes:
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_urlopen(url: str, timeout: int):  # noqa: ANN001
        assert "stooq.com" in url
        assert timeout == 20
        q = parse_qs(urlparse(url).query)
        symbol = q.get("s", ["aapl.us"])[0]
        return _Resp(_make_payload(symbol))

    monkeypatch.setattr("factorlab.data.adapters.urlopen", _fake_urlopen)

    cfg = {
        "run": {"factor_scope": "cs", "eval_axis": "cross_section", "standardization": "cs_rank"},
        "data": {
            "mode": "panel",
            "adapter": "stooq",
            "symbols": ["aapl", "msft", "googl", "amzn"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "request_timeout_sec": 20,
            "fields_required": ["date", "asset", "close", "volume"],
        },
        "factor": {"names": ["momentum_20", "volatility_20"]},
        "research": {
            "horizons": [1, 5],
            "quantiles": 3,
            "ic_rolling_window": 20,
            "preprocess_steps": ["standardize"],
            "neutralize": {"enabled": False},
        },
        "backtest": {"enabled": False},
    }
    cfg_path = tmp_path / "stooq.yaml"
    _write_yaml(cfg_path, cfg)
    out_dir = tmp_path / "out_stooq"
    result = run_from_config(config=cfg_path, out_dir=out_dir)
    assert result.index_html.exists()
    assert result.summary_csv.exists()


def test_run_from_config_sanitizes_factor_output_paths(tmp_path) -> None:
    malicious = "../../../../evil_factor"
    cfg = {
        "run": {"factor_scope": "cs", "eval_axis": "cross_section", "standardization": "cs_zscore"},
        "data": {
            "mode": "panel",
            "adapter": "synthetic",
            "fields_required": ["date", "asset", "close", "volume", "mkt_cap", "industry"],
            "synthetic": {"n_assets": 10, "n_days": 140, "seed": 77, "start_date": "2020-01-01"},
        },
        "factor": {
            "names": [malicious],
            "expressions": {malicious: "momentum_20 - volatility_20"},
            "expression_on_error": "raise",
            "on_missing": "raise",
        },
        "research": {
            "horizons": [1, 5],
            "quantiles": 5,
            "ic_rolling_window": 20,
            "winsorize": {"enabled": True, "method": "quantile"},
            "neutralize": {"enabled": True, "mode": "both"},
        },
        "backtest": {"enabled": False},
    }
    out_dir = tmp_path / "out_path_safe"
    result = run_from_config(config=cfg, out_dir=out_dir)
    assert result.index_html.exists()
    # 若目录拼接存在路径穿越，会在 out_dir 外生成目录；这里确保不会发生。
    assert not (tmp_path / "evil_factor").exists()
