"""Tests for config schema pre-validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from factorlab.workflows import run_from_config, validate_run_config_schema


def _valid_cfg() -> dict:
    return {
        "run": {"factor_scope": "cs", "eval_axis": "cross_section", "standardization": "cs_zscore"},
        "data": {
            "mode": "panel",
            "adapter": "synthetic",
            "fields_required": ["date", "asset", "close", "volume", "mkt_cap", "industry"],
            "synthetic": {"n_assets": 8, "n_days": 120, "seed": 2, "start_date": "2020-01-01"},
        },
        "factor": {"names": ["momentum_20"], "on_missing": "raise"},
        "research": {
            "horizons": [1, 5],
            "quantiles": 5,
            "ic_rolling_window": 20,
            "missing_policy": "drop",
            "preprocess_steps": ["winsorize", "standardize", "neutralize"],
            "winsorize": {"enabled": True, "method": "quantile"},
            "neutralize": {"enabled": True, "mode": "both"},
        },
        "backtest": {"enabled": False},
    }


def test_validate_run_config_schema_unknown_root_warns() -> None:
    cfg = _valid_cfg()
    cfg["unknown_section"] = {"foo": 1}
    warnings = validate_run_config_schema(cfg, strict=True)
    assert any("unknown_section" in x for x in warnings)


def test_validate_run_config_schema_invalid_scope_raises() -> None:
    cfg = _valid_cfg()
    cfg["run"]["factor_scope"] = "invalid_scope"
    with pytest.raises(ValueError, match="run.factor_scope"):
        validate_run_config_schema(cfg, strict=True)


def test_validate_run_config_schema_invalid_expression_raises() -> None:
    cfg = _valid_cfg()
    cfg["factor"]["expressions"] = {"bad_expr": "__import__('os').system('echo x')"}
    with pytest.raises(ValueError, match="factor.expressions"):
        validate_run_config_schema(cfg, strict=True)


def test_validate_run_config_schema_invalid_combination_raises() -> None:
    cfg = _valid_cfg()
    cfg["factor"]["combinations"] = [{"name": "bad_combo", "weights": {"momentum_20": "x"}}]
    with pytest.raises(ValueError, match="factor.combinations"):
        validate_run_config_schema(cfg, strict=True)


def test_run_from_config_can_skip_schema_validation(tmp_path: Path) -> None:
    cfg = _valid_cfg()
    cfg["research"]["preprocess_steps"] = ["winsorize", "bad_step"]
    with pytest.raises(ValueError, match="research.preprocess_steps"):
        run_from_config(cfg, out_dir=tmp_path / "strict_fail")

    res = run_from_config(cfg, out_dir=tmp_path / "skip_ok", validate_schema=False)
    assert res.index_html.exists()
    assert res.summary_csv.exists()


def test_validate_schema_custom_strategy_mode_requires_plugins() -> None:
    cfg = _valid_cfg()
    cfg["backtest"] = {
        "enabled": True,
        "strategy": {
            "mode": "custom_strategy_name",
        },
    }
    with pytest.raises(ValueError, match="backtest.strategy.mode"):
        validate_run_config_schema(cfg, strict=True)


def test_validate_schema_stooq_requires_symbols() -> None:
    cfg = _valid_cfg()
    cfg["data"]["adapter"] = "stooq"
    cfg["data"].pop("synthetic", None)
    cfg["data"].pop("path", None)
    cfg["data"].pop("data_dir", None)
    cfg["data"]["symbols"] = []
    with pytest.raises(ValueError, match="data.symbols"):
        validate_run_config_schema(cfg, strict=True)


def test_validate_schema_custom_adapter_allowed_with_plugins() -> None:
    cfg = _valid_cfg()
    cfg["data"]["adapter"] = "my_adapter"
    cfg["data"]["adapter_plugin_dirs"] = ["plugins/data_adapters"]
    warnings = validate_run_config_schema(cfg, strict=True)
    assert isinstance(warnings, list)


def test_validate_schema_stooq_rejects_bad_timeout() -> None:
    cfg = _valid_cfg()
    cfg["data"]["adapter"] = "stooq"
    cfg["data"]["symbols"] = ["aapl"]
    cfg["data"]["request_timeout_sec"] = 0
    with pytest.raises(ValueError, match="request_timeout_sec"):
        validate_run_config_schema(cfg, strict=True)


def test_validate_schema_synthetic_rejects_small_n_days() -> None:
    cfg = _valid_cfg()
    cfg["data"]["adapter"] = "synthetic"
    cfg["data"]["synthetic"]["n_days"] = 10
    with pytest.raises(ValueError, match="data.synthetic.n_days"):
        validate_run_config_schema(cfg, strict=True)


def test_validate_schema_rejects_bad_transform_plugin_on_error() -> None:
    cfg = _valid_cfg()
    cfg["research"]["transform_plugin_on_error"] = "ignore"
    with pytest.raises(ValueError, match="research.transform_plugin_on_error"):
        validate_run_config_schema(cfg, strict=True)


def test_validate_schema_rejects_bad_custom_transform_format() -> None:
    cfg = _valid_cfg()
    cfg["research"]["custom_transforms"] = [{"name": "x", "kwargs": "not_a_dict"}]
    with pytest.raises(ValueError, match="research.custom_transforms\\[x\\].kwargs"):
        validate_run_config_schema(cfg, strict=True)


def test_run_from_config_strict_mode_rejects_autocorrect_when_schema_skipped(tmp_path: Path) -> None:
    cfg = _valid_cfg()
    cfg["run"]["config_mode"] = "strict"
    cfg["run"]["standardization"] = "bad_standardization"
    with pytest.raises(ValueError, match="auto-correction"):
        run_from_config(cfg, out_dir=tmp_path / "strict_autocorrect_fail", validate_schema=False)


def test_run_from_config_warn_mode_records_autocorrect(tmp_path: Path) -> None:
    cfg = _valid_cfg()
    cfg["run"]["config_mode"] = "warn"
    cfg["run"]["standardization"] = "bad_standardization"
    res = run_from_config(cfg, out_dir=tmp_path / "warn_autocorrect_ok", validate_schema=False)
    assert res.index_html.exists()
    meta = json.loads(res.run_meta_json.read_text(encoding="utf-8"))
    assert meta["config_governance"]["config_mode"] == "warn"
    assert meta["config_governance"]["autocorrection_count"] >= 1


def test_run_from_config_leakage_guard_blocks_forbidden_expression(tmp_path: Path) -> None:
    cfg = _valid_cfg()
    cfg["run"]["leakage_guard_mode"] = "strict"
    cfg["factor"] = {
        "names": ["leaky_expr"],
        "on_missing": "raise",
        "expressions": {"leaky_expr": "fwd_ret_5"},
        "expression_on_error": "raise",
    }
    with pytest.raises(ValueError, match="Leakage guard blocked run"):
        run_from_config(cfg, out_dir=tmp_path / "leakage_blocked", validate_schema=False)
