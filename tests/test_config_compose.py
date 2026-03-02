"""Tests for config composition and dotted overrides."""

from __future__ import annotations

from pathlib import Path

import yaml

from factorlab.workflows import apply_config_override, compose_run_config, deep_merge_dict


def test_deep_merge_dict_recursive() -> None:
    base = {"run": {"factor_scope": "cs", "standardization": "cs_zscore"}, "research": {"quantiles": 5}}
    overlay = {"run": {"standardization": "cs_rank"}, "research": {"horizons": [1, 5, 10]}}
    out = deep_merge_dict(base, overlay)
    assert out["run"]["factor_scope"] == "cs"
    assert out["run"]["standardization"] == "cs_rank"
    assert out["research"]["quantiles"] == 5
    assert out["research"]["horizons"] == [1, 5, 10]


def test_apply_config_override_parses_value_types() -> None:
    cfg = {"research": {"quantiles": 5}, "backtest": {"enabled": False}}
    out = apply_config_override(cfg, "research.horizons=[1,5,10]")
    out = apply_config_override(out, "backtest.enabled=true")
    out = apply_config_override(out, "run.factor_scope=cs")
    assert out["research"]["horizons"] == [1, 5, 10]
    assert out["backtest"]["enabled"] is True
    assert out["run"]["factor_scope"] == "cs"


def test_compose_run_config_merge_and_override(tmp_path: Path) -> None:
    base = {
        "run": {"factor_scope": "cs", "eval_axis": "cross_section", "standardization": "cs_zscore"},
        "research": {"quantiles": 5},
    }
    local = {
        "run": {"standardization": "cs_rank"},
        "research": {"horizons": [1, 5]},
    }
    base_path = tmp_path / "base.yaml"
    local_path = tmp_path / "local.yaml"
    base_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    local_path.write_text(yaml.safe_dump(local, sort_keys=False), encoding="utf-8")

    out = compose_run_config(
        [base_path, local_path],
        overrides=["research.quantiles=10", "research.horizons=[1,5,10,20]"],
    )
    assert out["run"]["standardization"] == "cs_rank"
    assert out["research"]["quantiles"] == 10
    assert out["research"]["horizons"] == [1, 5, 10, 20]

