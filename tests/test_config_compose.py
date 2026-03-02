"""Tests for config composition and dotted overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
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


def test_apply_config_override_supports_list_append_and_remove() -> None:
    cfg = {"research": {"horizons": [1, 5], "tags": ["base"]}}
    out = apply_config_override(cfg, "research.horizons+=10")
    out = apply_config_override(out, "research.horizons+=[20,30]")
    out = apply_config_override(out, "research.tags+=fast")
    out = apply_config_override(out, "research.horizons-=5")
    out = apply_config_override(out, "research.horizons-=[20,999]")
    assert out["research"]["horizons"] == [1, 10, 30]
    assert out["research"]["tags"] == ["base", "fast"]


def test_apply_config_override_supports_dict_merge_and_remove() -> None:
    cfg = {"research": {"winsorize": {"enabled": True, "method": "quantile"}}}
    out = apply_config_override(cfg, "research.winsorize+={lower_q: 0.02, upper_q: 0.98}")
    out = apply_config_override(out, "research.winsorize-=method")
    assert out["research"]["winsorize"]["enabled"] is True
    assert out["research"]["winsorize"]["lower_q"] == 0.02
    assert out["research"]["winsorize"]["upper_q"] == 0.98
    assert "method" not in out["research"]["winsorize"]


def test_apply_config_override_append_to_missing_path_creates_list() -> None:
    cfg = {"research": {}}
    out = apply_config_override(cfg, "research.extra_factors+=momentum_60")
    assert out["research"]["extra_factors"] == ["momentum_60"]


def test_apply_config_override_remove_missing_path_raises() -> None:
    cfg = {"research": {}}
    with pytest.raises(ValueError, match="target path does not exist"):
        apply_config_override(cfg, "research.horizons-=5")


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
