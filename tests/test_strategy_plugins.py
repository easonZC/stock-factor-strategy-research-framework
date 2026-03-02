"""Tests for strategy plugin discovery and config integration."""

from __future__ import annotations

import json
from pathlib import Path

from factorlab.strategies import build_strategy_registry
from factorlab.workflows import run_from_config


def _write_strategy_plugin(plugin_dir: Path) -> Path:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    path = plugin_dir / "my_strategy_plugin.py"
    path.write_text(
        """
from dataclasses import dataclass
import pandas as pd
from factorlab.strategies.base import Strategy

@dataclass(slots=True)
class ScoreSignTiltStrategy(Strategy):
    threshold: float = 0.0

    def generate_weights(self, score_df: pd.DataFrame) -> pd.DataFrame:
        out = score_df[["date", "asset", "score"]].copy()
        out["score"] = pd.to_numeric(out["score"], errors="coerce")
        out["weight"] = out["score"].apply(lambda x: 1.0 if x > self.threshold else (-1.0 if x < -self.threshold else 0.0))
        return out[["date", "asset", "weight"]]
""".strip(),
        encoding="utf-8",
    )
    return path


def _write_duplicate_strategy_plugin(plugin_dir: Path) -> Path:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    path = plugin_dir / "dup_strategy.py"
    path.write_text(
        """
from factorlab.strategies.implementations import TopKLongStrategy

STRATEGY_REGISTRY = {
    "topk": lambda: TopKLongStrategy(name="topk_long", top_k=3)
}
""".strip(),
        encoding="utf-8",
    )
    return path


def test_build_strategy_registry_with_plugin_dir(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    _write_strategy_plugin(plugin_dir)
    reg = build_strategy_registry(plugin_dirs=[plugin_dir], on_plugin_error="raise", include_defaults=False)
    assert "score_sign_tilt" in reg


def test_build_strategy_registry_warn_skip_keeps_builtin_on_duplicate(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    _write_duplicate_strategy_plugin(plugin_dir)

    reg = build_strategy_registry(plugin_dirs=[plugin_dir], on_plugin_error="warn_skip", include_defaults=True)
    topk = reg["topk"]()
    assert int(getattr(topk, "top_k")) == 20


def test_run_from_config_with_plugin_strategy(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    _write_strategy_plugin(plugin_dir)

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
            "synthetic": {"n_assets": 12, "n_days": 150, "seed": 8, "start_date": "2020-01-01"},
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
        "backtest": {
            "enabled": True,
            "strategy": {
                "mode": "score_sign_tilt",
                "auto_discover": True,
                "plugin_dirs": [str(plugin_dir)],
                "plugin_on_error": "raise",
            },
            "commission_bps": 2.0,
            "slippage_bps": 1.0,
            "leverage": 1.0,
        },
    }
    out_dir = tmp_path / "out_strategy_plugin"
    res = run_from_config(cfg, out_dir=out_dir)

    assert res.index_html.exists()
    assert res.summary_csv.exists()
    assert res.backtest_summary_csv is not None and res.backtest_summary_csv.exists()

    meta = json.loads(res.run_meta_json.read_text(encoding="utf-8"))
    assert meta["backtest"]["config"]["strategy_mode"] == "score_sign_tilt"
