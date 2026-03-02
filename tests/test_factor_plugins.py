"""相关功能测试。"""

from __future__ import annotations

import json
from pathlib import Path

from factorlab.config import SyntheticConfig
from factorlab.data import generate_synthetic_panel
from factorlab.factors import apply_factors, build_factor_registry
from factorlab.workflows import run_from_config


def _write_plugin_file(plugin_dir: Path) -> Path:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    path = plugin_dir / "my_plugin.py"
    path.write_text(
        """
from dataclasses import dataclass
import pandas as pd
from factorlab.factors.base import Factor

@dataclass(slots=True)
class SimpleReversalFactor(Factor):
    lookback: int = 3

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        df = panel.sort_values(["asset", "date"])
        ret = df.groupby("asset")["close"].pct_change(self.lookback)
        return (-ret).reindex(panel.index)
""".strip(),
        encoding="utf-8",
    )
    return path


def _write_duplicate_factor_plugin(plugin_dir: Path) -> Path:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    path = plugin_dir / "dup_factor.py"
    path.write_text(
        """
from factorlab.factors.simple import MomentumFactor

FACTOR_REGISTRY = {
    "momentum_20": lambda: MomentumFactor(name="momentum_20", lookback=2)
}
""".strip(),
        encoding="utf-8",
    )
    return path


def test_build_factor_registry_with_plugin_dir(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    _write_plugin_file(plugin_dir)

    panel = generate_synthetic_panel(SyntheticConfig(n_assets=8, n_days=90, seed=12))
    registry = build_factor_registry(plugin_dirs=[plugin_dir], on_plugin_error="raise")
    assert "simple_reversal" in registry

    out = apply_factors(panel, ["simple_reversal"], inplace=False, registry=registry)
    assert "simple_reversal" in out.columns
    assert out["simple_reversal"].notna().mean() > 0.5


def test_build_factor_registry_warn_skip_keeps_builtin_on_duplicate(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    _write_duplicate_factor_plugin(plugin_dir)

    registry = build_factor_registry(plugin_dirs=[plugin_dir], on_plugin_error="warn_skip")
    momentum = registry["momentum_20"]()
    assert int(getattr(momentum, "lookback")) == 20


def test_run_from_config_with_plugin_auto_discovery(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    _write_plugin_file(plugin_dir)

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
            "synthetic": {"n_assets": 10, "n_days": 140, "seed": 22, "start_date": "2020-01-01"},
        },
        "factor": {
            "names": ["simple_reversal"],
            "on_missing": "raise",
            "auto_discover": True,
            "plugin_dirs": [str(plugin_dir)],
            "plugin_on_error": "raise",
        },
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
    out_dir = tmp_path / "out_plugin_run"
    result = run_from_config(cfg, out_dir=out_dir)

    assert result.index_html.exists()
    assert result.summary_csv.exists()
    meta = json.loads(result.run_meta_json.read_text(encoding="utf-8"))
    assert meta["factors"]["effective"] == ["simple_reversal"]
    assert meta["factors"]["plugin_config"]["auto_discover"] is True
