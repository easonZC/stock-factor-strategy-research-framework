"""模块说明。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from factorlab.preprocess import build_transform_registry
from factorlab.workflows import run_from_config


def _write_transform_plugin(plugin_dir: Path) -> Path:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    path = plugin_dir / "my_transforms.py"
    path.write_text(
        """
import pandas as pd

def transform_center_by_date(panel: pd.DataFrame, factor_col: str) -> pd.Series:
    s = panel[factor_col].astype(float)
    if "date" not in panel.columns:
        return s - s.mean()
    mu = panel.assign(_factor=s).groupby("date")["_factor"].transform("mean")
    return s - mu
""".strip(),
        encoding="utf-8",
    )
    return path


def _write_duplicate_transform_plugin(plugin_dir: Path) -> Path:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    path = plugin_dir / "dup_transform.py"
    path.write_text(
        """
import pandas as pd

def transform_clip(panel: pd.DataFrame, factor_col: str, lower=None, upper=None) -> pd.Series:
    return pd.Series(0.0, index=panel.index)
""".strip(),
        encoding="utf-8",
    )
    return path


def test_build_transform_registry_with_plugin_dir(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    _write_transform_plugin(plugin_dir)

    registry = build_transform_registry(plugin_dirs=[plugin_dir], include_defaults=False, on_plugin_error="raise")
    assert "center_by_date" in registry


def test_build_transform_registry_warn_skip_keeps_builtin_on_duplicate(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    _write_duplicate_transform_plugin(plugin_dir)

    registry = build_transform_registry(plugin_dirs=[plugin_dir], include_defaults=True, on_plugin_error="warn_skip")
    clip_fn = registry["clip"]
    panel = pd.DataFrame({"factor": [-10.0, 0.2, 9.0]})
    out = clip_fn(panel, "factor", lower=-1.0, upper=1.0)
    assert list(out.astype(float)) == [-1.0, 0.2, 1.0]


def test_run_from_config_with_custom_transform_plugin(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    _write_transform_plugin(plugin_dir)

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
            "synthetic": {"n_assets": 10, "n_days": 140, "seed": 11, "start_date": "2020-01-01"},
        },
        "factor": {"names": ["momentum_20"]},
        "research": {
            "horizons": [1, 5],
            "quantiles": 5,
            "ic_rolling_window": 20,
            "transform_auto_discover": True,
            "transform_plugin_dirs": [str(plugin_dir)],
            "transform_plugin_on_error": "raise",
            "custom_transforms": [
                {"name": "clip", "kwargs": {"lower": -5.0, "upper": 5.0}},
                {"name": "center_by_date"},
                {"name": "unknown_transform", "on_error": "warn_skip"},
            ],
            "winsorize": {"enabled": True, "method": "quantile"},
            "neutralize": {"enabled": False, "mode": "none"},
        },
        "backtest": {"enabled": False},
    }
    out_dir = tmp_path / "out_transform_plugin"
    res = run_from_config(cfg, out_dir=out_dir)

    assert res.index_html.exists()
    assert res.summary_csv.exists()

    meta = json.loads(res.run_meta_json.read_text(encoding="utf-8"))
    report = meta["research"]["custom_transform_report"]
    assert any(x["transform"] == "clip" for x in report["applied"])
    assert any(x["transform"] == "center_by_date" for x in report["applied"])
    assert any(x["transform"] == "unknown_transform" for x in report["skipped"])
    assert "center_by_date" in meta["research"]["transform_plugin_config"]["registry_transforms"]
