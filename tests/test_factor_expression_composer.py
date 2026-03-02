"""相关功能测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from factorlab.factors.expression import evaluate_factor_expression
from factorlab.workflows import run_from_config


def test_evaluate_factor_expression_basic() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        }
    )
    out = evaluate_factor_expression(df, "a - b / 2")
    expected = df["a"] - df["b"] / 2
    assert out.equals(expected.astype(float))


def test_run_from_config_with_expression_factor(tmp_path: Path) -> None:
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
            "synthetic": {"n_assets": 12, "n_days": 180, "seed": 9, "start_date": "2020-01-01"},
        },
        "factor": {
            "names": ["mom_minus_vol"],
            "on_missing": "raise",
            "expressions": {
                "mom_minus_vol": "momentum_20 - volatility_20",
            },
            "expression_on_error": "raise",
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
    out_dir = tmp_path / "out_expression"
    result = run_from_config(cfg, out_dir=out_dir)
    assert result.index_html.exists()
    assert result.summary_csv.exists()

    meta = json.loads(result.run_meta_json.read_text(encoding="utf-8"))
    assert meta["factors"]["effective"] == ["mom_minus_vol"]
    assert "mom_minus_vol" in meta["factors"]["computed_expression_factors"]
