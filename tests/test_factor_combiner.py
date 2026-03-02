"""相关功能测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.factors import apply_factor_combinations, normalize_factor_combinations


def _toy_panel() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"])
    return pd.DataFrame(
        {
            "date": dates,
            "asset": ["A", "B", "A", "B"],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [2.0, 0.0, 1.0, -1.0],
            "size": [10.0, 20.0, 11.0, 21.0],
        }
    )


def test_normalize_factor_combinations_from_dict() -> None:
    cfg = {
        "combo_a": {
            "weights": {"f1": 1.0, "f2": -0.5},
            "standardization": "cs_zscore",
            "orthogonalize_to": ["size"],
        }
    }
    out = normalize_factor_combinations(cfg, strict=True)
    assert len(out) == 1
    assert out[0]["name"] == "combo_a"
    assert out[0]["weights"] == {"f1": 1.0, "f2": -0.5}
    assert out[0]["standardization"] == "cs_zscore"
    assert out[0]["orthogonalize_to"] == ["size"]


def test_apply_factor_combinations_weighted_sum() -> None:
    panel = _toy_panel()
    combos = normalize_factor_combinations(
        [
            {
                "name": "combo",
                "weights": {"f1": 1.0, "f2": -0.5},
                "standardization": "none",
            }
        ],
        strict=True,
    )
    out, computed, skipped, errors = apply_factor_combinations(panel, combos, on_error="raise")
    assert computed == ["combo"]
    assert skipped == []
    assert errors == []
    expected = panel["f1"] - 0.5 * panel["f2"]
    assert np.allclose(out["combo"].to_numpy(dtype=float), expected.to_numpy(dtype=float), equal_nan=True)


def test_apply_factor_combinations_warn_skip_missing_column() -> None:
    panel = _toy_panel()
    combos = normalize_factor_combinations(
        [{"name": "bad_combo", "weights": {"f1": 1.0, "missing_col": 1.0}}],
        strict=True,
    )
    out, computed, skipped, errors = apply_factor_combinations(panel, combos, on_error="warn_skip")
    assert "bad_combo" not in out.columns
    assert computed == []
    assert skipped == ["bad_combo"]
    assert len(errors) == 1

