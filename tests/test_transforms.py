"""Preprocessing transform tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorlab.config import NeutralizationConfig
from factorlab.preprocess.transforms import apply_cs_standardize, apply_winsorize, handle_missing, neutralize_factor


def test_winsorize_and_neutralize() -> None:
    n = 120
    dates = pd.to_datetime(["2024-01-01"] * 60 + ["2024-01-02"] * 60)
    assets = [f"A{i:03d}" for i in range(60)] * 2
    factor = np.r_[np.linspace(-5, 5, 60), np.linspace(-6, 6, 60)]
    mkt_cap = np.linspace(1e8, 1e10, n)
    industry = ["Tech" if i % 2 == 0 else "Finance" for i in range(n)]

    df = pd.DataFrame(
        {"date": dates, "asset": assets, "factor": factor, "mkt_cap": mkt_cap, "industry": industry}
    )

    win = apply_winsorize(df.assign(factor=factor), factor_col="factor", method="quantile")
    neu = neutralize_factor(df.assign(factor=win), factor_col="factor", config=NeutralizationConfig(mode="both"))

    assert win.notna().mean() > 0.95
    assert neu.notna().mean() > 0.80


def test_handle_missing_policies() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "asset": ["A", "B", "A", "B", "A", "B"],
            "factor": [1.0, np.nan, np.nan, 4.0, 3.0, np.nan],
            "ret": [0.1, 0.2, np.nan, 0.3, 0.4, np.nan],
        }
    )
    cols = ["factor", "ret"]

    dropped = handle_missing(df, cols=cols, policy="drop")
    assert dropped[cols].isna().sum().sum() == 0

    zero = handle_missing(df, cols=cols, policy="fill_zero")
    assert zero[cols].isna().sum().sum() == 0
    assert (zero[cols] == 0.0).any().any()

    ffill = handle_missing(df, cols=cols, policy="ffill_by_asset")
    assert ffill[cols].isna().sum().sum() == 0

    cs_med = handle_missing(df, cols=cols, policy="cs_median_by_date")
    assert cs_med[cols].isna().sum().sum() == 0

    kept = handle_missing(df, cols=cols, policy="keep")
    assert kept[cols].isna().sum().sum() > 0


def test_cs_robust_zscore_standardization() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 5 + ["2024-01-02"] * 5),
            "factor": [1.0, 1.0, 2.0, 2.0, 100.0, 2.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    out = apply_cs_standardize(df, col="factor", method="cs_robust_zscore")
    assert out.notna().mean() > 0.7
    assert float(out.abs().max()) < 100.0
