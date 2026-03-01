"""Preprocessing transform tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ssf.config import NeutralizationConfig
from ssf.preprocess.transforms import apply_winsorize, neutralize_factor


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
