"""多持有期因子测试的未来收益标签生成。"""

from __future__ import annotations

import pandas as pd


def add_forward_returns(panel: pd.DataFrame, horizons: list[int], price_col: str = "close") -> pd.DataFrame:
    df = panel.sort_values(["asset", "date"]).copy()
    for h in horizons:
        df[f"fwd_ret_{h}"] = (
            df.groupby("asset")[price_col].shift(-h).astype("float32") / df[price_col].astype("float32") - 1.0
        ).astype("float32")
    return df
