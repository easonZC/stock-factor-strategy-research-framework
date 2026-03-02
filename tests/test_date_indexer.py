"""相关功能测试。"""

from __future__ import annotations

import pandas as pd

from factorlab.utils import DateFrameIndexer


def test_date_indexer_select_preserves_date_order() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-04",
                ]
            ),
            "asset": ["A", "B", "A", "B", "A"],
            "value": [1, 2, 3, 4, 5],
        }
    )
    indexer = DateFrameIndexer(df=df, date_col="date")
    out = indexer.select([pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-02")])
    assert list(out["date"].dt.strftime("%Y-%m-%d")) == [
        "2024-01-03",
        "2024-01-03",
        "2024-01-02",
        "2024-01-02",
    ]
    assert list(out["value"]) == [3, 4, 1, 2]


def test_date_indexer_select_missing_dates_returns_empty() -> None:
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "asset": ["A"], "value": [1.0]})
    indexer = DateFrameIndexer(df=df, date_col="date")
    out = indexer.select([pd.Timestamp("2025-01-01")])
    assert out.empty
