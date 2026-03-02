"""交易日索引与日期定位工具。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(slots=True)
class DateFrameIndexer:
    """基于日期索引缓存的 DataFrame 切片器。

    设计目标：
    - 避免在 walkforward/OOF 循环中反复使用 `isin` 触发整表扫描。
    - 通过日期到行号数组的映射，快速提取训练/验证窗口子集。
    """

    df: pd.DataFrame
    date_col: str = "date"
    _index_by_key: dict[int, np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.date_col not in self.df.columns:
            raise KeyError(f"DateFrameIndexer: missing date column '{self.date_col}'.")
        date_series = pd.to_datetime(self.df[self.date_col], errors="coerce")
        self._index_by_key: dict[int, np.ndarray] = {}
        groups = date_series.groupby(date_series).indices
        for raw_date, idx in groups.items():
            ts = pd.Timestamp(raw_date)
            if pd.isna(ts):
                continue
            self._index_by_key[int(ts.value)] = np.asarray(idx, dtype=np.int64)

    @staticmethod
    def _to_key(raw_date: object) -> int | None:
        ts = pd.Timestamp(raw_date)
        if pd.isna(ts):
            return None
        return int(ts.value)

    def select(self, dates: Iterable[object]) -> pd.DataFrame:
        """按给定日期集合返回子表（保持输入日期顺序）。"""
        date_list = list(dates)
        if not date_list:
            return self.df.iloc[0:0].copy()

        idx_parts: list[np.ndarray] = []
        for raw_date in date_list:
            key = self._to_key(raw_date)
            if key is None:
                continue
            arr = self._index_by_key.get(key)
            if arr is None:
                continue
            idx_parts.append(arr)
        if not idx_parts:
            return self.df.iloc[0:0].copy()

        idx = np.concatenate(idx_parts)
        return self.df.iloc[idx].copy()
