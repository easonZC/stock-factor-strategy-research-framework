"""Tests for strategy-level risk constraints."""

from __future__ import annotations

import pandas as pd
import pytest

from factorlab.strategies.implementations import _normalize_long


def test_normalize_long_enforces_max_weight_after_normalization() -> None:
    raw = pd.Series([0.9, 0.1], index=["A", "B"], dtype=float)
    out = _normalize_long(raw, max_weight=0.6)
    assert out.sum() == pytest.approx(1.0)
    assert float(out.max()) <= 0.6 + 1e-10


def test_normalize_long_raises_on_infeasible_max_weight() -> None:
    raw = pd.Series([0.9, 0.1], index=["A", "B"], dtype=float)
    with pytest.raises(ValueError):
        _normalize_long(raw, max_weight=0.4)
