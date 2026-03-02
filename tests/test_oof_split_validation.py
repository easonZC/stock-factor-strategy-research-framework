"""Tests for OOF split validation and guardrails."""

from __future__ import annotations

import pandas as pd
import pytest

from factorlab.models.trainer import OOFSplitConfig, _iter_folds


def test_oof_split_config_rejects_non_positive_step_days() -> None:
    with pytest.raises(ValueError):
        OOFSplitConfig(step_days=0)


def test_iter_folds_rejects_non_positive_step_days_even_if_mutated() -> None:
    cfg = OOFSplitConfig(step_days=5)
    cfg.step_days = 0
    dates = list(pd.date_range("2024-01-01", periods=40, freq="B"))
    with pytest.raises(ValueError):
        next(_iter_folds(dates, cfg=cfg))
