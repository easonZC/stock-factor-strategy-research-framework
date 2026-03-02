"""相关功能测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from factorlab.config import SyntheticConfig
from factorlab.data.synthetic import generate_synthetic_panel
from factorlab.models.trainer import OOFSplitConfig, _iter_folds, train_oof_model_factor


def test_oof_split_config_rejects_non_positive_step_days() -> None:
    with pytest.raises(ValueError):
        OOFSplitConfig(step_days=0)


def test_iter_folds_rejects_non_positive_step_days_even_if_mutated() -> None:
    cfg = OOFSplitConfig(step_days=5)
    cfg.step_days = 0
    dates = list(pd.date_range("2024-01-01", periods=40, freq="B"))
    with pytest.raises(ValueError):
        next(_iter_folds(dates, cfg=cfg))


def test_oof_split_config_rejects_invalid_split_mode() -> None:
    with pytest.raises(ValueError):
        OOFSplitConfig(split_mode="bad")  # type: ignore[arg-type]


def test_iter_folds_expanding_vs_rolling_window() -> None:
    dates = list(pd.date_range("2024-01-01", periods=80, freq="B"))
    rolling = OOFSplitConfig(
        train_days=20,
        valid_days=10,
        step_days=10,
        embargo_days=0,
        purge_days=2,
        split_mode="rolling",
    )
    expanding = OOFSplitConfig(
        train_days=20,
        valid_days=10,
        step_days=10,
        embargo_days=0,
        purge_days=2,
        split_mode="expanding",
    )
    roll_folds = list(_iter_folds(dates, cfg=rolling))
    exp_folds = list(_iter_folds(dates, cfg=expanding))
    assert len(roll_folds) == len(exp_folds) >= 2
    _, roll_train_2, _ = roll_folds[1]
    _, exp_train_2, _ = exp_folds[1]
    assert pd.Timestamp(min(roll_train_2)) > pd.Timestamp(min(exp_train_2))
    assert pd.Timestamp(min(exp_train_2)) == pd.Timestamp(dates[0])


def test_train_oof_model_factor_supports_mse_and_time_axis() -> None:
    panel = generate_synthetic_panel(SyntheticConfig(n_assets=10, n_days=240, seed=42, start_date="2020-01-01"))
    panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)
    panel["label_next_ret"] = panel.groupby("asset")["close"].pct_change().shift(-1)

    cfg = OOFSplitConfig(
        train_days=80,
        valid_days=20,
        step_days=20,
        embargo_days=3,
        purge_days=2,
        split_mode="expanding",
        min_train_rows=160,
        min_valid_rows=40,
    )
    res = train_oof_model_factor(
        panel=panel,
        feature_cols=["close", "volume", "mkt_cap"],
        model_name="ridge",
        split_config=cfg,
        param_grid=[{"alpha": 0.1}, {"alpha": 1.0}],
        label_col="label_next_ret",
        scoring_metric="mse",
        evaluation_axis="time",
    )
    assert not res.oof_predictions.empty
    assert "label" in res.oof_predictions.columns
    assert "label_next_ret" not in res.oof_predictions.columns
    assert {"mean_oof_score", "score_metric", "evaluation_axis"}.issubset(res.tuning_summary.columns)
    assert set(res.tuning_summary["score_metric"]) == {"mse"}
    assert set(res.tuning_summary["evaluation_axis"]) == {"time"}


def test_train_oof_model_factor_raises_when_dates_insufficient() -> None:
    panel = generate_synthetic_panel(SyntheticConfig(n_assets=6, n_days=70, seed=9, start_date="2021-01-01"))
    panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)
    cfg = OOFSplitConfig(
        train_days=60,
        valid_days=20,
        step_days=10,
        embargo_days=5,
        purge_days=0,
        split_mode="rolling",
        min_train_rows=120,
        min_valid_rows=40,
    )
    with pytest.raises(RuntimeError, match="Not enough unique dates"):
        train_oof_model_factor(
            panel=panel,
            feature_cols=["close", "volume", "mkt_cap"],
            model_name="ridge",
            label_horizon=5,
            split_config=cfg,
        )
