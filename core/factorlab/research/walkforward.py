"""模块说明。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from factorlab.backtest.engine import BacktestResult, run_backtest
from factorlab.config import BacktestConfig
from factorlab.models.registry import ModelRegistry
from factorlab.research.forward_returns import add_forward_returns
from factorlab.strategies.base import Strategy
from factorlab.utils import DateFrameIndexer, safe_corr


@dataclass(slots=True)
class WalkForwardConfig:
    """滚动 walk-forward 评估配置。"""

    feature_cols: list[str]
    label_horizon: int = 5
    model_name: str = "ridge"
    train_days: int = 252
    test_days: int = 21
    step_days: int = 21
    embargo_days: int | None = None
    min_train_rows: int = 500


@dataclass(slots=True)
class WalkForwardResult:
    """中文说明。"""

    oos_scores: pd.DataFrame
    fold_summary: pd.DataFrame
    weights: pd.DataFrame
    backtest: BacktestResult


def _average_daily_ic(df: pd.DataFrame, score_col: str, ret_col: str) -> float:
    rows: list[float] = []
    for _, grp in df.groupby("date"):
        g = grp[[score_col, ret_col]].dropna()
        if len(g) < 5:
            continue
        rows.append(float(safe_corr(g[score_col], g[ret_col], method="spearman", min_obs=5)))
    if not rows:
        return float("nan")
    return float(np.nanmean(rows))


def run_walkforward_strategy(
    panel: pd.DataFrame,
    strategy: Strategy,
    backtest_config: BacktestConfig,
    config: WalkForwardConfig,
) -> WalkForwardResult:
    """执行无前视泄露的 walk-forward 训练与回测。"""
    required = ["date", "asset", "close", *config.feature_cols]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise KeyError(f"panel missing required columns for walk-forward: {missing}")
    if config.train_days < 20:
        raise ValueError("WalkForwardConfig.train_days must be >= 20.")
    if config.test_days < 1:
        raise ValueError("WalkForwardConfig.test_days must be >= 1.")
    if config.step_days < 1:
        raise ValueError("WalkForwardConfig.step_days must be >= 1.")
    if config.label_horizon < 1:
        raise ValueError("WalkForwardConfig.label_horizon must be >= 1.")

    embargo_days = config.label_horizon if config.embargo_days is None else int(config.embargo_days)
    if embargo_days < 0:
        raise ValueError("WalkForwardConfig.embargo_days must be >= 0.")

    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["asset"] = df["asset"].astype(str)
    df = df.dropna(subset=["date", "asset"]).sort_values(["date", "asset"]).reset_index(drop=True)

    df = add_forward_returns(df, horizons=[config.label_horizon], price_col="close")
    label_col = f"fwd_ret_{config.label_horizon}"
    unique_dates = sorted(df["date"].dropna().unique())
    indexer = DateFrameIndexer(df=df, date_col="date")

    fold_rows: list[dict[str, float | int | str]] = []
    oos_score_parts: list[pd.DataFrame] = []
    fold_id = 0
    i = config.train_days + embargo_days

    while i < len(unique_dates):
        test_start = i
        test_end = min(i + config.test_days, len(unique_dates))
        train_end = test_start - embargo_days
        train_start = max(0, train_end - config.train_days)

        if train_end <= train_start:
            i += config.step_days
            continue

        train_dates = list(unique_dates[train_start:train_end])
        test_dates = list(unique_dates[test_start:test_end])

        train = indexer.select(train_dates)
        test = indexer.select(test_dates)

        train = train.dropna(subset=[*config.feature_cols, label_col])
        test = test.dropna(subset=config.feature_cols)

        if len(train) < config.min_train_rows or test.empty:
            i += config.step_days
            continue

        model = ModelRegistry.create(config.model_name)
        x_train = train[config.feature_cols].fillna(0.0)
        y_train = train[label_col].astype(float)
        model.fit(x_train, y_train)

        x_test = test[config.feature_cols].fillna(0.0)
        preds = model.predict(x_test)
        test_scored = test[["date", "asset", label_col]].copy()
        test_scored["score"] = preds.astype(float)
        test_scored["fold_id"] = fold_id

        fold_ic = _average_daily_ic(test_scored, score_col="score", ret_col=label_col)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "train_start": pd.Timestamp(min(train_dates)),
                "train_end": pd.Timestamp(max(train_dates)),
                "test_start": pd.Timestamp(min(test_dates)),
                "test_end": pd.Timestamp(max(test_dates)),
                "train_rows": int(len(train)),
                "test_rows": int(len(test_scored)),
                "oos_rank_ic": fold_ic,
            }
        )
        oos_score_parts.append(test_scored[["date", "asset", "score", "fold_id"]])

        fold_id += 1
        i += config.step_days

    if not oos_score_parts:
        raise RuntimeError("Walk-forward produced no OOS predictions. Check window sizes and panel length.")

    oos_scores = pd.concat(oos_score_parts, ignore_index=True)
    merged_scores = (
        oos_scores.groupby(["date", "asset"], as_index=False)["score"].mean().sort_values(["date", "asset"])
    )
    weights = strategy.generate_weights(merged_scores)
    bt = run_backtest(panel=df, weights=weights, config=backtest_config)

    fold_summary = pd.DataFrame(fold_rows).sort_values("fold_id").reset_index(drop=True)
    return WalkForwardResult(
        oos_scores=oos_scores.sort_values(["date", "asset", "fold_id"]).reset_index(drop=True),
        fold_summary=fold_summary,
        weights=weights.sort_values(["date", "asset"]).reset_index(drop=True),
        backtest=bt,
    )
