"""相关功能测试。"""

from __future__ import annotations

import pandas as pd

from factorlab.backtest import run_backtest
from factorlab.backtest.engine import _apply_weight_constraints
from factorlab.config import BacktestConfig, CostConfig, SyntheticConfig
from factorlab.data.synthetic import generate_synthetic_panel


def _build_noisy_weights(panel: pd.DataFrame, n_assets: int = 8) -> pd.DataFrame:
    dates = sorted(panel["date"].dropna().unique())
    assets = sorted(panel["asset"].astype(str).drop_duplicates().tolist())[:n_assets]
    rows: list[dict[str, object]] = []
    for i, dt in enumerate(dates):
        flip = -1.0 if i % 2 else 1.0
        for j, asset in enumerate(assets):
            base = 2.5 if j == 0 else (1.2 if j % 2 == 0 else -1.1)
            rows.append({"date": dt, "asset": asset, "weight": float(base * flip)})
    return pd.DataFrame(rows)


def test_backtest_constraints_are_enforced() -> None:
    panel = generate_synthetic_panel(SyntheticConfig(n_assets=10, n_days=120, seed=11, start_date="2020-01-01"))
    panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)
    weights = _build_noisy_weights(panel, n_assets=8)

    cfg = BacktestConfig(
        cost=CostConfig(commission_bps=1.0, slippage_bps=1.0),
        long_short_leverage=2.0,
        max_turnover=0.60,
        max_abs_weight=0.25,
        max_gross_exposure=1.00,
        max_net_exposure=0.20,
        benchmark_mode="cross_sectional_mean",
    )

    w_target = weights.pivot(index="date", columns="asset", values="weight").sort_index().fillna(0.0)
    constrained = _apply_weight_constraints(w_target=w_target, panel=panel, config=cfg)
    assert float(constrained.abs().max().max()) <= 0.25 + 1e-9

    res = run_backtest(panel=panel, weights=weights, config=cfg)
    daily = res.daily
    assert float(daily["turnover"].max()) <= 0.60 + 1e-9
    assert float(daily["gross_exposure"].max()) <= 1.00 + 1e-9
    assert float(daily["net_exposure"].abs().max()) <= 0.20 + 1e-9
    assert daily["benchmark_ret"].notna().all()

