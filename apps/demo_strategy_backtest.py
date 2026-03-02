"""Run a small synthetic strategy backtest demo (TopK + LongShort)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "core"
if str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))

import pandas as pd

from factorlab.backtest import run_backtest
from factorlab.config import BacktestConfig, SyntheticConfig
from factorlab.data.synthetic import generate_synthetic_panel
from factorlab.factors import apply_factors
from factorlab.strategies import LongShortQuantileStrategy, TopKLongStrategy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/strategy_demo", help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panel = generate_synthetic_panel(SyntheticConfig(n_assets=30, n_days=180, seed=99))
    panel = apply_factors(panel, ["momentum_20"])

    score_df = panel[["date", "asset", "momentum_20"]].rename(columns={"momentum_20": "score"})

    topk = TopKLongStrategy(name="topk_long", top_k=10)
    ls = LongShortQuantileStrategy(name="long_short_q", quantile=0.2)

    bt_cfg = BacktestConfig()
    topk_res = run_backtest(panel, topk.generate_weights(score_df), bt_cfg)
    ls_res = run_backtest(panel, ls.generate_weights(score_df), bt_cfg)

    topk_res.daily.to_csv(out / "topk_daily.csv", index=False)
    topk_res.metrics.to_csv(out / "topk_metrics.csv", index=False)
    ls_res.daily.to_csv(out / "ls_daily.csv", index=False)
    ls_res.metrics.to_csv(out / "ls_metrics.csv", index=False)


if __name__ == "__main__":
    main()
