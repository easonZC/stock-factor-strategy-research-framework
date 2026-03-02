"""Synthetic end-to-end factor research entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "core"
if str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))

from factorlab.config import ResearchConfig, SyntheticConfig
from factorlab.data.synthetic import generate_synthetic_panel
from factorlab.factors import apply_factors
from factorlab.research import FactorResearchPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic factor-research report.")
    parser.add_argument(
        "--out",
        default="outputs/research/factor/synthetic_report",
        help="Output report directory",
    )
    parser.add_argument("--assets", type=int, default=30, help="Synthetic asset count")
    parser.add_argument("--days", type=int, default=220, help="Synthetic day count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = generate_synthetic_panel(SyntheticConfig(n_assets=args.assets, n_days=args.days, seed=11))
    factor_names = ["momentum_20", "volatility_20", "liquidity_shock"]
    panel = apply_factors(panel, factor_names)

    cfg = ResearchConfig(horizons=[1, 5, 10, 20], quantiles=5)
    pipeline = FactorResearchPipeline(cfg)
    pipeline.run(panel=panel, factors=factor_names, out_dir=args.out)


if __name__ == "__main__":
    main()
