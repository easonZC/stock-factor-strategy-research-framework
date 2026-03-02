"""Run factor research from an input panel file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "core"
if str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))

from factorlab.config import NeutralizationConfig, ResearchConfig
from factorlab.data import read_panel
from factorlab.factors import apply_factors, default_factor_registry
from factorlab.research import FactorResearchPipeline
from factorlab.utils import get_logger

LOGGER = get_logger("factorlab.run_factor_research")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run factor research on a panel file.")
    parser.add_argument("--panel", required=True, help="Panel path (.parquet/.csv)")
    parser.add_argument(
        "--factors",
        default="momentum_20,volatility_20,liquidity_shock",
        help="Comma-separated factor names",
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 20], help="Forward-return horizons")
    parser.add_argument(
        "--out",
        default="outputs/research/factor/panel_report",
        help="Output directory",
    )
    parser.add_argument("--neutralize", choices=["none", "size", "industry", "both"], default="both")
    parser.add_argument("--winsorize", choices=["quantile", "mad"], default="quantile")
    parser.add_argument(
        "--standardization",
        choices=["cs_zscore", "cs_rank", "cs_robust_zscore", "none"],
        default="cs_zscore",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["drop", "fill_zero", "ffill_by_asset", "cs_median_by_date", "keep"],
        default="drop",
    )
    parser.add_argument(
        "--preprocess-steps",
        default="winsorize,standardize,neutralize",
        help="Comma-separated preprocess steps for CS pipeline",
    )
    parser.add_argument(
        "--on-missing-factor",
        choices=["raise", "warn_skip"],
        default="raise",
        help="Behavior when factor is neither in panel nor built-in registry",
    )
    parser.add_argument("--max-assets", type=int, default=None, help="Optional cap on number of assets")
    parser.add_argument("--start-date", default=None, help="Optional start date filter (YYYY-MM-DD)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = read_panel(args.panel)

    if args.start_date:
        panel = panel[panel["date"] >= pd.to_datetime(args.start_date)].copy()
    if args.max_assets is None:
        n_assets = panel["asset"].nunique()
        if n_assets > 1000:
            args.max_assets = 120
            LOGGER.warning(
                "Large panel detected (%s assets). Applying safety cap --max-assets=%s. "
                "Set --max-assets explicitly to override.",
                n_assets,
                args.max_assets,
            )
    if args.max_assets:
        keep_assets = panel["asset"].astype(str).drop_duplicates().head(args.max_assets)
        panel = panel[panel["asset"].astype(str).isin(set(keep_assets))].copy()

    factor_names = [x.strip() for x in args.factors.split(",") if x.strip()]
    registry = default_factor_registry()
    missing = [f for f in factor_names if f not in panel.columns]
    computable = [f for f in missing if f in registry]
    if computable:
        panel = apply_factors(panel, computable, inplace=True)

    unresolved = [f for f in factor_names if f not in panel.columns]
    if unresolved:
        if args.on_missing_factor == "warn_skip":
            LOGGER.warning("Skip unresolved factors due to --on-missing-factor=warn_skip: %s", unresolved)
            factor_names = [f for f in factor_names if f not in unresolved]
        else:
            raise KeyError(f"Factors not found and not computable: {unresolved}")
    if not factor_names:
        raise RuntimeError("No valid factors to run after resolving missing factors.")

    preprocess_steps = [x.strip().lower() for x in str(args.preprocess_steps).split(",") if x.strip()]

    cfg = ResearchConfig(
        horizons=args.horizons,
        quantiles=5,
        standardization=args.standardization,
        winsorize_method=args.winsorize,
        missing_policy=args.missing_policy,
        preprocess_steps=preprocess_steps,
        neutralization=NeutralizationConfig(mode=args.neutralize),
    )
    pipeline = FactorResearchPipeline(cfg)
    pipeline.run(panel=panel, factors=factor_names, out_dir=args.out)


if __name__ == "__main__":
    main()
