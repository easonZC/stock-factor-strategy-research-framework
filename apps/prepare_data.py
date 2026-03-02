"""Prepare canonical panel data from external sources (adapter-driven)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "core"
if str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))

from factorlab.config import AdapterConfig
from factorlab.data import prepare_sina_panel, prepare_stooq_panel, write_panel
from factorlab.utils import configure_logging, get_logger

LOGGER = get_logger("factorlab.prepare_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare panel data via adapter",
        epilog=(
            "Examples:\n"
            "  python apps/prepare_data.py --adapter sina --data-dir /stock_sina_update --out data/panel.parquet\n"
            "  python apps/prepare_data.py --adapter stooq --symbols aapl,msft,googl --start-date 2022-01-01 --out data/panel.parquet\n"
            "  python apps/prepare_data.py --adapter sina --data-dir /stock_sina_update --out data/panel.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--adapter", required=True, choices=["sina", "stooq"], help="Adapter type")
    parser.add_argument("--data-dir", default=None, help="Input data folder (required for sina)")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols (required for stooq)")
    parser.add_argument("--start-date", default=None, help="Optional start date for stooq (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Optional end date for stooq (YYYY-MM-DD)")
    parser.add_argument("--min-rows-per-asset", type=int, default=30, help="Minimum clean rows per asset")
    parser.add_argument("--request-timeout-sec", type=int, default=20, help="HTTP timeout in seconds (stooq)")
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level (DEBUG/INFO/WARNING/ERROR). Also supports env FACTORLAB_LOG_LEVEL.",
    )
    parser.add_argument("--log-file", default=None, help="Optional log file path for this run.")
    parser.add_argument("--out", required=True, help="Output panel path (.parquet/.csv)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(level=args.log_level, log_file=args.log_file, force=True)
    if args.adapter == "sina":
        if not args.data_dir:
            raise ValueError("--data-dir is required when --adapter=sina")
        panel = prepare_sina_panel(
            AdapterConfig(
                data_dir=args.data_dir,
                min_rows_per_asset=int(args.min_rows_per_asset),
            )
        )
    elif args.adapter == "stooq":
        if not args.symbols:
            raise ValueError("--symbols is required when --adapter=stooq")
        symbols = tuple(x.strip() for x in str(args.symbols).split(",") if x.strip())
        panel = prepare_stooq_panel(
            AdapterConfig(
                symbols=symbols,
                start_date=args.start_date,
                end_date=args.end_date,
                min_rows_per_asset=int(args.min_rows_per_asset),
                request_timeout_sec=int(args.request_timeout_sec),
            )
        )
    else:  # pragma: no cover
        raise ValueError(f"Unsupported adapter: {args.adapter}")
    out = write_panel(panel, args.out)
    LOGGER.info("Prepared panel saved to %s (rows=%s assets=%s)", out, len(panel), panel["asset"].nunique())


if __name__ == "__main__":
    main()
