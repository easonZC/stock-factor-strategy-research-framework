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
from factorlab.data import prepare_sina_panel, write_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare panel data via adapter",
        epilog=(
            "Examples:\n"
            "  python apps/prepare_data.py --adapter sina --data-dir /stock_sina_update --out data/panel.parquet\n"
            "  python apps/prepare_data.py --adapter sina --data-dir /stock_sina_update --out data/panel.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--adapter", required=True, choices=["sina"], help="Adapter type")
    parser.add_argument("--data-dir", required=True, help="Input data folder")
    parser.add_argument("--out", required=True, help="Output panel path (.parquet/.csv)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.adapter == "sina":
        panel = prepare_sina_panel(AdapterConfig(data_dir=args.data_dir))
    else:  # pragma: no cover
        raise ValueError(f"Unsupported adapter: {args.adapter}")
    write_panel(panel, args.out)


if __name__ == "__main__":
    main()
