"""Prepare canonical panel data from external sources (adapter-driven)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ssf.config import AdapterConfig
from ssf.data import prepare_sina_panel, write_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare panel data via adapter")
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

