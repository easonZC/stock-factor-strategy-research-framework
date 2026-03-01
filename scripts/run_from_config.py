"""Single entrypoint for config-driven TS/CS factor research runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ssf.utils import get_logger  # noqa: E402
from ssf.workflows import run_from_config  # noqa: E402

LOGGER = get_logger("ssf.run_from_config")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run factor pipeline from YAML config.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--out", required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_from_config(
        config=args.config,
        out_dir=args.out,
        repo_root=ROOT,
    )
    LOGGER.info(
        "Config run completed. report=%s summary=%s meta=%s",
        result.index_html,
        result.summary_csv,
        result.run_meta_json,
    )


if __name__ == "__main__":
    main()
