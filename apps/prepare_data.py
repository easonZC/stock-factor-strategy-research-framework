"""通过适配器准备标准化面板数据。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "core"
if str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))

from factorlab.config import AdapterConfig
from factorlab.data import build_data_adapter_registry, build_data_adapter_validator_registry, write_panel
from factorlab.utils import configure_logging, get_logger

LOGGER = get_logger("factorlab.prepare_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare panel data via adapter",
        epilog=(
            "Examples:\n"
            "  python apps/prepare_data.py --adapter sina --data-dir /stock_sina_update --out data/panel.parquet\n"
            "  python apps/prepare_data.py --adapter stooq --symbols aapl,msft,googl --start-date 2022-01-01 --out data/panel.parquet\n"
            "  python apps/prepare_data.py --adapter my_feed --adapter-plugin-dir plugins/data_adapters --out data/panel.parquet\n"
            "  python apps/prepare_data.py --adapter sina --data-dir /stock_sina_update --out data/panel.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--adapter", required=True, help="Adapter type (built-in or plugin adapter name)")
    parser.add_argument("--data-dir", default=None, help="Input data folder (required for sina)")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols (required for stooq)")
    parser.add_argument("--start-date", default=None, help="Optional start date for stooq (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Optional end date for stooq (YYYY-MM-DD)")
    parser.add_argument("--min-rows-per-asset", type=int, default=30, help="Minimum clean rows per asset")
    parser.add_argument("--request-timeout-sec", type=int, default=20, help="HTTP timeout in seconds (stooq)")
    parser.add_argument(
        "--adapter-plugin-dir",
        dest="adapter_plugin_dirs",
        action="append",
        default=[],
        help="Data adapter plugin directory (repeatable).",
    )
    parser.add_argument(
        "--adapter-plugin",
        dest="adapter_plugins",
        action="append",
        default=[],
        help="Data adapter plugin module/spec (repeatable).",
    )
    parser.add_argument(
        "--adapter-plugin-on-error",
        choices=["raise", "warn_skip"],
        default="raise",
        help="Plugin load conflict/error behavior.",
    )
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

    registry = build_data_adapter_registry(
        plugin_dirs=args.adapter_plugin_dirs,
        plugin_specs=args.adapter_plugins,
        on_plugin_error=args.adapter_plugin_on_error,
        include_defaults=True,
    )
    validator_registry = build_data_adapter_validator_registry(
        plugin_dirs=args.adapter_plugin_dirs,
        plugin_specs=args.adapter_plugins,
        on_plugin_error=args.adapter_plugin_on_error,
        include_defaults=True,
    )
    adapter_name = str(args.adapter).strip().lower()
    if adapter_name not in registry:
        raise KeyError(f"Unknown adapter '{adapter_name}'. Available adapters: {sorted(registry.keys())}")

    symbols = tuple(x.strip() for x in str(args.symbols).split(",") if x.strip()) if args.symbols else ()
    if adapter_name == "sina" and not args.data_dir:
        raise ValueError("--data-dir is required when --adapter=sina")
    if adapter_name == "stooq" and not symbols:
        raise ValueError("--symbols is required when --adapter=stooq")

    adapter_cfg = AdapterConfig(
        data_dir=str(args.data_dir or ""),
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        min_rows_per_asset=int(args.min_rows_per_asset),
        request_timeout_sec=int(args.request_timeout_sec),
    )
    validator = validator_registry.get(adapter_name)
    if validator is not None:
        validator(adapter_cfg)
    panel = registry[adapter_name](adapter_cfg)
    out = write_panel(panel, args.out)
    LOGGER.info("Prepared panel saved to %s (rows=%s assets=%s)", out, len(panel), panel["asset"].nunique())


if __name__ == "__main__":
    main()
