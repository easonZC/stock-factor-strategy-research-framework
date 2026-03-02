"""Single entrypoint for config-driven TS/CS factor research runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "core"
if str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))

import yaml

from factorlab.utils import get_logger  # noqa: E402
from factorlab.workflows import compose_run_config, run_from_config  # noqa: E402

LOGGER = get_logger("factorlab.run_from_config")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run factor pipeline from YAML config(s).",
        epilog=(
            "Examples:\n"
            "  python apps/run_from_config.py --config configs/cs_factor_demo.yaml --out outputs/research/factor/cs\n"
            "  python apps/run_from_config.py --config base.yaml --config local.yaml --set research.horizons='[1,5,10]' --out outputs/research/factor/merged\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="YAML config path (repeatable). Later files override earlier files.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override key via dotted path, e.g. research.quantiles=10 (repeatable).",
    )
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--save-effective-config",
        action="store_true",
        help="Save merged+overridden effective config to <out>/effective_config.yaml.",
    )
    parser.add_argument(
        "--skip-schema-validation",
        action="store_true",
        help="Skip strict pre-validation of config schema before runtime.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    effective_cfg = compose_run_config(config_paths=args.config, overrides=args.overrides)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_effective_config:
        (out_dir / "effective_config.yaml").write_text(
            yaml.safe_dump(effective_cfg, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )

    result = run_from_config(
        config=effective_cfg,
        out_dir=args.out,
        repo_root=ROOT,
        validate_schema=not args.skip_schema_validation,
    )
    LOGGER.info(
        "Config run completed. report=%s summary=%s meta=%s",
        result.index_html,
        result.summary_csv,
        result.run_meta_json,
    )


if __name__ == "__main__":
    main()
