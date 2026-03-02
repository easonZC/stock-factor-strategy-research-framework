"""Generate run-config templates for TS/CS workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "core"
if str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))

from factorlab.workflows import apply_config_override  # noqa: E402


def _base_run(scope: str) -> dict[str, Any]:
    is_cs = scope == "cs"
    return {
        "factor_scope": scope,
        "eval_axis": "cross_section" if is_cs else "time",
        "standardization": "cs_zscore" if is_cs else "ts_rolling_zscore",
    }


def _base_data(scope: str, adapter: str) -> dict[str, Any]:
    is_cs = scope == "cs"
    out: dict[str, Any] = {
        "mode": "panel" if is_cs else "single_asset",
        "adapter": adapter,
        "sanitize": True,
        "duplicate_policy": "last",
    }

    if adapter == "synthetic":
        out["fields_required"] = ["date", "asset", "close", "volume", "mkt_cap", "industry"] if is_cs else [
            "date",
            "close",
        ]
        out["synthetic"] = {
            "n_assets": 60 if is_cs else 1,
            "n_days": 320,
            "seed": 2026,
            "start_date": "2020-01-01",
        }
    elif adapter in {"parquet", "csv"}:
        out["path"] = "data/panel.parquet" if adapter == "parquet" else "data/panel.csv"
        out["fields_required"] = ["date", "asset", "close", "volume", "mkt_cap", "industry"] if is_cs else [
            "date",
            "close",
        ]
    elif adapter == "sina":
        out["data_dir"] = "/stock_sina_update"
        out["fields_required"] = ["date", "asset", "close", "volume", "mkt_cap", "industry"] if is_cs else [
            "date",
            "close",
        ]
    else:  # pragma: no cover
        raise ValueError(f"Unsupported adapter: {adapter}")

    return out


def _base_research(scope: str) -> dict[str, Any]:
    if scope == "cs":
        return {
            "horizons": [1, 5, 10, 20],
            "quantiles": 5,
            "ic_rolling_window": 20,
            "missing_policy": "drop",
            "preprocess_steps": ["winsorize", "standardize", "neutralize"],
            "winsorize": {
                "enabled": True,
                "method": "quantile",
                "lower_q": 0.01,
                "upper_q": 0.99,
                "mad_scale": 5.0,
            },
            "neutralize": {
                "enabled": True,
                "mode": "both",
            },
        }
    return {
        "horizons": [1, 5, 10, 20],
        "quantiles": 5,
        "ic_rolling_window": 30,
        "ts_standardize_window": 60,
        "ts_quantile_lookback": 80,
    }


def build_template(scope: str, adapter: str, factors: list[str]) -> dict[str, Any]:
    """Build a default YAML template payload."""
    scope = scope.strip().lower()
    adapter = adapter.strip().lower()
    if scope not in {"cs", "ts"}:
        raise ValueError("scope must be one of: cs, ts")
    if adapter not in {"synthetic", "parquet", "csv", "sina"}:
        raise ValueError("adapter must be one of: synthetic, parquet, csv, sina")
    if scope == "ts" and adapter == "sina":
        # Sina adapter is panel-oriented; TS users can still reduce to single-asset at runtime.
        pass

    factor_list = factors if factors else (["momentum_20", "volatility_20"] if scope == "ts" else [
        "momentum_20",
        "volatility_20",
        "liquidity_shock",
        "size",
    ])
    return {
        "run": _base_run(scope),
        "data": _base_data(scope, adapter),
        "factor": {
            "names": factor_list,
            "on_missing": "raise",
        },
        "research": _base_research(scope),
        "backtest": {
            "enabled": False,
            "strategy": {
                "mode": "sign" if scope == "ts" else "longshort",
            },
            "commission_bps": 3.0,
            "slippage_bps": 2.0,
            "leverage": 1.0,
            "execution_delay_days": 1,
            "execution_price_col": "close",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TS/CS run-config template YAML.",
        epilog=(
            "Examples:\n"
            "  python apps/generate_run_config.py --scope cs --adapter synthetic --out configs/generated_cs.yaml\n"
            "  python apps/generate_run_config.py --scope cs --adapter parquet --set data.path=data/panel.parquet --set factor.on_missing=warn_skip --out configs/generated_cs_parquet.yaml\n"
            "  python apps/generate_run_config.py --scope ts --factors momentum_20,volatility_20 --out configs/generated_ts.yaml\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--scope", required=True, choices=["cs", "ts"], help="Factor scope")
    parser.add_argument("--adapter", default="synthetic", choices=["synthetic", "parquet", "csv", "sina"])
    parser.add_argument(
        "--factors",
        default="",
        help="Comma-separated factor names. If empty, use scope defaults.",
    )
    parser.add_argument("--out", required=True, help="Output yaml path")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override key via dotted path, e.g. research.quantiles=10 (repeatable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    factors = [x.strip() for x in args.factors.split(",") if x.strip()]
    payload = build_template(scope=args.scope, adapter=args.adapter, factors=factors)
    for ov in args.overrides:
        payload = apply_config_override(payload, ov)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()

