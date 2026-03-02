"""运行配置模板生成器。

用于按 TS/CS 作用域与数据适配器快速生成可运行 YAML，并支持 `--set` 临时覆盖。
"""

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
        "config_mode": "warn",
        "fail_on_autocorrect": False,
        "leakage_guard_mode": "strict",
        "research_profile": "full",
    }


def _base_data(scope: str, adapter: str) -> dict[str, Any]:
    is_cs = scope == "cs"
    out: dict[str, Any] = {
        "mode": "panel" if is_cs else "single_asset",
        "adapter": adapter,
        "sanitize": True,
        "duplicate_policy": "last",
        "adapter_auto_discover": False,
        "adapter_plugin_dirs": [],
        "adapter_plugins": [],
        "adapter_plugin_on_error": "warn_skip",
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
        out["min_rows_per_asset"] = 30
        out["fields_required"] = ["date", "asset", "close", "volume", "mkt_cap", "industry"] if is_cs else [
            "date",
            "close",
        ]
    elif adapter == "stooq":
        out["symbols"] = ["aapl", "msft", "googl"] if is_cs else ["aapl"]
        out["start_date"] = "2020-01-01"
        out["end_date"] = None
        out["request_timeout_sec"] = 20
        out["min_rows_per_asset"] = 30
        out["fields_required"] = ["date", "asset", "close", "volume"] if is_cs else ["date", "close"]
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
            "transform_auto_discover": False,
            "transform_plugin_dirs": [],
            "transform_plugins": [],
            "transform_plugin_on_error": "warn_skip",
            "custom_transforms": [],
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
        "transform_auto_discover": False,
        "transform_plugin_dirs": [],
        "transform_plugins": [],
        "transform_plugin_on_error": "warn_skip",
        "custom_transforms": [],
    }


def build_template(scope: str, adapter: str, factors: list[str]) -> dict[str, Any]:
    """构建默认 YAML 模板内容。"""
    scope = scope.strip().lower()
    adapter = adapter.strip().lower()
    if scope not in {"cs", "ts"}:
        raise ValueError("scope must be one of: cs, ts")
    if adapter not in {"synthetic", "parquet", "csv", "sina", "stooq"}:
        raise ValueError("adapter must be one of: synthetic, parquet, csv, sina, stooq")
    if scope == "ts" and adapter == "sina":
        # 新浪适配器天然输出面板，时序场景可在运行时再筛到单资产。
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
            "on_missing": "warn_skip",
            "auto_discover": False,
            "plugin_dirs": [],
            "plugins": [],
            "plugin_on_error": "warn_skip",
            "expressions": {},
            "expression_on_error": "warn_skip",
            "combinations": [],
            "combination_on_error": "warn_skip",
        },
        "research": _base_research(scope),
        "backtest": {
            "enabled": False,
            "strategy": {
                "mode": "sign" if scope == "ts" else "longshort",
                "auto_discover": False,
                "plugin_dirs": [],
                "plugins": [],
                "plugin_on_error": "raise",
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
            "  python apps/generate_run_config.py --scope cs --set research.custom_transforms='[{\"name\":\"clip\",\"kwargs\":{\"lower\":-4,\"upper\":4}}]' --out configs/generated_cs_custom.yaml\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--scope", required=True, choices=["cs", "ts"], help="Factor scope")
    parser.add_argument("--adapter", default="synthetic", choices=["synthetic", "parquet", "csv", "sina", "stooq"])
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
