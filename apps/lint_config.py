"""配置体检工具。

用于在正式运行前检查 YAML 配置质量，输出：
1. 结构错误与非阻断告警
2. 别名键迁移记录（建议改写为标准键）
3. 研究档位与运行成本建议
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "core"
if str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))

from factorlab.workflows import compose_run_config_with_alias_report, validate_run_config_schema  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lint run config before executing workflow.",
        epilog=(
            "Examples:\n"
            "  python apps/lint_config.py --config configs/cs_factor.yaml\n"
            "  python apps/lint_config.py --config configs/minimal_local.yaml\n"
            "  python apps/lint_config.py --config configs/cs_factor.yaml --set run.std=cs_rank --set research.q=10\n"
            "  python apps/lint_config.py --config configs/cs_factor.yaml --strict\n"
            "  python apps/lint_config.py --config configs/cs_factor.yaml --json\n"
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
        help="Dotted-path override (repeatable), e.g. research.horizons=[1,5,10].",
    )
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero code if schema has errors.")
    parser.add_argument("--json", action="store_true", help="Output JSON report.")
    parser.add_argument(
        "--save-effective-config",
        default=None,
        help="Optional path to save effective config after imports/merge/alias migration.",
    )
    return parser.parse_args()


def _build_suggestions(
    cfg: dict[str, Any],
    alias_events: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
) -> list[str]:
    suggestions: list[str] = []
    for event in alias_events:
        if event.get("applied"):
            suggestions.append(
                f"将别名键 `{event['alias']}` 改为标准键 `{event['canonical']}`，减少后续兼容迁移。"
            )
    run_cfg = cfg.get("run", {}) if isinstance(cfg.get("run"), dict) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    profile = str(run_cfg.get("research_profile", "full")).strip().lower()
    if profile == "full" and str(data_cfg.get("adapter", "")).strip().lower() == "synthetic":
        syn = data_cfg.get("synthetic", {}) if isinstance(data_cfg.get("synthetic"), dict) else {}
        n_assets = int(syn.get("n_assets", 0) or 0)
        n_days = int(syn.get("n_days", 0) or 0)
        if n_assets * n_days >= 40000:
            suggestions.append(
                "当前样本规模较大且 profile=full，开发阶段可先切换为 `run.research_profile=dev` 提升迭代速度。"
            )
    if not errors and not warnings and not suggestions:
        suggestions.append("配置体检通过，当前未发现需要调整的项。")
    return suggestions


def main() -> None:
    args = parse_args()
    effective_cfg, alias_events = compose_run_config_with_alias_report(
        config_paths=args.config,
        overrides=args.overrides,
    )
    schema_warnings = validate_run_config_schema(effective_cfg, strict=False)
    errors = [x[len("ERROR_AS_WARNING: ") :] for x in schema_warnings if x.startswith("ERROR_AS_WARNING: ")]
    warnings = [x for x in schema_warnings if not x.startswith("ERROR_AS_WARNING: ")]
    suggestions = _build_suggestions(
        cfg=effective_cfg,
        alias_events=alias_events,
        errors=errors,
        warnings=warnings,
    )

    if args.save_effective_config:
        out = Path(args.save_effective_config)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            yaml.safe_dump(effective_cfg, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )

    payload = {
        "ok": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "alias_migration_count": len(alias_events),
        "errors": errors,
        "warnings": warnings,
        "alias_migrations": alias_events,
        "suggestions": suggestions,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"[lint] ok={payload['ok']} errors={len(errors)} warnings={len(warnings)} aliases={len(alias_events)}")
        if errors:
            print("\n[errors]")
            for item in errors:
                print(f"- {item}")
        if warnings:
            print("\n[warnings]")
            for item in warnings:
                print(f"- {item}")
        if alias_events:
            print("\n[alias_migrations]")
            for event in alias_events:
                state = "applied" if event.get("applied") else "skipped"
                print(f"- {event['alias']} -> {event['canonical']} ({state})")
        if suggestions:
            print("\n[suggestions]")
            for item in suggestions:
                print(f"- {item}")

    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
