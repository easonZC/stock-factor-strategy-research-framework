"""工作流插件/组件运行前预检工具。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from factorlab.utils import get_logger

LOGGER = get_logger("factorlab.workflows.plugin_preflight")


@dataclass(slots=True)
class PreflightReport:
    """预检结果。"""

    kind: str
    requested: list[str]
    available: list[str]
    resolved: list[str]
    missing: list[str]
    on_missing: str
    alias_hits: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def preflight_requested_components(
    *,
    kind: str,
    requested: Iterable[str],
    available: Iterable[str],
    on_missing: str = "raise",
    alias_map: dict[str, str] | None = None,
    logger_name: str = "factorlab.workflows.plugin_preflight",
) -> PreflightReport:
    """根据可用注册表对请求组件执行统一预检。"""
    logger = get_logger(logger_name)
    available_raw = [str(x).strip() for x in available if str(x).strip()]
    available_map: dict[str, str] = {}
    for item in available_raw:
        key = item.lower()
        if key not in available_map:
            available_map[key] = item
    available_set = set(available_map.keys())
    alias_map_norm = {str(k).strip().lower(): str(v).strip().lower() for k, v in (alias_map or {}).items() if str(k).strip() and str(v).strip()}

    requested_list: list[str] = []
    resolved: list[str] = []
    missing: list[str] = []
    alias_hits: dict[str, str] = {}
    seen_resolved: set[str] = set()

    for raw in requested:
        text = str(raw).strip()
        if not text:
            continue
        requested_list.append(text)
        key = text.lower()
        candidate = alias_map_norm.get(key, key)
        if candidate in available_set:
            canonical = available_map[candidate]
            if canonical not in seen_resolved:
                seen_resolved.add(canonical)
                resolved.append(canonical)
            if candidate != key:
                alias_hits[text] = canonical
            continue
        missing.append(text)

    report = PreflightReport(
        kind=str(kind).strip().lower(),
        requested=requested_list,
        available=sorted(available_raw),
        resolved=resolved,
        missing=missing,
        on_missing=str(on_missing).strip().lower(),
        alias_hits=alias_hits,
    )
    if missing:
        msg = (
            f"{report.kind} preflight missing requested entries: {missing}. "
            f"available={report.available}"
        )
        if report.on_missing == "warn_skip":
            logger.warning(msg)
        else:
            raise KeyError(msg)
    return report


__all__ = [
    "PreflightReport",
    "preflight_requested_components",
]
