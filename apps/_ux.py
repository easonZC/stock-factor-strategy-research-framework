"""应用入口的人机友好工具。"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any


def _slug(text: str, default: str) -> str:
    value = str(text).strip().lower()
    if not value:
        return default
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value or default


def _timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_output_dir(
    *,
    out: str | None,
    run_name: str | None,
    category: str,
    default_name: str,
) -> Path:
    """解析输出目录。

    优先级：
    1) 显式 --out
    2) 自动目录：outputs/research/<category>/<run_name>_<timestamp>
    """
    if out:
        return Path(out)
    base = Path("outputs") / "research" / _slug(category, "misc")
    name = _slug(run_name or default_name, "run")
    return base / f"{name}_{_timestamp_now()}"


def render_run_summary(title: str, lines: dict[str, Any]) -> str:
    """生成统一的命令行结果摘要。"""
    parts = [f"[{title}]"]
    for key, val in lines.items():
        parts.append(f"- {key}: {val}")
    return "\n".join(parts)


__all__ = [
    "render_run_summary",
    "resolve_output_dir",
]

