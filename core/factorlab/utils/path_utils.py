"""模块说明。"""

from __future__ import annotations

import re
from pathlib import Path


_SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")


def safe_slug(text: str, default: str = "item", max_len: int = 80) -> str:
    """将任意文本转换为稳定、安全的路径片段。"""
    raw = str(text).strip()
    slug = _SLUG_PATTERN.sub("_", raw).strip("._-")
    if not slug:
        slug = str(default).strip() or "item"
    return slug[: max(8, int(max_len))]


def ensure_within(base_dir: str | Path, candidate: str | Path) -> Path:
    """校验 candidate 真实路径位于 base_dir 之内。"""
    base = Path(base_dir).resolve()
    target = Path(candidate).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"Unsafe output path outside base_dir: {target}") from exc
    return target
