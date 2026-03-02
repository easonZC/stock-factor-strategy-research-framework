"""应用入口公共引导工具。"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_core_path(caller_file: str) -> Path:
    """确保 `core/` 在 `sys.path` 中，并返回仓库根目录。"""
    root = Path(caller_file).resolve().parents[1]
    core_path = root / "core"
    if str(core_path) not in sys.path:
        sys.path.insert(0, str(core_path))
    return root
