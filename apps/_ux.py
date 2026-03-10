"""兼容层：CLI UX 工具已迁移到 `factorlab.cli.utils`。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factorlab.cli.utils import render_run_summary, resolve_output_dir

__all__ = ["render_run_summary", "resolve_output_dir"]
