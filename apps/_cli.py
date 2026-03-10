"""兼容层：CLI 参数工具已迁移到 `factorlab.cli.utils`。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from factorlab.cli.utils import add_logging_args, add_output_args, setup_logging_from_args

__all__ = ["add_logging_args", "add_output_args", "setup_logging_from_args"]
