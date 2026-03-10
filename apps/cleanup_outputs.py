"""兼容入口：请优先使用 `python -m factorlab cleanup-outputs ...`。"""

from __future__ import annotations

from _bootstrap import ensure_core_path

ensure_core_path(__file__)

from factorlab.cli.main import cleanup_outputs_main  # noqa: E402


if __name__ == "__main__":
    cleanup_outputs_main()
