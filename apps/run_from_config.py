"""兼容入口：请优先使用 `factorlab run ...` 或 `python -m factorlab run ...`。"""

from __future__ import annotations

from _bootstrap import ensure_core_path

ensure_core_path(__file__)

from factorlab.cli.main import run_main  # noqa: E402


if __name__ == "__main__":
    run_main()
