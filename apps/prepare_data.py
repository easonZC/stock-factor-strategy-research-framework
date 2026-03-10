"""兼容入口：请优先使用 `python -m factorlab prepare-data ...`。"""

from __future__ import annotations

from _bootstrap import ensure_core_path

ensure_core_path(__file__)

from factorlab.cli.main import prepare_data_main  # noqa: E402


if __name__ == "__main__":
    prepare_data_main()
