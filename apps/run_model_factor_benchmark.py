"""兼容入口：请优先使用 `python -m factorlab run-model-benchmark ...`。"""

from __future__ import annotations

from _bootstrap import ensure_core_path

ensure_core_path(__file__)

from factorlab.cli.main import model_benchmark_main  # noqa: E402


if __name__ == "__main__":
    model_benchmark_main()
