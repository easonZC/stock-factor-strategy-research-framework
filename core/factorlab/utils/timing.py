"""阶段计时与性能记录工具。"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from factorlab.utils.logging_utils import get_logger


@contextmanager
def timed_stage(
    stage: str,
    timings: dict[str, float] | None = None,
    logger_name: str = "factorlab",
) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = float(time.perf_counter() - start)
        if timings is not None:
            timings[str(stage)] = elapsed
        logger = get_logger(logger_name)
        logger.info("Stage '%s' finished in %.3fs", stage, elapsed)

