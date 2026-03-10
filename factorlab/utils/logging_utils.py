"""日志工具：支持级别控制与可选文件输出。"""

from __future__ import annotations

import logging
import os
from pathlib import Path


def _resolve_level(level: str | None) -> int:
    text = str(level or os.getenv("FACTORLAB_LOG_LEVEL", "INFO")).strip().upper()
    return getattr(logging, text, logging.INFO)


def configure_logging(
    level: str | None = None,
    log_file: str | Path | None = None,
    force: bool = False,
) -> logging.Logger:
    root = logging.getLogger("factorlab")
    if force:
        for h in list(root.handlers):
            root.removeHandler(h)
            h.close()

    root.setLevel(_resolve_level(level))
    root.propagate = False
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    if not root.handlers:
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

    if log_file:
        log_path = Path(log_file).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        existing = {
            getattr(h, "baseFilename", None)
            for h in root.handlers
            if isinstance(h, logging.FileHandler)
        }
        if str(log_path.resolve()) not in existing:
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setFormatter(formatter)
            root.addHandler(fh)

    return root


def get_logger(name: str = "factorlab") -> logging.Logger:
    root = configure_logging()
    if name == "factorlab":
        return root

    logger = logging.getLogger(name)
    if name.startswith("factorlab."):
        logger.handlers = []
        logger.propagate = True
    return logger
