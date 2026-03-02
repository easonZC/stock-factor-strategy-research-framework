"""Small logging utility to keep script output explicit and consistent."""

from __future__ import annotations

import logging


def get_logger(name: str = "factorlab") -> logging.Logger:
    """Return a configured logger for console output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
