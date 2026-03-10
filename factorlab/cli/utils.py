"""CLI 共享参数与输出工具。"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def add_output_args(parser: argparse.ArgumentParser, category: str) -> None:
    parser.add_argument(
        "--out",
        default=None,
        help=f"输出目录；不填则自动生成到 outputs/research/{category}/<name>_<timestamp>",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="运行名称（仅在未提供 --out 时生效）。",
    )


def add_logging_args(parser: argparse.ArgumentParser, *, include_log_file: bool = False) -> None:
    parser.add_argument(
        "--log-level",
        default=None,
        help="日志级别（DEBUG/INFO/WARNING/ERROR），支持环境变量 FACTORLAB_LOG_LEVEL。",
    )
    if include_log_file:
        parser.add_argument(
            "--log-file",
            default=None,
            help="日志文件路径（可选）。",
        )


def setup_logging_from_args(args: argparse.Namespace) -> None:
    from factorlab.utils import configure_logging

    configure_logging(
        level=getattr(args, "log_level", None),
        log_file=getattr(args, "log_file", None),
        force=True,
    )


def _slug(text: str, default: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return value or default


def resolve_output_dir(
    *,
    out: str | None,
    run_name: str | None,
    category: str,
    default_name: str,
) -> Path:
    if out:
        return Path(out)
    base = Path("outputs") / "research" / _slug(category, "misc")
    return base / f"{_slug(run_name or default_name, 'run')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def render_run_summary(title: str, lines: dict[str, Any]) -> str:
    return "\n".join([f"[{title}]", *[f"- {key}: {val}" for key, val in lines.items()]])


__all__ = [
    "add_logging_args",
    "add_output_args",
    "render_run_summary",
    "resolve_output_dir",
    "setup_logging_from_args",
]
