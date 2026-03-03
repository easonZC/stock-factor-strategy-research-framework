"""历史输出清理工具。

按时间与保留数量策略清理 `outputs/` 下的旧运行目录。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from _bootstrap import ensure_core_path
from _cli import add_logging_args, setup_logging_from_args

ensure_core_path(__file__)

from factorlab.ops import OutputRetentionManager, RetentionPolicy
from factorlab.utils import get_logger

LOGGER = get_logger("factorlab.cleanup_outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按策略清理 outputs 历史运行目录。",
        epilog=(
            "示例:\n"
            "  python apps/cleanup_outputs.py --root outputs/research --older-than-days 7 --keep-latest 30\n"
            "  python apps/cleanup_outputs.py --root outputs/research --older-than-days 14 --keep-latest 20 --dry-run\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--root", default="outputs/research", help="待清理根目录")
    parser.add_argument("--older-than-days", type=int, default=14, help="仅删除早于该天数的运行目录")
    parser.add_argument("--keep-latest", type=int, default=20, help="无论时间如何，至少保留最新 N 个运行目录")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不实际删除")
    parser.add_argument("--json", action="store_true", help="输出 JSON 结果")
    parser.add_argument(
        "--purge-all",
        action="store_true",
        help="直接清空 root 下所有内容（忽略时间与保留策略）。",
    )
    add_logging_args(parser)
    return parser.parse_args()


def _purge_all(root: str) -> dict[str, object]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        return {
            "root_dir": str(root_path),
            "scanned": 0,
            "removed": 0,
            "kept": 0,
            "dry_run": False,
            "removed_paths": [],
        }
    if str(root_path) in {"/", str(Path.home().resolve())}:
        raise ValueError(f"Refuse to purge unsafe root: {root_path}")

    removed_paths: list[str] = []
    scanned = 0
    removed = 0
    for p in root_path.iterdir():
        scanned += 1
        removed_paths.append(str(p))
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)
        removed += 1
    root_path.mkdir(parents=True, exist_ok=True)
    return {
        "root_dir": str(root_path),
        "scanned": scanned,
        "removed": removed,
        "kept": 0,
        "dry_run": False,
        "removed_paths": removed_paths,
    }


def main() -> None:
    args = parse_args()
    setup_logging_from_args(args)
    if args.purge_all:
        payload = _purge_all(args.root)
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return
        LOGGER.info(
            "purge completed: root=%s scanned=%s removed=%s",
            payload["root_dir"],
            payload["scanned"],
            payload["removed"],
        )
        return

    policy = RetentionPolicy(
        older_than_days=int(args.older_than_days),
        keep_latest=int(args.keep_latest),
        dry_run=bool(args.dry_run),
    )
    result = OutputRetentionManager(root_dir=args.root, policy=policy).cleanup()

    payload = {
        "root_dir": result.root_dir,
        "scanned": result.scanned,
        "removed": result.removed,
        "kept": result.kept,
        "dry_run": bool(args.dry_run),
        "removed_paths": result.removed_paths,
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    LOGGER.info(
        "cleanup completed: root=%s scanned=%s removed=%s kept=%s dry_run=%s",
        result.root_dir,
        result.scanned,
        result.removed,
        result.kept,
        args.dry_run,
    )
    if result.removed_paths:
        LOGGER.info("removed sample: %s", result.removed_paths[:5])


if __name__ == "__main__":
    main()
