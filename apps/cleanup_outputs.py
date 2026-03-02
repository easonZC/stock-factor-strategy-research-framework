"""模块说明。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "core"
if str(CORE_PATH) not in sys.path:
    sys.path.insert(0, str(CORE_PATH))

from factorlab.ops import OutputRetentionManager, RetentionPolicy
from factorlab.utils import configure_logging, get_logger

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
        "--log-level",
        default=None,
        help="日志级别（DEBUG/INFO/WARNING/ERROR），支持环境变量 FACTORLAB_LOG_LEVEL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(level=args.log_level, force=True)
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
