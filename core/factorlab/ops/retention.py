"""输出目录保留与清理策略实现。"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass(slots=True)
class RetentionPolicy:
    """留存策略配置。"""

    older_than_days: int = 14
    keep_latest: int = 20
    dry_run: bool = False


@dataclass(slots=True)
class RetentionRunResult:
    """留存清理执行结果。"""

    root_dir: str
    scanned: int
    removed: int
    kept: int
    removed_paths: list[str] = field(default_factory=list)
    kept_paths: list[str] = field(default_factory=list)


class OutputRetentionManager:
    """按策略清理输出目录中的历史运行结果。"""

    def __init__(self, root_dir: str | Path, policy: RetentionPolicy):
        self.root_dir = Path(root_dir)
        self.policy = policy

    def _is_run_dir(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        return (path / "index.html").exists() and (path / "run_meta.json").exists()

    def _discover_run_dirs(self) -> list[Path]:
        if not self.root_dir.exists() or not self.root_dir.is_dir():
            return []
        candidates = [p for p in self.root_dir.rglob("*") if self._is_run_dir(p)]
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates

    def cleanup(self) -> RetentionRunResult:
        run_dirs = self._discover_run_dirs()
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=max(0, int(self.policy.older_than_days)))

        keep_latest = max(0, int(self.policy.keep_latest))
        latest_keep_set = set(run_dirs[:keep_latest])

        removed_paths: list[str] = []
        kept_paths: list[str] = []

        for run_dir in run_dirs:
            mtime = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc)
            should_remove = (run_dir not in latest_keep_set) and (mtime < cutoff)
            if should_remove:
                removed_paths.append(str(run_dir))
                if not self.policy.dry_run:
                    shutil.rmtree(run_dir, ignore_errors=True)
            else:
                kept_paths.append(str(run_dir))

        return RetentionRunResult(
            root_dir=str(self.root_dir),
            scanned=len(run_dirs),
            removed=len(removed_paths),
            kept=len(kept_paths),
            removed_paths=removed_paths,
            kept_paths=kept_paths,
        )
