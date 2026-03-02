"""输出目录留存清理测试。"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from factorlab.ops import OutputRetentionManager, RetentionPolicy


def _touch_run_dir(path: Path, age_days: int) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "index.html").write_text("ok", encoding="utf-8")
    (path / "run_meta.json").write_text("{}", encoding="utf-8")
    ts = (datetime.now(timezone.utc) - timedelta(days=age_days)).timestamp()
    for item in [path, path / "index.html", path / "run_meta.json"]:
        item.touch(exist_ok=True)
        item.chmod(0o644 if item.is_file() else 0o755)
        os.utime(item, (ts, ts))
    return path


def test_retention_manager_removes_old_runs(tmp_path: Path) -> None:
    root = tmp_path / "outputs" / "research"
    new_run = _touch_run_dir(root / "factor" / "run_new", age_days=1)
    old_run_1 = _touch_run_dir(root / "factor" / "run_old_1", age_days=30)
    old_run_2 = _touch_run_dir(root / "strategy" / "run_old_2", age_days=20)

    res = OutputRetentionManager(
        root_dir=root,
        policy=RetentionPolicy(older_than_days=7, keep_latest=1, dry_run=False),
    ).cleanup()

    assert res.scanned == 3
    assert res.removed == 2
    assert res.kept == 1
    assert new_run.exists()
    assert not old_run_1.exists()
    assert not old_run_2.exists()


def test_retention_manager_dry_run_keeps_files(tmp_path: Path) -> None:
    root = tmp_path / "outputs" / "research"
    old_run = _touch_run_dir(root / "factor" / "run_old", age_days=40)

    res = OutputRetentionManager(
        root_dir=root,
        policy=RetentionPolicy(older_than_days=7, keep_latest=0, dry_run=True),
    ).cleanup()

    assert res.scanned == 1
    assert res.removed == 1
    assert old_run.exists()
