"""Runtime and git metadata helpers for experiment traceability."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return out.strip() or None


def collect_runtime_manifest(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Collect reproducibility metadata for a workflow run."""
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    branch = _run_git(["branch", "--show-current"], cwd=root)
    commit = _run_git(["rev-parse", "--short", "HEAD"], cwd=root)
    dirty_raw = _run_git(["status", "--porcelain"], cwd=root)
    dirty = bool(dirty_raw) if dirty_raw is not None else None

    return {
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "repo_root": str(root.resolve()),
        "git": {
            "branch": branch,
            "commit": commit,
            "dirty": dirty,
        },
    }
