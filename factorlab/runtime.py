"""Shared runtime context, path, and stdio helpers."""

from __future__ import annotations

import locale
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_TEXT_ENCODING = "utf-8"


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
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else Path.cwd().resolve()
    branch = _run_git(["branch", "--show-current"], cwd=root)
    commit = _run_git(["rev-parse", "--short", "HEAD"], cwd=root)
    dirty_raw = _run_git(["status", "--porcelain"], cwd=root)
    dirty = bool(dirty_raw) if dirty_raw is not None else None

    return {
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "repo_root": str(root),
        "cwd": str(Path.cwd().resolve()),
        "encoding": {
            "preferred": locale.getpreferredencoding(False),
            "filesystem": sys.getfilesystemencoding(),
            "stdout": getattr(sys.stdout, "encoding", None),
            "stderr": getattr(sys.stderr, "encoding", None),
        },
        "git": {
            "branch": branch,
            "commit": commit,
            "dirty": dirty,
        },
    }


def enable_utf8_stdio(encoding: str = DEFAULT_TEXT_ENCODING) -> None:
    """Best-effort UTF-8 stdio setup for Windows and mixed terminal environments."""
    for name, errors in (("stdin", None), ("stdout", "replace"), ("stderr", "replace")):
        stream = getattr(sys, name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        kwargs: dict[str, Any] = {"encoding": encoding}
        if errors is not None:
            kwargs["errors"] = errors
        try:
            stream.reconfigure(**kwargs)
        except Exception:
            continue


@dataclass(frozen=True, slots=True)
class OutputContext:
    """Normalized output root plus default text encoding."""

    root: Path
    encoding: str = DEFAULT_TEXT_ENCODING

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    def ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        return candidate.resolve()

    def child(self, *parts: str | Path) -> Path:
        if not parts:
            return self.root
        path = self.root
        for part in parts:
            path = path / Path(part)
        return path

    def ensure_dir(self, *parts: str | Path) -> Path:
        path = self.child(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_text(self, path: str | Path, payload: str) -> Path:
        target = self.resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding=self.encoding)
        return target


def coerce_output_context(
    out_dir: OutputContext | str | Path,
    *,
    encoding: str = DEFAULT_TEXT_ENCODING,
) -> OutputContext:
    if isinstance(out_dir, OutputContext):
        return out_dir
    return OutputContext(root=Path(out_dir), encoding=encoding)


@dataclass(frozen=True, slots=True)
class RunContext:
    """Shared runtime context passed across CLI, workflows, and reporting."""

    repo_root: Path
    outputs: OutputContext
    runtime_manifest: dict[str, Any] = field(default_factory=dict)
    text_encoding: str = DEFAULT_TEXT_ENCODING

    def __post_init__(self) -> None:
        repo_root = Path(self.repo_root).expanduser().resolve()
        object.__setattr__(self, "repo_root", repo_root)
        object.__setattr__(self, "text_encoding", self.outputs.encoding or self.text_encoding)
        if not self.runtime_manifest:
            object.__setattr__(self, "runtime_manifest", collect_runtime_manifest(repo_root=repo_root))

    @property
    def out_dir(self) -> Path:
        return self.outputs.root

    @classmethod
    def create(
        cls,
        *,
        out_dir: OutputContext | str | Path,
        repo_root: str | Path | None = None,
        text_encoding: str = DEFAULT_TEXT_ENCODING,
    ) -> "RunContext":
        outputs = coerce_output_context(out_dir, encoding=text_encoding)
        resolved_repo = Path(repo_root).expanduser().resolve() if repo_root is not None else Path.cwd().resolve()
        return cls(
            repo_root=resolved_repo,
            outputs=outputs,
            runtime_manifest=collect_runtime_manifest(repo_root=resolved_repo),
            text_encoding=outputs.encoding,
        )


def coerce_run_context(
    *,
    run_context: RunContext | None = None,
    out_dir: OutputContext | str | Path | None = None,
    repo_root: str | Path | None = None,
    text_encoding: str = DEFAULT_TEXT_ENCODING,
) -> RunContext:
    if run_context is not None:
        return run_context
    if out_dir is None:
        raise ValueError("out_dir is required when run_context is not provided.")
    return RunContext.create(out_dir=out_dir, repo_root=repo_root, text_encoding=text_encoding)


__all__ = [
    "DEFAULT_TEXT_ENCODING",
    "OutputContext",
    "RunContext",
    "collect_runtime_manifest",
    "coerce_output_context",
    "coerce_run_context",
    "enable_utf8_stdio",
]
