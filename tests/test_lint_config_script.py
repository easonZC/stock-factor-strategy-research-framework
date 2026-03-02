"""配置体检脚本测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


def _run_cmd(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def test_lint_config_json_reports_alias_migration(tmp_path: Path) -> None:
    cfg = {
        "run": {
            "factor_scope": "cs",
            "eval_axis": "cross_section",
            "std": "cs_rank",
        },
        "data": {
            "mode": "panel",
            "adapter": "synthetic",
            "fields_required": ["date", "asset", "close", "volume", "mkt_cap", "industry"],
            "synthetic": {"n_assets": 6, "n_days": 120, "seed": 1, "start_date": "2020-01-01"},
        },
        "factor": {"names": ["momentum_20"]},
        "research": {"q": 5},
    }
    cfg_path = tmp_path / "lint_alias.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cmd(
        [sys.executable, "apps/lint_config.py", "--config", str(cfg_path), "--json"],
        cwd=repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["alias_migration_count"] >= 1


def test_lint_config_strict_fails_on_schema_errors(tmp_path: Path) -> None:
    cfg = {
        "run": {
            "factor_scope": "cs",
            "eval_axis": "cross_section",
            "standardization": "cs_rank",
        },
        "data": {"mode": "panel", "adapter": "synthetic"},
        "factor": {"names": ["momentum_20"]},
        "research": {"horizons": [0]},
    }
    cfg_path = tmp_path / "lint_error.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cmd(
        [sys.executable, "apps/lint_config.py", "--config", str(cfg_path), "--strict"],
        cwd=repo_root,
    )
    assert proc.returncode != 0
