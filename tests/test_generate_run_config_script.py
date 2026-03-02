"""Smoke tests for config template generator CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def test_generate_run_config_script_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "generated_cs.yaml"
    cmd = [
        sys.executable,
        str(root / "apps" / "generate_run_config.py"),
        "--scope",
        "cs",
        "--adapter",
        "synthetic",
        "--set",
        "research.quantiles=7",
        "--set",
        "factor.on_missing=warn_skip",
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    assert out_path.exists()

    cfg = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert cfg["run"]["factor_scope"] == "cs"
    assert cfg["run"]["eval_axis"] == "cross_section"
    assert cfg["research"]["quantiles"] == 7
    assert cfg["factor"]["on_missing"] == "warn_skip"
    assert cfg["factor"]["combinations"] == []


def test_generate_run_config_supports_stooq_adapter(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    out_path = tmp_path / "generated_stooq.yaml"
    cmd = [
        sys.executable,
        str(root / "apps" / "generate_run_config.py"),
        "--scope",
        "cs",
        "--adapter",
        "stooq",
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    cfg = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert cfg["data"]["adapter"] == "stooq"
    assert isinstance(cfg["data"]["symbols"], list) and cfg["data"]["symbols"]
    assert cfg["data"]["min_rows_per_asset"] == 30
