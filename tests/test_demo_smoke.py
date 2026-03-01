"""Smoke test for synthetic end-to-end report generation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_demo_report_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "factor_report_demo"
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / "scripts" / "demo_factor_research.py"),
        "--out",
        str(out_dir),
        "--assets",
        "12",
        "--days",
        "120",
    ]
    subprocess.run(cmd, check=True)

    assert (out_dir / "index.html").exists()
    assert (out_dir / "tables" / "summary.csv").exists()

    pngs = list((out_dir / "assets").glob("*.png"))
    assert len(pngs) >= 6
