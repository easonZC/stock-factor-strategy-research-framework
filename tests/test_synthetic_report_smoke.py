"""Smoke test for synthetic end-to-end report generation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_synthetic_report_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "research" / "factor" / "synthetic_report"
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / "apps" / "run_factor_research_synthetic.py"),
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

    pngs = list((out_dir / "assets").rglob("*.png"))
    assert len(pngs) >= 6
