"""统一入口冒烟测试。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_synthetic_report_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "research" / "factor" / "synthetic_report"
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        str(root / "apps" / "run_from_config.py"),
        "--config",
        str(root / "configs" / "cs_factor.yaml"),
        "--set",
        "data.adapter=synthetic",
        "--set",
        "factor.names=[momentum_20,volatility_20,liquidity_shock]",
        "--set",
        "backtest.enabled=false",
        "--out",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    assert (out_dir / "index.html").exists()
    assert (out_dir / "tables" / "summary.csv").exists()

    pngs = list((out_dir / "assets").rglob("*.png"))
    assert len(pngs) >= 6
