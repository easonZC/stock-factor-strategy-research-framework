"""因子目录与定义治理测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from factorlab.config import SyntheticConfig
from factorlab.data import generate_synthetic_panel
from factorlab.factors import apply_factors, build_factor_registry, describe_factor_registry


def test_new_builtin_factor_has_catalog_metadata_and_values() -> None:
    registry = build_factor_registry()
    panel = generate_synthetic_panel(SyntheticConfig(n_assets=6, n_days=90, seed=9))

    out = apply_factors(panel, ["volume_price_pressure_20"], inplace=False, registry=registry)
    assert "volume_price_pressure_20" in out.columns
    assert out["volume_price_pressure_20"].notna().mean() > 0.4

    definitions = describe_factor_registry(registry, names=["volume_price_pressure_20"])
    assert len(definitions) == 1
    item = definitions[0]
    assert item.family == "price_volume"
    assert set(item.required_columns) == {"close", "volume"}
    assert "relative volume shock" in item.description
    assert "rolling_mean" in item.formula


def test_list_factors_script_shows_new_factor_metadata() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "apps/list_factors.py",
            "--json",
            "--name",
            "volume_price_pressure_20",
        ],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload[0]["name"] == "volume_price_pressure_20"
    assert payload[0]["required_columns"] == ["close", "volume"]
    assert payload[0]["family"] == "price_volume"
