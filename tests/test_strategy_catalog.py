"""策略目录与定义治理测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from factorlab.strategies import build_strategy_registry, describe_strategy_registry


def test_builtin_strategy_has_catalog_metadata() -> None:
    registry = build_strategy_registry()
    definitions = describe_strategy_registry(registry, names=["longshort"])

    assert len(definitions) == 1
    item = definitions[0]
    assert item.name == "long_short_quantile"
    assert item.family == "long_short"
    assert set(item.constraints) == {"dollar_neutral", "symmetric_quantiles", "optional_max_weight"}
    assert "Dollar-neutral long-short" in item.description


def test_list_strategies_script_shows_strategy_metadata() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "apps/list_strategies.py",
            "--json",
            "--name",
            "meanvar",
        ],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload[0]["name"] == "mean_variance_optimizer"
    assert payload[0]["family"] == "optimizer"
    assert "gross_target" in payload[0]["constraints"]
