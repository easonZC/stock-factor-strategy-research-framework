"""统一 CLI 与结构兼容测试。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from factorlab.cli.main import build_parser
from factorlab.workflows import compose_run_config


def test_scripts_factorlab_lists_strategy_catalog() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/factorlab.py", "list-strategies", "--json", "--name", "meanvar"],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload[0]["name"] == "mean_variance_optimizer"
    assert payload[0]["family"] == "optimizer"


def test_legacy_config_shim_matches_canonical_example() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    legacy = compose_run_config([repo_root / "configs" / "cs_factor.yaml"])
    canonical = compose_run_config([repo_root / "examples" / "workflows" / "cs_factor.yaml"])
    assert legacy == canonical


def test_python_module_entrypoint_lists_factor_catalog() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "factorlab", "list-factors", "--json", "--name", "momentum_20"],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload[0]["name"] == "momentum_20"


def test_cleanup_outputs_subcommand_supports_json(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    root = tmp_path / "outputs"
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_a").mkdir()
    proc = subprocess.run(
        [sys.executable, "scripts/factorlab.py", "cleanup-outputs", "--root", str(root), "--dry-run", "--json"],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["root_dir"].endswith("outputs")
    assert payload["dry_run"] is True


def test_train_model_factor_parser_accepts_mlp() -> None:
    parser = build_parser()
    args = parser.parse_args(["train-model-factor", "--model", "mlp"])
    assert args.model == "mlp"
