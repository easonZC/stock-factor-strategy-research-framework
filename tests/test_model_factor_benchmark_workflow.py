"""相关功能测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from factorlab.config import SyntheticConfig, UniverseFilterConfig
from factorlab.data.synthetic import generate_model_factor_benchmark_panel, generate_synthetic_panel
from factorlab.workflows import ModelFactorBenchmarkConfig, run_model_factor_benchmark


def _write_model_plugin(plugin_dir) -> None:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "custom_model.py").write_text(
        """
from sklearn.linear_model import Ridge

MODEL_DEFAULTS = {
    "tiny_ridge": {"alpha": 0.3, "solver": "svd"}
}

def build_tiny_ridge_model(params):
    return Ridge(**params)
""".strip(),
        encoding="utf-8",
    )


def test_model_factor_benchmark_workflow_smoke(tmp_path) -> None:
    panel = generate_synthetic_panel(
        SyntheticConfig(
            n_assets=16,
            n_days=220,
            seed=2026,
            start_date="2020-01-01",
        )
    )
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)

    out_dir = tmp_path / "model_factor_benchmark_smoke"
    cfg = ModelFactorBenchmarkConfig(
        models="Ridge, random_forest, unsupported_model",
        feature_cols="momentum_20,volatility_20,liquidity_shock,size",
        extra_report_factors="",
        label_horizon=5,
        train_days=100,
        valid_days=20,
        step_days=20,
        purge_days=2,
        split_mode="expanding",
        scoring_metric="rank_ic",
        evaluation_axis="cross_section",
        min_train_rows=250,
        min_valid_rows=80,
        horizons="5,10",
        neutralize="INVALID",  # type: ignore[arg-type]
        winsorize="UNKNOWN",  # type: ignore[arg-type]
        duplicate_policy="BAD_POLICY",  # type: ignore[arg-type]
        preferred_metric_variant="AUTO",
        max_assets=12,
        apply_universe_filter=True,
        universe_filter=UniverseFilterConfig(
            min_close=0.0,
            min_history_days=20,
            min_median_dollar_volume=0.0,
            liquidity_lookback=20,
        ),
    )
    result = run_model_factor_benchmark(
        panel_path=panel_path,
        out_dir=out_dir,
        config=cfg,
    )

    comparison = pd.read_csv(result.comparison_csv)
    assert set(comparison["model"]) == {"ridge", "rf"}
    assert comparison["best_oof_rank_ic"].notna().all()
    assert comparison["research_horizon"].notna().all()
    assert result.summary_csv.exists()
    assert result.index_html.exists()
    assert result.run_meta_json.exists()
    assert result.run_manifest_json.exists()

    run_meta = json.loads(result.run_meta_json.read_text(encoding="utf-8"))
    assert "resolved_models" in run_meta
    assert "model_preflight_report" in run_meta
    assert run_meta["model_preflight_report"]["alias_hits"].get("random_forest") == "rf"
    assert "timings_seconds" in run_meta
    assert "warning_summary" in run_meta


def test_model_factor_benchmark_supports_model_plugin(tmp_path) -> None:
    panel = generate_synthetic_panel(
        SyntheticConfig(
            n_assets=12,
            n_days=180,
            seed=2027,
            start_date="2020-01-01",
        )
    )
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    plugin_dir = tmp_path / "plugins" / "models"
    _write_model_plugin(plugin_dir)

    cfg = ModelFactorBenchmarkConfig(
        models="tiny_ridge",
        feature_cols="momentum_20,volatility_20,liquidity_shock,size",
        label_horizon=5,
        train_days=80,
        valid_days=20,
        step_days=20,
        split_mode="expanding",
        min_train_rows=180,
        min_valid_rows=60,
        horizons="5",
        max_assets=10,
        model_auto_discover=True,
        model_plugin_dirs=[str(plugin_dir)],
        model_plugin_on_error="raise",
    )
    out_dir = tmp_path / "benchmark_model_plugin"
    result = run_model_factor_benchmark(panel_path=panel_path, out_dir=out_dir, config=cfg)
    comparison = pd.read_csv(result.comparison_csv)
    assert set(comparison["model"]) == {"tiny_ridge"}
    run_meta = json.loads(result.run_meta_json.read_text(encoding="utf-8"))
    assert run_meta["model_plugin_config"]["auto_discover"] is True
    assert "tiny_ridge" in run_meta["model_plugin_config"]["registry_models"]


def test_model_factor_benchmark_supports_mlp_alias_and_artifact(tmp_path) -> None:
    panel = generate_synthetic_panel(
        SyntheticConfig(
            n_assets=12,
            n_days=180,
            seed=2029,
            start_date="2020-01-01",
        )
    )
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)

    artifact_dir = tmp_path / "artifacts" / "models"
    cfg = ModelFactorBenchmarkConfig(
        models="nn",
        feature_cols="momentum_20,volatility_20,liquidity_shock,size",
        label_horizon=5,
        train_days=80,
        valid_days=20,
        step_days=20,
        split_mode="expanding",
        min_train_rows=180,
        min_valid_rows=60,
        horizons="5",
        max_assets=10,
        evaluation_axis="cross_section",
        save_model_artifacts=True,
        model_artifact_dir=str(artifact_dir),
    )
    out_dir = tmp_path / "benchmark_mlp_alias"
    result = run_model_factor_benchmark(panel_path=panel_path, out_dir=out_dir, config=cfg)

    comparison = pd.read_csv(result.comparison_csv)
    assert set(comparison["model"]) == {"mlp"}
    assert set(comparison["factor_name"]) == {"model_factor_oof_mlp"}
    model_path = Path(str(comparison.loc[0, "model_path"]))
    assert model_path.exists()
    assert model_path.name == "model_factor_oof_mlp.joblib"

    run_meta = json.loads(result.run_meta_json.read_text(encoding="utf-8"))
    assert run_meta["resolved_models"] == ["mlp"]
    assert run_meta["model_preflight_report"]["alias_hits"].get("nn") == "mlp"


def test_compact_benchmark_panel_delivers_strong_mlp_rank_ic(tmp_path) -> None:
    panel = generate_model_factor_benchmark_panel(
        SyntheticConfig(
            n_assets=24,
            n_days=220,
            seed=11,
            start_date="2021-01-01",
        ),
        tier="engineering_demo",
    )
    assert len(panel) == 24 * 220
    panel_path = tmp_path / "compact_benchmark_panel.parquet"
    panel.to_parquet(panel_path, index=False)

    cfg = ModelFactorBenchmarkConfig(
        models="ridge,mlp",
        feature_cols="momentum_20,volatility_20,liquidity_shock,size",
        label_horizon=5,
        train_days=100,
        valid_days=20,
        step_days=20,
        split_mode="expanding",
        min_train_rows=300,
        min_valid_rows=120,
        horizons="5",
        max_assets=24,
        evaluation_axis="cross_section",
        neutralize="none",
    )
    result = run_model_factor_benchmark(panel_path=panel_path, out_dir=tmp_path / "compact_benchmark", config=cfg)
    comparison = pd.read_csv(result.comparison_csv)
    mlp_row = comparison.loc[comparison["model"] == "mlp"].iloc[0]
    ridge_row = comparison.loc[comparison["model"] == "ridge"].iloc[0]

    assert float(mlp_row["research_rank_ic_mean"]) >= 0.55
    assert float(mlp_row["research_rank_icir"]) >= 3.5
    assert float(mlp_row["research_rank_ic_mean"]) > float(ridge_row["research_rank_ic_mean"])


def test_realistic_benchmark_panel_keeps_mlp_metrics_in_plausible_range(tmp_path) -> None:
    panel = generate_model_factor_benchmark_panel(
        SyntheticConfig(
            n_assets=40,
            n_days=260,
            seed=11,
            start_date="2021-01-01",
        ),
        tier="research_realistic",
    )
    assert len(panel) == 40 * 260
    panel_path = tmp_path / "realistic_benchmark_panel.parquet"
    panel.to_parquet(panel_path, index=False)

    cfg = ModelFactorBenchmarkConfig(
        models="mlp",
        feature_cols="momentum_20,volatility_20,liquidity_shock,size",
        label_horizon=5,
        train_days=140,
        valid_days=20,
        step_days=20,
        split_mode="expanding",
        min_train_rows=500,
        min_valid_rows=160,
        horizons="5",
        max_assets=40,
        evaluation_axis="cross_section",
        neutralize="none",
    )
    result = run_model_factor_benchmark(panel_path=panel_path, out_dir=tmp_path / "realistic_benchmark", config=cfg)
    comparison = pd.read_csv(result.comparison_csv)
    mlp_row = comparison.loc[comparison["model"] == "mlp"].iloc[0]

    assert 0.02 <= float(mlp_row["research_rank_ic_mean"]) <= 0.10
    assert 0.20 <= float(mlp_row["research_rank_icir"]) <= 1.20


def test_model_factor_benchmark_rejects_all_unsupported_models(tmp_path) -> None:
    panel = generate_synthetic_panel(
        SyntheticConfig(
            n_assets=8,
            n_days=160,
            seed=2028,
            start_date="2020-01-01",
        )
    )
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    cfg = ModelFactorBenchmarkConfig(
        models="unsupported_a,unsupported_b",
        feature_cols="momentum_20,volatility_20,liquidity_shock,size",
        train_days=80,
        valid_days=20,
        step_days=20,
        min_train_rows=120,
        min_valid_rows=40,
    )
    with pytest.raises(ValueError, match="No valid models"):
        run_model_factor_benchmark(panel_path=panel_path, out_dir=tmp_path / "out_invalid_models", config=cfg)
