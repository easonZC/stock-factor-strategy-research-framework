"""Smoke tests for reusable model-factor benchmark workflow."""

from __future__ import annotations

import json

import pandas as pd

from ssf.config import SyntheticConfig, UniverseFilterConfig
from ssf.data.synthetic import generate_synthetic_panel
from ssf.workflows import ModelFactorBenchmarkConfig, run_model_factor_benchmark


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
