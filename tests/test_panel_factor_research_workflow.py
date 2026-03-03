"""面板研究工作流封装测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from factorlab.config import SyntheticConfig
from factorlab.data import generate_synthetic_panel, write_panel
from factorlab.workflows import (
    PanelFactorResearchConfig,
    build_panel_factor_research_run_config,
    run_panel_factor_research,
)


def _make_panel_file(tmp_path: Path) -> Path:
    panel = generate_synthetic_panel(SyntheticConfig(n_assets=10, n_days=140, seed=2026))
    out = tmp_path / "panel.parquet"
    write_panel(panel, out)
    return out


def test_build_panel_factor_research_run_config_normalization() -> None:
    cfg = PanelFactorResearchConfig(
        factors="momentum_20, volatility_20",
        horizons="1, 5, 10",
        preprocess_steps="winsorize,standardize",
        neutralize="none",
        winsorize="mad",
        standardization="cs_rank",
        missing_policy="drop",
        quantiles=7,
        ic_rolling_window=18,
        on_missing_factor="warn_skip",
    )
    out = build_panel_factor_research_run_config(panel_path="data/panel.parquet", config=cfg)
    assert out["data"]["path"] == "data/panel.parquet"
    assert out["factor"]["names"] == ["momentum_20", "volatility_20"]
    assert out["research"]["horizons"] == [1, 5, 10]
    assert out["research"]["preprocess_steps"] == ["winsorize", "standardize"]
    assert out["research"]["neutralize"]["enabled"] is False
    assert out["research"]["winsorize"]["method"] == "mad"
    assert out["run"]["standardization"] == "cs_rank"


def test_run_panel_factor_research_smoke(tmp_path: Path) -> None:
    panel_path = _make_panel_file(tmp_path)
    out_dir = tmp_path / "out"
    cfg = PanelFactorResearchConfig(
        factors=["momentum_20", "volatility_20"],
        horizons=[1, 5],
        preprocess_steps=["winsorize", "standardize", "neutralize"],
        neutralize="both",
        winsorize="quantile",
        standardization="cs_zscore",
        missing_policy="drop",
        quantiles=5,
        ic_rolling_window=20,
        on_missing_factor="warn_skip",
    )
    result = run_panel_factor_research(
        panel_path=panel_path,
        out_dir=out_dir,
        config=cfg,
        validate_schema=True,
    )
    assert result.index_html.exists()
    assert result.summary_csv.exists()
    assert result.run_meta_json.exists()
    summary = pd.read_csv(result.summary_csv)
    assert not summary.empty

