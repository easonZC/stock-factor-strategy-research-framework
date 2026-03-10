"""报告产物治理测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from factorlab.research.report import ReportRenderer


def test_report_catalog_deduplicates_overview_and_tracks_figure_sources(tmp_path: Path) -> None:
    out_dir = tmp_path / "report"
    figure_dir = out_dir / "assets" / "detail" / "alpha_factor__raw"
    table_dir = out_dir / "tables" / "detail" / "alpha_factor__raw"
    overview_dir = out_dir / "tables" / "overview"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    overview_dir.mkdir(parents=True, exist_ok=True)

    figure_path = figure_dir / "ic.png"
    source_table = table_dir / "ic_daily_h1.csv"
    strategy_definitions = overview_dir / "strategy_definitions.csv"
    data_lineage = out_dir / "data_lineage.json"
    figure_path.write_bytes(b"png")
    source_table.write_text("date,ic\n2020-01-01,0.1\n", encoding="utf-8")
    strategy_definitions.write_text(
        "name,family,constraints,description,parameters_json\nalpha,long_only,long_only,test,{}\n",
        encoding="utf-8",
    )
    data_lineage.write_text(
        json.dumps({"source": {"adapter": "csv", "mode": "panel"}, "panel_profile": {"rows": 1}, "fingerprint": "abc"}),
        encoding="utf-8",
    )

    summary = pd.DataFrame(
        {
            "factor": ["alpha_factor"],
            "variant": ["raw"],
            "horizon": [1],
            "rank_ic_mean": [0.12],
        }
    )

    ReportRenderer(out_dir=out_dir).render(
        summary=summary,
        figure_map={"alpha_factor": [figure_path]},
        table_map={"alpha_factor": [source_table]},
        figure_sources={
            figure_path: {
                "factor": "alpha_factor",
                "variant": "raw",
                "chart": "ic.png",
                "label": "IC Series",
                "source_tables": [source_table],
                "description": "Unit-test IC provenance.",
            }
        },
        overview_files={
            "data_lineage": data_lineage,
            "strategy_definitions": strategy_definitions,
        },
    )

    manifest = json.loads((out_dir / "overview" / "manifest.json").read_text(encoding="utf-8"))
    catalog = json.loads((out_dir / "artifact_catalog.json").read_text(encoding="utf-8"))
    attribution = (out_dir / "tables" / "overview" / "figure_attribution.csv").read_text(encoding="utf-8")
    index_html = (out_dir / "index.html").read_text(encoding="utf-8")

    assert manifest["navigation_only"] is True
    assert manifest["files"]["quick_summary"] == "tables/overview/quick_summary.csv"
    assert manifest["files"]["data_lineage"] == "data_lineage.json"
    assert manifest["key_figures"][0]["source_tables"] == ["tables/detail/alpha_factor__raw/ic_daily_h1.csv"]
    assert catalog["deduplicated_overview"] is True
    assert (out_dir / "tables" / "overview" / "quick_summary.csv").exists()
    assert not (out_dir / "overview" / "factor_scorecard.csv").exists()
    assert not (out_dir / "assets" / "key").exists()
    assert "tables/detail/alpha_factor__raw/ic_daily_h1.csv" in attribution
    assert "data_lineage.json" in index_html
    assert "strategy_definitions.csv" in index_html
