"""Minimal reporting outputs for stage-stop workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from factorlab.reporting import render_report
from factorlab.runtime import OutputContext, coerce_output_context


def write_stage_only_outputs(
    out_dir: OutputContext | str | Path,
    stage: str,
    summary_row: dict[str, Any],
    extra_tables: list[Path] | None = None,
    config_payload: dict[str, Any] | None = None,
    overview_files: dict[str, Path] | None = None,
) -> dict[str, Path]:
    """Generate minimal auditable outputs when the workflow stops early."""
    output_context = coerce_output_context(out_dir)
    root = output_context.ensure_root()
    assets_dir = output_context.ensure_dir("assets")
    tables_dir = output_context.ensure_dir("tables")

    summary = pd.DataFrame([summary_row])
    summary_path = root / "tables" / "summary.csv"
    summary.to_csv(summary_path, index=False)

    table_map: dict[str, list[Path]] = {"global": [summary_path]}
    if extra_tables:
        table_map["global"].extend(extra_tables)

    index_html = render_report(
        out_dir=output_context,
        summary=summary,
        figure_map={},
        table_map=table_map,
        overview_files=overview_files,
    )
    config_json = output_context.write_text(
        "config.json",
        json.dumps({"stop_after": stage, **(config_payload or {})}, indent=2, ensure_ascii=False),
    )
    return {
        "index_html": index_html,
        "summary_csv": summary_path,
        "config_json": config_json,
        "assets_dir": assets_dir,
        "tables_dir": tables_dir,
    }


__all__ = ["write_stage_only_outputs"]
