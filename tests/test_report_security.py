"""模块说明。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from factorlab.research.report import ReportRenderer


def _make_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "factor": ["<script>alert('x')</script>"],
            "rank_ic_mean": [0.1234],
        }
    )


def test_report_renderer_escapes_html_sections(tmp_path: Path) -> None:
    out_dir = tmp_path / "report"
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    fig = out_dir / "assets" / "plot.png"
    tbl = out_dir / "tables" / "summary.csv"
    fig.write_bytes(b"png")
    tbl.write_text("a,b\n1,2\n", encoding="utf-8")

    html_path = ReportRenderer(out_dir=out_dir).render(
        summary=_make_summary(),
        figure_map={"<img src=x onerror=alert(1)>": [fig]},
        table_map={"<b>unsafe</b>": [tbl]},
    )
    text = html_path.read_text(encoding="utf-8")
    assert "<img src=x onerror=alert(1)>" not in text
    assert "&lt;img src=x onerror=alert(1)&gt;" in text
    assert "<script>alert('x')</script>" not in text
    assert "&lt;script&gt;alert('x')&lt;/script&gt;" in text


def test_report_renderer_blocks_external_paths(tmp_path: Path) -> None:
    out_dir = tmp_path / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"png")
    with pytest.raises(ValueError, match="Unsafe output path"):
        ReportRenderer(out_dir=out_dir).render(
            summary=_make_summary(),
            figure_map={"sec": [outside]},
            table_map={},
        )
