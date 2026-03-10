"""因子研究 HTML 报告渲染器。"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from factorlab.reporting.catalog import FigureAttribution
from factorlab.reporting.overview import ReportOverviewBuilder
from factorlab.utils import ensure_within

_OVERVIEW_LINKS = (
    ("data_lineage", "data_lineage.json", "数据血缘与输入指纹"),
    ("factor_definitions", "factor_definitions.csv", "因子定义与输入契约"),
    ("strategy_definitions", "strategy_definitions.csv", "策略定义与约束"),
)


class ReportRenderer:
    """研究报告渲染器（OOP 接口）。"""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

    def _safe_rel(self, path: Path) -> str:
        safe_path = ensure_within(self.out_dir, path)
        rel = safe_path.relative_to(self.out_dir.resolve()).as_posix()
        return html.escape(rel, quote=True)

    def _render_table_links(self, rows: list[str], table_map: dict[str, list[Path]]) -> None:
        rows.append("<ul>")
        for section, tables in table_map.items():
            rows.append(f"<li><strong>{html.escape(str(section))}</strong><ul>")
            for table_path in tables:
                rel = self._safe_rel(table_path)
                rows.append(f"<li><a href='{rel}'>{rel}</a></li>")
            rows.append("</ul></li>")
        rows.append("</ul>")

    def _render_all_figures(
        self,
        rows: list[str],
        figure_map: dict[str, list[Path]],
        figure_lookup: dict[str, FigureAttribution],
    ) -> None:
        for section, figs in figure_map.items():
            rows.append(f"<h3>{html.escape(str(section))}</h3>")
            for fig in figs:
                rel = self._safe_rel(fig)
                rows.append("<div class='figure-card'>")
                rows.append(f"<img src='{rel}' alt='{rel}'/>")
                entry = figure_lookup.get(rel)
                if entry is not None:
                    rows.append(self._render_figure_caption(entry))
                else:
                    rows.append(f"<p class='figure-caption'><code>{rel}</code></p>")
                rows.append("</div>")

    def _render_figure_caption(self, entry: FigureAttribution) -> str:
        parts = [html.escape(entry.label)]
        if entry.factor:
            parts.append(html.escape(entry.factor))
        if entry.variant and entry.variant != "global":
            parts.append(html.escape(entry.variant))
        if entry.rank is not None:
            parts.append(f"Top #{entry.rank}")
        if entry.metric_name and entry.metric_value is not None:
            parts.append(f"{html.escape(entry.metric_name)}={entry.metric_value:.4f}")
        if entry.direction:
            parts.append(html.escape(entry.direction))

        if entry.source_tables:
            source_links = ", ".join(
                f"<a href='{self._safe_rel(path)}'>{self._safe_rel(path)}</a>" for path in entry.source_tables
            )
        else:
            source_links = "same-run research artifacts"
        desc = html.escape(entry.description)
        return (
            "<p class='figure-caption'>"
            f"<strong>{' · '.join(parts)}</strong><br/>"
            f"{desc}<br/>"
            f"Sources: {source_links}"
            "</p>"
        )

    def _render_key_figures(self, rows: list[str], entries: list[FigureAttribution]) -> None:
        if not entries:
            return
        rows.append("<h2>关键图（引用 canonical 产物）</h2>")
        rows.append(
            "<p>关键图不再复制到单独目录；这里直接引用明细图，并补充可追溯的数据来源。</p>"
        )
        for entry in entries:
            rel = self._safe_rel(entry.path)
            rows.append("<div class='figure-card'>")
            rows.append(f"<img src='{rel}' alt='{rel}'/>")
            rows.append(self._render_figure_caption(entry))
            rows.append("</div>")

    def _render_adapter_audit(self, rows: list[str]) -> None:
        audit_dir = self.out_dir / "tables" / "data"
        if not audit_dir.exists():
            return
        audit_files = sorted(audit_dir.glob("*.csv"))
        if not audit_files:
            return
        rows.append("<h2>Data Adapter Audit / 数据适配器审计</h2>")
        rows.append("<ul>")
        for table_path in audit_files:
            rel = self._safe_rel(table_path)
            rows.append(f"<li><a href='{rel}'>{rel}</a></li>")
        rows.append("</ul>")

        summary_path = audit_dir / "adapter_quality_audit.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            rows.append("<h3>审计预览</h3>")
            rows.append(summary.to_html(index=False))

    def _iter_overview_links(self, overview_files: dict[str, Path]) -> list[tuple[Path, str, str]]:
        links: list[tuple[Path, str, str]] = []
        for key, filename, label in _OVERVIEW_LINKS:
            path = overview_files.get(key)
            if path is not None and path.exists():
                links.append((path, filename, label))
        return links

    def _render_table_preview(
        self,
        rows: list[str],
        *,
        path: Path | None,
        title: str,
        description: str,
        preview_cols: list[str],
    ) -> None:
        if path is None or not path.exists():
            return
        frame = pd.read_csv(path)
        if frame.empty:
            return
        rows.append(f"<h2>{title}</h2>")
        rows.append(f"<p>{description}</p>")
        cols = [col for col in preview_cols if col in frame.columns] or list(frame.columns[:5])
        rows.append(frame[cols].head(10).to_html(index=False))

    def _render_data_lineage(self, rows: list[str], overview_files: dict[str, Path]) -> None:
        path = overview_files.get("data_lineage")
        if path is None or not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = payload.get("source", {})
        profile = payload.get("panel_profile", {})
        rows.append("<h2>数据血缘与输入指纹</h2>")
        rows.append("<p>记录本次运行的数据入口、面板规模与稳定指纹，便于复现与排查差异。</p>")
        rows.append(
            pd.DataFrame(
                [
                    {
                        "adapter": source.get("adapter"),
                        "mode": source.get("mode"),
                        "rows": profile.get("rows"),
                        "assets": profile.get("assets"),
                        "dates": profile.get("dates"),
                        "date_min": profile.get("date_min"),
                        "date_max": profile.get("date_max"),
                        "fingerprint": payload.get("fingerprint"),
                    }
                ]
            ).to_html(index=False)
        )

    def _render_governance_overview(self, rows: list[str], overview_files: dict[str, Path]) -> None:
        self._render_data_lineage(rows, overview_files)
        self._render_table_preview(
            rows,
            path=overview_files.get("factor_definitions"),
            title="因子定义与实现",
            description="该表展示当前运行实际使用的因子定义、输入列要求、公式说明与实现位置，便于审计和复现。",
            preview_cols=["name", "family", "required_columns", "description", "formula"],
        )
        self._render_table_preview(
            rows,
            path=overview_files.get("strategy_definitions"),
            title="策略定义与约束",
            description="该表展示当前运行实际使用的策略定义、组合约束、参数摘要与实现位置，便于回测治理。",
            preview_cols=["name", "family", "constraints", "description", "parameters_json"],
        )

    def render(
        self,
        summary: pd.DataFrame,
        figure_map: dict[str, list[Path]],
        table_map: dict[str, list[Path]],
        figure_sources: dict[str | Path, dict[str, Any]] | None = None,
        overview_files: dict[str, Path] | None = None,
    ) -> Path:
        table_map = {k: list(v) for k, v in table_map.items()}
        table_map.setdefault("global", [])
        overview = ReportOverviewBuilder(self.out_dir).build(
            summary=summary,
            figure_map=figure_map,
            table_map=table_map,
            figure_sources=figure_sources,
            overview_files=overview_files,
        )
        figure_lookup = {
            self._safe_rel(entry.path): entry
            for entry in overview.figure_entries
        }

        rows: list[str] = []
        rows.append("<h1>Stock Factor Research Report</h1>")
        rows.append(
            "<p>Generated by FactorLab pipeline. Canonical tables live under <code>tables/overview</code> and "
            "detail figures keep explicit same-run source attribution.</p>"
        )

        rows.append("<h2>快速开始（30 秒）</h2>")
        rows.append("<ol>")
        rows.append("<li>先看 <a href='README_FIRST.md'>README_FIRST.md</a>（如何解读）</li>")
        rows.append(
            f"<li>进入 <a href='{self._safe_rel(overview.overview_dir / 'README.md')}'>overview/README.md</a>（一站式入口）</li>"
        )
        rows.append(
            f"<li>看 <a href='{self._safe_rel(overview.quick_summary_path)}'>quick_summary.csv</a>（最关键排名）</li>"
        )
        rows.append(
            f"<li>看 <a href='{self._safe_rel(overview.figure_attribution_path)}'>figure_attribution.csv</a>（图表归因）</li>"
        )
        for path, filename, label in self._iter_overview_links(overview.overview_files):
            rows.append(
                f"<li>看 <a href='{self._safe_rel(path)}'>{filename}</a>（{label}）</li>"
            )
        rows.append(
            f"<li>再看 <a href='{self._safe_rel(overview.artifact_catalog_path)}'>artifact_catalog.json</a>（程序化导航）</li>"
        )
        rows.append("</ol>")

        if not overview.quick_summary.empty:
            rows.append("<h2>Top 因子速览</h2>")
            rows.append(overview.quick_summary.head(10).to_html(index=False, float_format=lambda x: f"{x:.4f}"))

        if not overview.factor_scorecard.empty:
            rows.append("<h2>核心评分卡（建议优先）</h2>")
            rows.append("<p>仅显示用于决策的核心指标组合，诊断类指标仍保留在明细表中。</p>")
            rows.append(overview.factor_scorecard.head(10).to_html(index=False, float_format=lambda x: f"{x:.4f}"))

        if not overview.factor_insights.empty:
            rows.append("<h2>自动解读模板（可直接读结论）</h2>")
            rows.append(
                overview.factor_insights[
                    ["rank", "factor", "strength", "confidence", "risk", "summary_text", "action_text"]
                ]
                .head(10)
                .to_html(index=False)
            )

        if not overview.metric_inventory.empty:
            rows.append("<h2>指标分层（Core / Diagnostic）</h2>")
            rows.append(overview.metric_inventory.to_html(index=False, float_format=lambda x: f"{x:.4f}"))

        self._render_governance_overview(rows, overview.overview_files)
        self._render_key_figures(rows, overview.key_figure_entries)

        rows.append("<details>")
        rows.append("<summary>查看完整 Summary 表</summary>")
        rows.append(summary.to_html(index=False, float_format=lambda x: f"{x:.4f}"))
        rows.append("</details>")

        rows.append("<details>")
        rows.append("<summary>查看全部图表</summary>")
        self._render_all_figures(rows, figure_map=figure_map, figure_lookup=figure_lookup)
        rows.append("</details>")

        rows.append("<details>")
        rows.append("<summary>查看全部数据表</summary>")
        self._render_table_links(rows, table_map=table_map)
        rows.append("</details>")

        self._render_adapter_audit(rows)

        html_text = "\n".join(
            [
                "<!doctype html>",
                "<html>",
                (
                    "<head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
                    "<title>FactorLab Report</title>"
                    "<style>"
                    "body{font-family:Arial,sans-serif;max-width:1200px;margin:0 auto;padding:24px;line-height:1.5;color:#17202a;}"
                    "table{border-collapse:collapse;width:100%;margin:12px 0 20px;}"
                    "th,td{border:1px solid #d5d8dc;padding:6px 8px;text-align:left;font-size:13px;}"
                    "th{background:#f4f6f7;}"
                    "details{margin:16px 0;padding:12px 14px;border:1px solid #d5d8dc;border-radius:8px;background:#fbfcfc;}"
                    ".figure-card{margin:18px 0;padding:14px;border:1px solid #d5d8dc;border-radius:10px;background:#ffffff;}"
                    ".figure-card img{max-width:980px;width:100%;display:block;margin:0 auto;}"
                    ".figure-caption{font-size:13px;color:#34495e;margin:12px 0 0;}"
                    "code{background:#f4f6f7;padding:1px 4px;border-radius:4px;}"
                    "a{color:#1f618d;text-decoration:none;}a:hover{text-decoration:underline;}"
                    "</style></head>"
                ),
                "<body>",
                *rows,
                "</body></html>",
            ]
        )
        out = self.out_dir / "index.html"
        out.write_text(html_text, encoding="utf-8")
        return out


def render_report(
    out_dir: Path,
    summary: pd.DataFrame,
    figure_map: dict[str, list[Path]],
    table_map: dict[str, list[Path]],
    figure_sources: dict[str | Path, dict[str, Any]] | None = None,
    overview_files: dict[str, Path] | None = None,
) -> Path:
    """兼容函数式调用，内部委托给 OOP 报告渲染器。"""
    return ReportRenderer(out_dir=out_dir).render(
        summary=summary,
        figure_map=figure_map,
        table_map=table_map,
        figure_sources=figure_sources,
        overview_files=overview_files,
    )
