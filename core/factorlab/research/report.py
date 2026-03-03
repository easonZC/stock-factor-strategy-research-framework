"""因子研究 HTML 报告渲染器。"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factorlab.utils import ensure_within, safe_slug


class ReportRenderer:
    """研究报告渲染器（OOP 接口）。"""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

    def _safe_rel(self, path: Path) -> str:
        safe_path = ensure_within(self.out_dir, path)
        rel = safe_path.relative_to(self.out_dir.resolve()).as_posix()
        return html.escape(rel, quote=True)

    def _pick_primary_metric(self, summary: pd.DataFrame) -> tuple[str, bool]:
        for col in ["rank_ic_mean", "ic_mean"]:
            if col in summary.columns and pd.to_numeric(summary[col], errors="coerce").notna().any():
                return col, True
        for col in ["ls_sharpe", "signal_lag0_ic_mean"]:
            if col in summary.columns and pd.to_numeric(summary[col], errors="coerce").notna().any():
                return col, False
        return "", True

    def _build_quick_summary(self, summary: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        if summary.empty or "factor" not in summary.columns:
            return pd.DataFrame(), ""
        metric_col, use_abs = self._pick_primary_metric(summary)
        if not metric_col:
            return pd.DataFrame(), ""

        work = summary.copy()
        work["factor"] = work["factor"].astype(str)
        if "variant" not in work.columns:
            work["variant"] = "default"
        if "horizon" not in work.columns:
            work["horizon"] = np.nan

        metric_val = pd.to_numeric(work[metric_col], errors="coerce")
        work["_metric_value"] = metric_val
        work["_score"] = metric_val.abs() if use_abs else metric_val
        work = work[np.isfinite(work["_score"])].copy()
        if work.empty:
            return pd.DataFrame(), metric_col

        best_idx = work.groupby("factor")["_score"].idxmax()
        top = work.loc[best_idx].copy()
        top = top.sort_values("_score", ascending=False).reset_index(drop=True)
        top["direction"] = np.where(top["_metric_value"] >= 0, "正向", "反向")
        top["rank"] = np.arange(1, len(top) + 1)

        cols = ["rank", "factor", "variant", "horizon", "_metric_value", "direction"]
        out = top[cols].rename(columns={"_metric_value": metric_col})
        return out, metric_col

    def _find_figure_for_variant(self, paths: list[Path], variant: str, filename: str) -> Path | None:
        variant_token = f"/{str(variant).strip()}/"
        picked = [p for p in paths if p.name == filename and variant_token in p.as_posix()]
        if picked:
            return picked[0]
        fallback = [p for p in paths if p.name == filename]
        return fallback[0] if fallback else None

    def _build_key_figures(
        self,
        quick_summary: pd.DataFrame,
        figure_map: dict[str, list[Path]],
    ) -> list[Path]:
        if quick_summary.empty:
            return []

        key_dir = self.out_dir / "assets" / "key"
        key_dir.mkdir(parents=True, exist_ok=True)
        copied: list[Path] = []
        chart_order = ["ic.png", "quantile_nav.png", "turnover.png", "coverage.png"]

        for _, row in quick_summary.head(3).iterrows():
            factor = str(row["factor"])
            variant = str(row.get("variant", ""))
            paths = figure_map.get(factor, [])
            if not paths:
                continue
            for chart in chart_order[:2]:
                src = self._find_figure_for_variant(paths=paths, variant=variant, filename=chart)
                if src is None or not src.exists():
                    continue
                name = f"{int(row['rank']):02d}_{safe_slug(factor, default='factor')}_{safe_slug(variant, default='variant')}_{chart}"
                dst = key_dir / name
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
                copied.append(dst)
        return copied

    def _render_table_links(self, rows: list[str], table_map: dict[str, list[Path]]) -> None:
        rows.append("<ul>")
        for section, tables in table_map.items():
            rows.append(f"<li><strong>{html.escape(str(section))}</strong><ul>")
            for table_path in tables:
                rel = self._safe_rel(table_path)
                rows.append(f"<li><a href='{rel}'>{rel}</a></li>")
            rows.append("</ul></li>")
        rows.append("</ul>")

    def _render_all_figures(self, rows: list[str], figure_map: dict[str, list[Path]]) -> None:
        for section, figs in figure_map.items():
            rows.append(f"<h3>{html.escape(str(section))}</h3>")
            for fig in figs:
                rel = self._safe_rel(fig)
                rows.append(f"<div><img src='{rel}' style='max-width:980px; width:100%;'/></div>")

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

    def _write_readme_first(
        self,
        quick_summary: pd.DataFrame,
        metric_col: str,
        key_figures: list[Path],
    ) -> Path:
        lines: list[str] = [
            "# 结果速览（先看这个）",
            "",
            "建议顺序：",
            "1. 打开 `index.html`",
            "2. 打开 `tables/quick_summary.csv`（因子排名速览）",
            "3. 打开 `assets/key/`（关键图汇总）",
            "",
        ]
        if not quick_summary.empty:
            lines.append(f"当前主指标：`{metric_col}`")
            lines.append("")
            lines.append("Top 因子：")
            for _, row in quick_summary.head(5).iterrows():
                lines.append(
                    f"- #{int(row['rank'])} `{row['factor']}` | variant={row['variant']} | horizon={row['horizon']} | {metric_col}={float(row[metric_col]):.4f} | {row['direction']}"
                )
            lines.append("")
        if key_figures:
            lines.append("关键图文件：")
            for p in key_figures:
                rel = p.relative_to(self.out_dir).as_posix()
                lines.append(f"- `{rel}`")
            lines.append("")
        lines.append("完整明细仍在 `assets/factors/` 与 `tables/factors/`。")

        out = self.out_dir / "README_FIRST.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _write_navigation_json(
        self,
        quick_summary_path: Path,
        readme_path: Path,
        key_figures: list[Path],
    ) -> Path:
        payload: dict[str, Any] = {
            "entry": "index.html",
            "quick_readme": str(readme_path.relative_to(self.out_dir).as_posix()),
            "quick_summary_csv": str(quick_summary_path.relative_to(self.out_dir).as_posix()),
            "key_figures": [str(p.relative_to(self.out_dir).as_posix()) for p in key_figures],
            "full_assets_root": "assets/factors",
            "full_tables_root": "tables/factors",
        }
        out = self.out_dir / "report_navigation.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    def render(
        self,
        summary: pd.DataFrame,
        figure_map: dict[str, list[Path]],
        table_map: dict[str, list[Path]],
    ) -> Path:
        table_map = {k: list(v) for k, v in table_map.items()}
        table_map.setdefault("global", [])

        quick_summary, metric_col = self._build_quick_summary(summary)
        quick_summary_path = self.out_dir / "tables" / "quick_summary.csv"
        if not quick_summary.empty:
            quick_summary_path.parent.mkdir(parents=True, exist_ok=True)
            quick_summary.to_csv(quick_summary_path, index=False)
            table_map["global"].insert(0, quick_summary_path)

        key_figures = self._build_key_figures(quick_summary=quick_summary, figure_map=figure_map)
        readme_path = self._write_readme_first(
            quick_summary=quick_summary,
            metric_col=metric_col,
            key_figures=key_figures,
        )
        nav_path = self._write_navigation_json(
            quick_summary_path=quick_summary_path,
            readme_path=readme_path,
            key_figures=key_figures,
        )

        rows: list[str] = []
        rows.append("<h1>Stock Factor Research Report</h1>")
        rows.append(
            "<p>Generated by FactorLab pipeline. Includes IC, quantiles, turnover, diagnostics and significance.</p>"
        )

        rows.append("<h2>快速开始（30 秒）</h2>")
        rows.append("<ol>")
        rows.append("<li>先看 <a href='README_FIRST.md'>README_FIRST.md</a>（如何解读）</li>")
        if quick_summary_path.exists():
            rows.append(f"<li>看 <a href='{self._safe_rel(quick_summary_path)}'>quick_summary.csv</a>（最关键排名）</li>")
        rows.append(f"<li>再看 <a href='{self._safe_rel(nav_path)}'>report_navigation.json</a>（程序化导航）</li>")
        rows.append("</ol>")

        if not quick_summary.empty:
            rows.append("<h2>Top 因子速览</h2>")
            rows.append(quick_summary.head(10).to_html(index=False, float_format=lambda x: f"{x:.4f}"))

        if key_figures:
            rows.append("<h2>关键图（优先看）</h2>")
            for fig in key_figures:
                rel = self._safe_rel(fig)
                rows.append(f"<div><img src='{rel}' style='max-width:980px; width:100%;'/></div>")

        rows.append("<details>")
        rows.append("<summary>查看完整 Summary 表</summary>")
        rows.append(summary.to_html(index=False, float_format=lambda x: f"{x:.4f}"))
        rows.append("</details>")

        rows.append("<details>")
        rows.append("<summary>查看全部图表</summary>")
        self._render_all_figures(rows, figure_map=figure_map)
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
                "<head><meta charset='utf-8'><title>FactorLab Report</title></head>",
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
) -> Path:
    """兼容函数式调用，内部委托给 OOP 报告渲染器。"""
    return ReportRenderer(out_dir=out_dir).render(summary=summary, figure_map=figure_map, table_map=table_map)
