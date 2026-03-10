"""报告总览层产物构建工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factorlab.reporting.catalog import (
    FigureAttribution,
    normalize_figure_attributions,
    write_artifact_catalog,
    write_figure_attribution_csv,
)
from factorlab.research.insights import build_factor_insights
from factorlab.research.metric_catalog import build_factor_scorecard, build_metric_inventory

_GUIDED_OVERVIEW_FILES = (
    ("data_lineage", "数据血缘与输入指纹"),
    ("factor_definitions", "因子定义与输入契约"),
    ("strategy_definitions", "策略定义与约束"),
)


@dataclass(slots=True)
class ReportOverviewArtifacts:
    """报告总览产物集合。"""

    quick_summary: pd.DataFrame
    metric_col: str
    metric_inventory: pd.DataFrame
    factor_scorecard: pd.DataFrame
    factor_insights: pd.DataFrame
    figure_entries: list[FigureAttribution]
    overview_files: dict[str, Path]
    key_figures: list[Path]
    key_figure_entries: list[FigureAttribution]
    quick_summary_path: Path
    figure_attribution_path: Path
    artifact_catalog_path: Path
    readme_path: Path
    nav_path: Path
    overview_dir: Path


class ReportOverviewBuilder:
    """构建去冗余、可审计的报告总览层。"""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

    def _guided_overview_refs(self, overview_files: dict[str, Path]) -> list[tuple[str, str]]:
        refs: list[tuple[str, str]] = []
        for key, label in _GUIDED_OVERVIEW_FILES:
            path = overview_files.get(key)
            if path is not None:
                refs.append((path.relative_to(self.out_dir).as_posix(), label))
        return refs

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

    def _select_key_figures(
        self,
        *,
        quick_summary: pd.DataFrame,
        metric_col: str,
        figure_entries: list[FigureAttribution],
    ) -> list[FigureAttribution]:
        if quick_summary.empty:
            return []

        figure_lookup: dict[tuple[str | None, str, str], FigureAttribution] = {}
        for entry in figure_entries:
            figure_lookup[(entry.factor, entry.variant, entry.chart)] = entry

        selected: list[FigureAttribution] = []
        chart_order = ["ic.png", "quantile_nav.png"]
        for _, row in quick_summary.head(3).iterrows():
            factor = str(row["factor"])
            variant = str(row.get("variant", "default"))
            metric_value_raw = pd.to_numeric(pd.Series([row.get(metric_col)]), errors="coerce").iloc[0]
            horizon_raw = pd.to_numeric(pd.Series([row.get("horizon")]), errors="coerce").iloc[0]
            for chart in chart_order:
                entry = figure_lookup.get((factor, variant, chart))
                if entry is None:
                    continue
                selected.append(
                    replace(
                        entry,
                        rank=int(row["rank"]),
                        horizon=int(horizon_raw) if pd.notna(horizon_raw) else None,
                        metric_name=metric_col,
                        metric_value=float(metric_value_raw) if pd.notna(metric_value_raw) else None,
                        direction=str(row.get("direction", "")),
                    )
                )
        return selected

    def _write_readme_first(
        self,
        *,
        quick_summary: pd.DataFrame,
        metric_col: str,
        key_figures: list[FigureAttribution],
        insights: pd.DataFrame,
        figure_attribution_path: Path,
        artifact_catalog_path: Path,
        overview_files: dict[str, Path],
    ) -> Path:
        steps = [
            "打开 `index.html`",
            "打开 `overview/README.md`（总览导航）",
            "打开 `tables/overview/quick_summary.csv`（因子排名速览）",
            "打开 `tables/overview/figure_attribution.csv`（图表归因）",
            *[f"打开 `{rel}`（{label}）" for rel, label in self._guided_overview_refs(overview_files)],
            "打开 `artifact_catalog.json`（程序化产物目录）",
        ]
        lines: list[str] = ["# 结果速览（先看这个）", "", "建议顺序：", *[f"{idx}. {step}" for idx, step in enumerate(steps, start=1)], ""]
        if not quick_summary.empty:
            lines.append(f"当前主指标：`{metric_col}`")
            lines.append("")
            lines.append("Top 因子：")
            for _, row in quick_summary.head(5).iterrows():
                lines.append(
                    f"- #{int(row['rank'])} `{row['factor']}` | variant={row['variant']} | "
                    f"horizon={row['horizon']} | {metric_col}={float(row[metric_col]):.4f} | {row['direction']}"
                )
            lines.append("")
        if not insights.empty:
            lines.append("自动解读（模板）：")
            for _, row in insights.head(5).iterrows():
                lines.append(f"- {row['summary_text']}")
                lines.append(f"  建议：{row['action_text']}")
            lines.append("")
        if key_figures:
            lines.append("关键图（引用 canonical 产物，不再复制）：")
            for item in key_figures:
                rel = item.path.relative_to(self.out_dir).as_posix()
                srcs = [p.relative_to(self.out_dir).as_posix() for p in item.source_tables]
                source_text = ", ".join(f"`{src}`" for src in srcs) if srcs else "同目录研究产物"
                rank_text = f"#{item.rank} " if item.rank is not None else ""
                lines.append(
                    f"- {rank_text}`{rel}` | {item.factor}/{item.variant} | {item.label} | 数据来源：{source_text}"
                )
            lines.append("")
        lines.extend(f"{label}：`{rel}`" for rel, label in self._guided_overview_refs(overview_files))
        lines.append(f"图表归因表：`{figure_attribution_path.relative_to(self.out_dir).as_posix()}`")
        lines.append(f"产物目录：`{artifact_catalog_path.relative_to(self.out_dir).as_posix()}`")
        lines.append("完整明细仍在 `assets/detail/` 与 `tables/detail/`。")

        out = self.out_dir / "README_FIRST.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _write_navigation_json(
        self,
        *,
        quick_summary_path: Path,
        readme_path: Path,
        figure_attribution_path: Path,
        artifact_catalog_path: Path,
        key_figures: list[FigureAttribution],
        overview_files: dict[str, Path],
    ) -> Path:
        payload = {
            "entry": "index.html",
            "quick_readme": str(readme_path.relative_to(self.out_dir).as_posix()),
            "overview_readme": "overview/README.md",
            "quick_summary_csv": str(quick_summary_path.relative_to(self.out_dir).as_posix()),
            "figure_attribution_csv": str(figure_attribution_path.relative_to(self.out_dir).as_posix()),
            "artifact_catalog": str(artifact_catalog_path.relative_to(self.out_dir).as_posix()),
            "overview_files": {name: str(path.relative_to(self.out_dir).as_posix()) for name, path in overview_files.items()},
            "key_figures": [item.to_record(self.out_dir) for item in key_figures],
            "overview_root": "overview",
            "overview_tables_root": "tables/overview",
            "full_assets_root": "assets/detail",
            "full_tables_root": "tables/detail",
        }
        out = self.out_dir / "report_navigation.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    def _write_overview_bundle(
        self,
        *,
        overview_files: dict[str, Path],
        key_figures: list[FigureAttribution],
        artifact_catalog_path: Path,
    ) -> Path:
        overview_dir = self.out_dir / "overview"
        overview_dir.mkdir(parents=True, exist_ok=True)

        readme_lines = [
            "# Overview",
            "",
            "该目录只保留导航与 manifest；核心表格统一保存在 `../tables/overview/`，避免重复快照。",
            "",
        ]
        readme_lines.append("建议阅读顺序：")
        ordered_refs = [
            "../tables/overview/quick_summary.csv",
            "../tables/overview/factor_scorecard.csv",
            "../tables/overview/factor_insights.csv",
            *[f"../{rel}" for rel, _ in self._guided_overview_refs(overview_files)],
            "../tables/overview/metric_inventory.csv",
            "../tables/overview/figure_attribution.csv",
            "../artifact_catalog.json",
        ]
        for idx, ref in enumerate(ordered_refs, start=1):
            readme_lines.append(f"{idx}. `{ref}`")
        readme_lines.append("")
        if key_figures:
            readme_lines.append("关键图引用：")
            for item in key_figures:
                rel = item.path.relative_to(self.out_dir).as_posix()
                srcs = [p.relative_to(self.out_dir).as_posix() for p in item.source_tables]
                src_text = ", ".join(f"`../{src}`" for src in srcs) if srcs else "同目录研究产物"
                readme_lines.append(f"- `../{rel}` | {item.factor}/{item.variant} | {item.label} | 来源：{src_text}")
            readme_lines.append("")
        readme_lines.extend(
            [
                "完整明细：",
                "- `../tables/detail/`",
                "- `../assets/detail/`",
                f"- `../{artifact_catalog_path.relative_to(self.out_dir).as_posix()}`",
            ]
        )

        readme_path = overview_dir / "README.md"
        readme_path.write_text("\n".join(readme_lines), encoding="utf-8")

        manifest = {
            "layout_version": 2,
            "overview_dir": "overview",
            "navigation_only": True,
            "deduplicated_tables": True,
            "deduplicated_key_figures": True,
            "files": {name: path.relative_to(self.out_dir).as_posix() for name, path in overview_files.items()},
            "key_figures": [item.to_record(self.out_dir) for item in key_figures],
            "detail_tables_root": "tables/detail",
            "detail_assets_root": "assets/detail",
            "artifact_catalog": artifact_catalog_path.relative_to(self.out_dir).as_posix(),
        }
        manifest_path = overview_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return overview_dir

    def build(
        self,
        *,
        summary: pd.DataFrame,
        figure_map: dict[str, list[Path]],
        table_map: dict[str, list[Path]],
        figure_sources: dict[str | Path, dict[str, Any]] | None = None,
        overview_files: dict[str, Path] | None = None,
    ) -> ReportOverviewArtifacts:
        table_map.setdefault("overview", [])

        metric_inventory = build_metric_inventory(summary)
        factor_scorecard = build_factor_scorecard(summary)
        factor_insights = build_factor_insights(factor_scorecard)
        quick_summary, metric_col = self._build_quick_summary(summary)

        overview_table_dir = self.out_dir / "tables" / "overview"
        overview_table_dir.mkdir(parents=True, exist_ok=True)

        quick_summary_path = overview_table_dir / "quick_summary.csv"
        if not quick_summary.empty:
            quick_summary.to_csv(quick_summary_path, index=False)
            table_map["overview"].append(quick_summary_path)

        metric_inventory_path = overview_table_dir / "metric_inventory.csv"
        metric_inventory.to_csv(metric_inventory_path, index=False)
        table_map["overview"].append(metric_inventory_path)

        factor_scorecard_path = overview_table_dir / "factor_scorecard.csv"
        factor_scorecard.to_csv(factor_scorecard_path, index=False)
        table_map["overview"].append(factor_scorecard_path)

        factor_insights_path = overview_table_dir / "factor_insights.csv"
        factor_insights.to_csv(factor_insights_path, index=False)
        table_map["overview"].append(factor_insights_path)

        figure_entries = normalize_figure_attributions(
            out_dir=self.out_dir,
            figure_map=figure_map,
            table_map=table_map,
            figure_sources=figure_sources,
        )
        figure_attribution_path = overview_table_dir / "figure_attribution.csv"
        write_figure_attribution_csv(
            out_dir=self.out_dir,
            out_path=figure_attribution_path,
            entries=figure_entries,
        )
        table_map["overview"].append(figure_attribution_path)

        key_figure_entries = self._select_key_figures(
            quick_summary=quick_summary,
            metric_col=metric_col,
            figure_entries=figure_entries,
        )
        key_figures = [item.path for item in key_figure_entries]

        merged_overview_files = dict(overview_files or {})
        merged_overview_files.update(
            {
            "quick_summary": quick_summary_path,
            "metric_inventory": metric_inventory_path,
            "factor_scorecard": factor_scorecard_path,
            "factor_insights": factor_insights_path,
            "figure_attribution": figure_attribution_path,
            "readme": self.out_dir / "overview" / "README.md",
            }
        )
        artifact_catalog_path = write_artifact_catalog(
            out_dir=self.out_dir,
            out_path=self.out_dir / "artifact_catalog.json",
            table_map=table_map,
            figure_entries=figure_entries,
            overview_files=merged_overview_files,
            key_figures=key_figure_entries,
        )

        readme_path = self._write_readme_first(
            quick_summary=quick_summary,
            metric_col=metric_col,
            key_figures=key_figure_entries,
            insights=factor_insights,
            figure_attribution_path=figure_attribution_path,
            artifact_catalog_path=artifact_catalog_path,
            overview_files=merged_overview_files,
        )
        nav_path = self._write_navigation_json(
            quick_summary_path=quick_summary_path,
            readme_path=readme_path,
            figure_attribution_path=figure_attribution_path,
            artifact_catalog_path=artifact_catalog_path,
            key_figures=key_figure_entries,
            overview_files=merged_overview_files,
        )
        overview_dir = self._write_overview_bundle(
            overview_files=merged_overview_files,
            key_figures=key_figure_entries,
            artifact_catalog_path=artifact_catalog_path,
        )

        return ReportOverviewArtifacts(
            quick_summary=quick_summary,
            metric_col=metric_col,
            metric_inventory=metric_inventory,
            factor_scorecard=factor_scorecard,
            factor_insights=factor_insights,
            figure_entries=figure_entries,
            overview_files=merged_overview_files,
            key_figures=key_figures,
            key_figure_entries=key_figure_entries,
            quick_summary_path=quick_summary_path,
            figure_attribution_path=figure_attribution_path,
            artifact_catalog_path=artifact_catalog_path,
            readme_path=readme_path,
            nav_path=nav_path,
            overview_dir=overview_dir,
        )
