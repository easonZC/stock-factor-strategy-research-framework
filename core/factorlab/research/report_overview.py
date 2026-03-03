"""报告总览层产物构建工具。"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from factorlab.research.insights import build_factor_insights
from factorlab.research.metric_catalog import build_factor_scorecard, build_metric_inventory
from factorlab.utils import safe_slug


@dataclass(slots=True)
class ReportOverviewArtifacts:
    """报告总览产物集合。"""

    quick_summary: pd.DataFrame
    metric_col: str
    metric_inventory: pd.DataFrame
    factor_scorecard: pd.DataFrame
    factor_insights: pd.DataFrame
    key_figures: list[Path]
    quick_summary_path: Path
    readme_path: Path
    nav_path: Path
    overview_dir: Path


class ReportOverviewBuilder:
    """构建报告总览层（overview/key/quick）。"""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir

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
        variant_key = str(variant).strip().lower()
        picked: list[Path] = []
        for p in paths:
            if p.name != filename:
                continue
            posix = p.as_posix().lower()
            parent_name = p.parent.name.lower()
            if f"/{variant_key}/" in posix or parent_name.endswith(f"__{variant_key}"):
                picked.append(p)
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
                name = (
                    f"{int(row['rank']):02d}_"
                    f"{safe_slug(factor, default='factor')}_"
                    f"{safe_slug(variant, default='variant')}_{chart}"
                )
                dst = key_dir / name
                if src.resolve() != dst.resolve():
                    shutil.copy2(src, dst)
                copied.append(dst)
        return copied

    def _write_readme_first(
        self,
        quick_summary: pd.DataFrame,
        metric_col: str,
        key_figures: list[Path],
        insights: pd.DataFrame,
    ) -> Path:
        lines: list[str] = [
            "# 结果速览（先看这个）",
            "",
            "建议顺序：",
            "1. 打开 `index.html`",
            "2. 打开 `overview/`（一站式入口）",
            "3. 打开 `tables/quick_summary.csv`（因子排名速览）",
            "4. 打开 `assets/key/`（关键图汇总）",
            "",
        ]
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
            lines.append("关键图文件：")
            for p in key_figures:
                rel = p.relative_to(self.out_dir).as_posix()
                lines.append(f"- `{rel}`")
            lines.append("")
        lines.append("完整明细仍在 `assets/detail/` 与 `tables/detail/`。")

        out = self.out_dir / "README_FIRST.md"
        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def _write_navigation_json(
        self,
        quick_summary_path: Path,
        readme_path: Path,
        key_figures: list[Path],
    ) -> Path:
        payload = {
            "entry": "index.html",
            "quick_readme": str(readme_path.relative_to(self.out_dir).as_posix()),
            "quick_summary_csv": str(quick_summary_path.relative_to(self.out_dir).as_posix()),
            "key_figures": [str(p.relative_to(self.out_dir).as_posix()) for p in key_figures],
            "overview_root": "overview",
            "full_assets_root": "assets/detail",
            "full_tables_root": "tables/detail",
        }
        out = self.out_dir / "report_navigation.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    def _write_overview_bundle(
        self,
        quick_summary: pd.DataFrame,
        scorecard: pd.DataFrame,
        insights: pd.DataFrame,
        metric_inventory: pd.DataFrame,
        key_figures: list[Path],
    ) -> Path:
        overview_dir = self.out_dir / "overview"
        overview_dir.mkdir(parents=True, exist_ok=True)

        files_map: dict[str, str] = {}
        if not quick_summary.empty:
            p = overview_dir / "top_factors.csv"
            quick_summary.to_csv(p, index=False)
            files_map["top_factors"] = p.name
        if not scorecard.empty:
            p = overview_dir / "factor_scorecard.csv"
            scorecard.to_csv(p, index=False)
            files_map["factor_scorecard"] = p.name
        if not insights.empty:
            p = overview_dir / "factor_insights.csv"
            insights.to_csv(p, index=False)
            files_map["factor_insights"] = p.name
        if not metric_inventory.empty:
            p = overview_dir / "metric_inventory.csv"
            metric_inventory.to_csv(p, index=False)
            files_map["metric_inventory"] = p.name

        rel_key_figs = [p.relative_to(self.out_dir).as_posix() for p in key_figures]
        readme_lines = [
            "# Overview",
            "",
            "这是最简入口目录。建议按以下顺序查看：",
            "1. factor_insights.csv（自动结论模板）",
            "2. factor_scorecard.csv（核心评分）",
            "3. top_factors.csv（Top 因子）",
            "4. ../assets/key/（关键图）",
            "",
        ]
        if rel_key_figs:
            readme_lines.append("关键图：")
            for rel in rel_key_figs:
                readme_lines.append(f"- `../{rel}`")
            readme_lines.append("")
        readme_lines.append("完整明细：")
        readme_lines.append("- `../tables/detail/`")
        readme_lines.append("- `../assets/detail/`")

        readme_path = overview_dir / "README.md"
        readme_path.write_text("\n".join(readme_lines), encoding="utf-8")
        files_map["readme"] = readme_path.name

        manifest = {
            "overview_dir": "overview",
            "files": files_map,
            "key_figures": rel_key_figs,
            "detail_tables_root": "tables/detail",
            "detail_assets_root": "assets/detail",
        }
        manifest_path = overview_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return overview_dir

    def build(
        self,
        summary: pd.DataFrame,
        figure_map: dict[str, list[Path]],
        table_map: dict[str, list[Path]],
    ) -> ReportOverviewArtifacts:
        table_map.setdefault("global", [])

        metric_inventory = build_metric_inventory(summary)
        factor_scorecard = build_factor_scorecard(summary)
        factor_insights = build_factor_insights(factor_scorecard)
        quick_summary, metric_col = self._build_quick_summary(summary)

        overview_table_dir = self.out_dir / "tables" / "overview"
        overview_table_dir.mkdir(parents=True, exist_ok=True)

        quick_summary_path = self.out_dir / "tables" / "quick_summary.csv"
        if not quick_summary.empty:
            quick_summary.to_csv(quick_summary_path, index=False)
            table_map["global"].insert(0, quick_summary_path)

        metric_inventory_path = overview_table_dir / "metric_inventory.csv"
        metric_inventory.to_csv(metric_inventory_path, index=False)
        table_map["global"].append(metric_inventory_path)

        factor_scorecard_path = overview_table_dir / "factor_scorecard.csv"
        factor_scorecard.to_csv(factor_scorecard_path, index=False)
        table_map["global"].append(factor_scorecard_path)

        factor_insights_path = overview_table_dir / "factor_insights.csv"
        factor_insights.to_csv(factor_insights_path, index=False)
        table_map["global"].append(factor_insights_path)

        key_figures = self._build_key_figures(quick_summary=quick_summary, figure_map=figure_map)
        readme_path = self._write_readme_first(
            quick_summary=quick_summary,
            metric_col=metric_col,
            key_figures=key_figures,
            insights=factor_insights,
        )
        nav_path = self._write_navigation_json(
            quick_summary_path=quick_summary_path,
            readme_path=readme_path,
            key_figures=key_figures,
        )
        overview_dir = self._write_overview_bundle(
            quick_summary=quick_summary,
            scorecard=factor_scorecard,
            insights=factor_insights,
            metric_inventory=metric_inventory,
            key_figures=key_figures,
        )

        return ReportOverviewArtifacts(
            quick_summary=quick_summary,
            metric_col=metric_col,
            metric_inventory=metric_inventory,
            factor_scorecard=factor_scorecard,
            factor_insights=factor_insights,
            key_figures=key_figures,
            quick_summary_path=quick_summary_path,
            readme_path=readme_path,
            nav_path=nav_path,
            overview_dir=overview_dir,
        )

