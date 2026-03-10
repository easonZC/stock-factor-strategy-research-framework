"""报告产物目录与图表归因工具。"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import pandas as pd

from factorlab.utils import ensure_within, safe_slug


_CHART_META: dict[str, tuple[str, str, list[str]]] = {
    "coverage.png": (
        "Coverage",
        "Coverage monitor generated from the factor coverage table for the same run.",
        ["coverage.csv"],
    ),
    "fama_macbeth_beta.png": (
        "Fama-MacBeth Beta",
        "Cross-sectional regression summary generated from same-run Fama-MacBeth tables.",
        ["fama_macbeth_summary.csv", "fama_macbeth_daily.csv"],
    ),
    "factor_corr_spearman.png": (
        "Factor Correlation",
        "Global factor correlation heatmap generated from the same-run correlation tables.",
        ["factor_corr_spearman.csv", "factor_corr_pearson.csv"],
    ),
    "ic.png": (
        "IC Series",
        "Primary IC / RankIC series generated from same-run IC tables.",
        ["ic_daily_h*.csv"],
    ),
    "ic_decay.png": (
        "IC Decay",
        "IC decay profile generated from same-run IC summary tables.",
        ["ic_decay.csv", "ic_daily_h*.csv"],
    ),
    "industry_decomposition.png": (
        "Industry Decomposition",
        "Industry long-short decomposition generated from same-run industry tables.",
        ["industry_decomposition_summary.csv", "industry_decomposition_daily.csv"],
    ),
    "outlier_before_after.png": (
        "Outlier Monitoring",
        "Global outlier monitoring figure generated from same-run preprocessing audit tables.",
        ["outlier_before_after.csv"],
    ),
    "quantile_nav.png": (
        "Quantile NAV",
        "Quantile NAV figure generated from same-run quantile return tables.",
        ["quantile_nav.csv", "quantile_profile.csv", "quantile_daily.csv"],
    ),
    "signal_lag_ic.png": (
        "Signal Lag IC",
        "Signal lag IC figure generated from same-run lag evaluation tables.",
        ["signal_lag_ic.csv"],
    ),
    "stability.png": (
        "Stability",
        "Factor stability figure generated from same-run stability tables.",
        ["stability.csv"],
    ),
    "style_decomposition.png": (
        "Style Decomposition",
        "Style long-short decomposition generated from same-run style tables.",
        ["style_decomposition_summary.csv", "style_decomposition_daily.csv"],
    ),
    "turnover.png": (
        "Turnover",
        "Turnover figure generated from same-run turnover tables.",
        ["turnover.csv"],
    ),
}


@dataclass(slots=True)
class FigureAttribution:
    """图表归因元数据。"""

    path: Path
    factor: str | None
    variant: str
    chart: str
    label: str
    description: str
    source_tables: list[Path]
    scope: str = "detail"
    rank: int | None = None
    horizon: int | None = None
    metric_name: str | None = None
    metric_value: float | None = None
    direction: str | None = None

    def to_record(self, out_dir: Path) -> dict[str, Any]:
        """转换为可序列化字典。"""
        artifact_id_base = "_".join(
            [self.factor or "global", self.variant or "global", Path(self.chart).stem or "figure"]
        )
        return {
            "artifact_id": safe_slug(artifact_id_base, default="figure", max_len=120),
            "path": _relative_path(out_dir, self.path),
            "factor": self.factor,
            "variant": self.variant,
            "chart": self.chart,
            "label": self.label,
            "description": self.description,
            "source_tables": [_relative_path(out_dir, p) for p in self.source_tables],
            "scope": self.scope,
            "rank": self.rank,
            "horizon": self.horizon,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "direction": self.direction,
        }


def _relative_path(out_dir: Path, path: Path) -> str:
    safe_path = ensure_within(out_dir, path)
    return safe_path.relative_to(Path(out_dir).resolve()).as_posix()


def _normalize_unique_paths(paths: list[Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(Path(path).resolve())] = Path(path)
    return sorted(unique.values(), key=lambda item: item.as_posix())


def _infer_variant_from_path(path: Path) -> tuple[str, str]:
    parent = path.parent.name
    if "__" not in parent:
        return "global", "global"
    factor_slug, variant = parent.rsplit("__", 1)
    return factor_slug, variant or "default"


def _build_table_lookup(table_map: dict[str, list[Path]]) -> dict[tuple[str, str], list[Path]]:
    lookup: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for section, paths in table_map.items():
        if section == "global":
            lookup[(section, "global")].extend(_normalize_unique_paths(paths))
            continue
        for path in _normalize_unique_paths(paths):
            _, variant = _infer_variant_from_path(path)
            lookup[(section, variant)].append(path)
    return lookup


def _select_source_tables(chart: str, candidate_tables: list[Path]) -> list[Path]:
    meta = _CHART_META.get(chart)
    if meta is None:
        return []
    patterns = meta[2]
    picked = [
        table_path
        for table_path in _normalize_unique_paths(candidate_tables)
        for pattern in patterns
        if fnmatch(table_path.name, pattern)
    ]
    return _normalize_unique_paths(picked)


def _fallback_figure_attribution(
    *,
    out_dir: Path,
    figure_path: Path,
    section: str,
    table_lookup: dict[tuple[str, str], list[Path]],
) -> FigureAttribution:
    factor_slug, variant = _infer_variant_from_path(figure_path)
    factor = None if section == "global" else section
    chart = figure_path.name
    label, description, _ = _CHART_META.get(
        chart,
        (
            Path(chart).stem.replace("_", " ").title(),
            "Figure generated by FactorLab from same-run research artifacts.",
            [],
        ),
    )
    candidates = table_lookup.get((section, variant), [])
    if not candidates and factor is not None:
        candidates = table_lookup.get((section, "global"), [])
    if not candidates and section == "global":
        candidates = table_lookup.get(("global", "global"), [])
    source_tables = _select_source_tables(chart, candidates)
    scope = "global" if section == "global" or variant == "global" else "detail"
    if factor is None and factor_slug not in {"global", ""}:
        factor = factor_slug
    return FigureAttribution(
        path=ensure_within(out_dir, figure_path),
        factor=factor,
        variant=variant,
        chart=chart,
        label=label,
        description=description,
        source_tables=source_tables,
        scope=scope,
    )


def normalize_figure_attributions(
    *,
    out_dir: Path,
    figure_map: dict[str, list[Path]],
    table_map: dict[str, list[Path]],
    figure_sources: dict[str | Path, dict[str, Any]] | None = None,
) -> list[FigureAttribution]:
    """标准化图表归因，优先使用流水线显式提供的 provenance。"""
    lookup = _build_table_lookup(table_map)
    provenance_lookup: dict[Path, dict[str, Any]] = {}
    for key, value in (figure_sources or {}).items():
        provenance_lookup[Path(key).resolve()] = dict(value)

    entries: list[FigureAttribution] = []
    seen: set[str] = set()

    for section, figure_paths in figure_map.items():
        for figure_path in _normalize_unique_paths(figure_paths):
            resolved = ensure_within(out_dir, figure_path)
            resolved_key = str(resolved.resolve())
            if resolved_key in seen:
                continue
            seen.add(resolved_key)

            meta = provenance_lookup.get(resolved.resolve())
            if meta is None:
                entries.append(
                    _fallback_figure_attribution(
                        out_dir=out_dir,
                        figure_path=resolved,
                        section=section,
                        table_lookup=lookup,
                    )
                )
                continue

            raw_source_tables = [ensure_within(out_dir, Path(p)) for p in meta.get("source_tables", [])]
            entries.append(
                FigureAttribution(
                    path=resolved,
                    factor=meta.get("factor"),
                    variant=str(meta.get("variant", "global")),
                    chart=str(meta.get("chart", resolved.name)),
                    label=str(meta.get("label", _CHART_META.get(resolved.name, (resolved.stem, "", []))[0])),
                    description=str(
                        meta.get(
                            "description",
                            _CHART_META.get(resolved.name, (resolved.stem, "", []))[1]
                            or "Figure generated by FactorLab from same-run research artifacts.",
                        )
                    ),
                    source_tables=_normalize_unique_paths(raw_source_tables),
                    scope=str(meta.get("scope", "detail")),
                )
            )
    return sorted(entries, key=lambda item: item.to_record(out_dir)["path"])


def write_figure_attribution_csv(
    *,
    out_dir: Path,
    out_path: Path,
    entries: list[FigureAttribution],
) -> Path:
    """写出图表归因表。"""
    records = []
    for entry in entries:
        record = entry.to_record(out_dir)
        record["source_tables"] = "; ".join(record["source_tables"])
        records.append(record)
    frame = pd.DataFrame(
        records,
        columns=[
            "artifact_id",
            "path",
            "factor",
            "variant",
            "chart",
            "label",
            "description",
            "source_tables",
            "scope",
            "rank",
            "horizon",
            "metric_name",
            "metric_value",
            "direction",
        ],
    )
    out_path.write_text(frame.to_csv(index=False), encoding="utf-8")
    return out_path


def write_artifact_catalog(
    *,
    out_dir: Path,
    out_path: Path,
    table_map: dict[str, list[Path]],
    figure_entries: list[FigureAttribution],
    overview_files: dict[str, Path],
    key_figures: list[FigureAttribution],
) -> Path:
    """写出运行级产物目录。"""
    table_records: list[dict[str, Any]] = []
    seen_tables: set[str] = set()
    for section, paths in sorted(table_map.items(), key=lambda item: item[0]):
        for table_path in _normalize_unique_paths(paths):
            rel = _relative_path(out_dir, table_path)
            if rel in seen_tables:
                continue
            seen_tables.add(rel)
            factor = None if section in {"global", "overview"} else section
            _, variant = _infer_variant_from_path(table_path)
            role = "overview" if section == "overview" else ("global" if section == "global" else "detail")
            table_records.append(
                {
                    "artifact_id": safe_slug(rel.replace("/", "_"), default="table", max_len=120),
                    "kind": "table",
                    "section": section,
                    "role": role,
                    "factor": factor,
                    "variant": variant if role == "detail" else None,
                    "path": rel,
                }
            )

    payload = {
        "layout_version": 2,
        "deduplicated_overview": True,
        "key_figures_are_references": True,
        "overview_files": {name: _relative_path(out_dir, path) for name, path in overview_files.items()},
        "tables": table_records,
        "figures": [entry.to_record(out_dir) for entry in figure_entries],
        "key_figures": [entry.to_record(out_dir) for entry in key_figures],
        "counts": {
            "tables": len(table_records),
            "figures": len(figure_entries),
            "key_figures": len(key_figures),
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path

