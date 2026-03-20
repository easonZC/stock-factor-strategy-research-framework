"""Data adapter registry, validation, loading, and quality-audit stages."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from factorlab.config import AdapterConfig, SyntheticConfig
from factorlab.data import (
    PanelSanitizationConfig,
    build_data_adapter_registry,
    build_data_adapter_validator_registry,
    generate_synthetic_panel,
    read_panel,
    read_panel_directory,
)
from factorlab.utils import get_logger, timed_stage
from factorlab.workflows.config_normalization import as_list as _as_list
from factorlab.workflows.config_normalization import to_int as _to_int


LOGGER = get_logger("factorlab.workflows.config_runner")


@dataclass(slots=True)
class DataAdapterWorkflowResult:
    """Data adapter stage outputs."""

    panel: pd.DataFrame
    load_report: dict[str, Any]
    mode_report: dict[str, Any]
    adapter_validation_report: dict[str, Any]
    adapter_audit_tables: dict[str, str]
    adapter_registry: dict[str, Any]
    adapter_validator_registry: dict[str, Any]
    adapter_cfg: AdapterConfig | None


def build_adapter_config(data_cfg: dict[str, Any]) -> AdapterConfig:
    return AdapterConfig(
        data_dir=str(data_cfg.get("data_dir") or ""),
        required_cols=tuple(str(c).strip() for c in data_cfg.get("fields_required", []) if str(c).strip())
        or ("date", "close"),
        min_rows_per_asset=int(data_cfg.get("min_rows_per_asset", 30)),
        symbols=tuple(str(x).strip() for x in data_cfg.get("symbols", []) if str(x).strip()),
        start_date=str(data_cfg.get("start_date")) if data_cfg.get("start_date") else None,
        end_date=str(data_cfg.get("end_date")) if data_cfg.get("end_date") else None,
        request_timeout_sec=int(data_cfg.get("request_timeout_sec", 20)),
    )


def normalize_adapter_validation_warnings(result: Any) -> list[str]:
    if result is None:
        return []
    if isinstance(result, bool):
        if result:
            return []
        raise ValueError("Adapter config validator returned False.")
    if isinstance(result, str):
        text = result.strip()
        return [text] if text else []
    if isinstance(result, dict):
        raw = result.get("warnings")
        if raw is None:
            return []
        return [str(x).strip() for x in _as_list(raw) if str(x).strip()]
    if isinstance(result, (list, tuple, set)):
        return [str(x).strip() for x in result if str(x).strip()]
    raise TypeError(f"Unsupported validator return type: {type(result)}")


def validate_adapter_config(
    adapter: str,
    adapter_cfg: AdapterConfig,
    validator_registry: dict[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "adapter": adapter,
        "validated": False,
        "validator_found": False,
        "validation_seconds": 0.0,
        "warnings": [],
    }
    validator = validator_registry.get(adapter)
    if validator is None:
        return report

    report["validator_found"] = True
    t0 = time.perf_counter()
    result = validator(adapter_cfg)
    report["validation_seconds"] = float(time.perf_counter() - t0)
    report["warnings"] = normalize_adapter_validation_warnings(result)
    report["validated"] = True
    return report


def write_adapter_quality_audit_tables(
    panel: pd.DataFrame,
    data_cfg: dict[str, Any],
    load_report: dict[str, Any],
    out_dir: Path,
) -> dict[str, str]:
    table_dir = out_dir / "tables" / "data"
    table_dir.mkdir(parents=True, exist_ok=True)

    df = panel.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    rows = int(len(df))
    assets = int(df["asset"].nunique()) if ("asset" in df.columns and rows > 0) else 0
    dates = int(df["date"].nunique()) if ("date" in df.columns and rows > 0) else 0
    min_rows_threshold = int(data_cfg.get("min_rows_per_asset", 0))

    summary_rows: list[dict[str, Any]] = [
        {"category": "source", "metric": "adapter", "value": str(data_cfg.get("adapter", "")), "note": ""},
        {"category": "shape", "metric": "rows", "value": rows, "note": ""},
        {"category": "shape", "metric": "assets", "value": assets, "note": ""},
        {"category": "shape", "metric": "dates", "value": dates, "note": ""},
        {
            "category": "threshold",
            "metric": "min_rows_per_asset",
            "value": min_rows_threshold,
            "note": "用于资产样本量达标判定",
        },
        {
            "category": "timing",
            "metric": "adapter_load_seconds",
            "value": float(load_report.get("adapter_load_seconds", 0.0)),
            "note": "",
        },
    ]

    field_rows: list[dict[str, Any]] = []
    for col in [str(x).strip() for x in data_cfg.get("fields_required", []) if str(x).strip()]:
        if col not in df.columns:
            missing_rate = 1.0
            coverage_rate = 0.0
        else:
            missing_rate = float(df[col].isna().mean()) if rows > 0 else 1.0
            coverage_rate = float(1.0 - missing_rate)
        field_rows.append({"field": col, "missing_rate": missing_rate, "coverage_rate": coverage_rate})
        summary_rows.append(
            {
                "category": "missing",
                "metric": f"missing_rate__{col}",
                "value": missing_rate,
                "note": "",
            }
        )

    asset_rows = pd.DataFrame(columns=["asset", "rows", "meets_min_rows"])
    if "asset" in df.columns and rows > 0:
        asset_counts = (
            df.groupby("asset", as_index=False)
            .size()
            .rename(columns={"size": "rows"})
            .sort_values("rows", ascending=False)
            .reset_index(drop=True)
        )
        asset_counts["meets_min_rows"] = asset_counts["rows"] >= min_rows_threshold if min_rows_threshold > 0 else True
        asset_rows = asset_counts

        summary_rows.extend(
            [
                {"category": "asset_rows", "metric": "asset_rows_min", "value": int(asset_counts["rows"].min()), "note": ""},
                {
                    "category": "asset_rows",
                    "metric": "asset_rows_median",
                    "value": float(asset_counts["rows"].median()),
                    "note": "",
                },
                {"category": "asset_rows", "metric": "asset_rows_max", "value": int(asset_counts["rows"].max()), "note": ""},
                {
                    "category": "threshold",
                    "metric": "assets_meet_min_rows",
                    "value": int(asset_counts["meets_min_rows"].sum()),
                    "note": "",
                },
                {
                    "category": "threshold",
                    "metric": "assets_below_min_rows",
                    "value": int((~asset_counts["meets_min_rows"]).sum()),
                    "note": "",
                },
                {
                    "category": "threshold",
                    "metric": "assets_meet_min_rows_rate",
                    "value": float(asset_counts["meets_min_rows"].mean()),
                    "note": "",
                },
            ]
        )

    date_cov_rows = pd.DataFrame(columns=["date", "assets_covered", "coverage_rate"])
    if {"date", "asset"}.issubset(df.columns) and rows > 0 and assets > 0:
        date_cov = (
            df.groupby("date", as_index=False)["asset"]
            .nunique()
            .rename(columns={"asset": "assets_covered"})
            .sort_values("date")
            .reset_index(drop=True)
        )
        date_cov["coverage_rate"] = date_cov["assets_covered"] / max(assets, 1)
        date_cov_rows = date_cov
        summary_rows.extend(
            [
                {
                    "category": "coverage",
                    "metric": "date_coverage_mean",
                    "value": float(date_cov["coverage_rate"].mean()),
                    "note": "按交易日统计的资产覆盖率均值",
                },
                {"category": "coverage", "metric": "date_coverage_min", "value": float(date_cov["coverage_rate"].min()), "note": ""},
                {
                    "category": "coverage",
                    "metric": "date_coverage_p10",
                    "value": float(date_cov["coverage_rate"].quantile(0.1)),
                    "note": "",
                },
                {
                    "category": "coverage",
                    "metric": "date_coverage_p50",
                    "value": float(date_cov["coverage_rate"].quantile(0.5)),
                    "note": "",
                },
                {
                    "category": "coverage",
                    "metric": "date_coverage_p90",
                    "value": float(date_cov["coverage_rate"].quantile(0.9)),
                    "note": "",
                },
            ]
        )

    summary_path = table_dir / "adapter_quality_audit.csv"
    fields_path = table_dir / "field_missing_rates.csv"
    asset_path = table_dir / "asset_row_counts.csv"
    coverage_path = table_dir / "date_coverage.csv"

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(field_rows).to_csv(fields_path, index=False)
    asset_rows.to_csv(asset_path, index=False)
    date_cov_rows.to_csv(coverage_path, index=False)

    return {
        "adapter_quality_audit_csv": str(summary_path),
        "field_missing_rates_csv": str(fields_path),
        "asset_row_counts_csv": str(asset_path),
        "date_coverage_csv": str(coverage_path),
    }


def load_data(
    data_cfg: dict[str, Any],
    scope_cfg: dict[str, Any],
    adapter_registry: dict[str, Any],
    adapter_cfg: AdapterConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    adapter = data_cfg["adapter"]
    sanitize = bool(data_cfg["sanitize"])
    duplicate_policy = str(data_cfg["duplicate_policy"])
    loader_report: dict[str, Any] = {"adapter": adapter, "sanitize": sanitize}

    def attach_panel_profile(df: pd.DataFrame, source: str) -> None:
        if df.empty:
            loader_report["panel_profile"] = {
                "source": source,
                "rows": 0,
                "assets": 0,
                "dates": 0,
                "date_min": None,
                "date_max": None,
                "columns": int(len(df.columns)),
            }
            return
        dates = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else pd.Series(dtype="datetime64[ns]")
        loader_report["panel_profile"] = {
            "source": source,
            "rows": int(len(df)),
            "assets": int(df["asset"].nunique()) if "asset" in df.columns else 0,
            "dates": int(dates.nunique()) if not dates.empty else 0,
            "date_min": str(dates.min()) if not dates.empty else None,
            "date_max": str(dates.max()) if not dates.empty else None,
            "columns": int(len(df.columns)),
        }

    if adapter == "synthetic":
        synthetic_cfg = data_cfg["synthetic"]
        n_assets_default = 1 if data_cfg["mode"] == "single_asset" else 40
        syn_cfg = SyntheticConfig(
            n_assets=max(1, _to_int(synthetic_cfg.get("n_assets"), n_assets_default)),
            n_days=max(60, _to_int(synthetic_cfg.get("n_days"), 260)),
            seed=_to_int(synthetic_cfg.get("seed"), 7),
            start_date=str(synthetic_cfg.get("start_date", "2021-01-01")),
        )
        t0 = time.perf_counter()
        panel = generate_synthetic_panel(syn_cfg)
        loader_report["adapter_load_seconds"] = float(time.perf_counter() - t0)
        loader_report["synthetic"] = {
            "n_assets": syn_cfg.n_assets,
            "n_days": syn_cfg.n_days,
            "seed": syn_cfg.seed,
            "start_date": syn_cfg.start_date,
        }
        attach_panel_profile(panel, source="synthetic")
        return panel, loader_report

    if adapter in {"parquet", "csv"}:
        path = data_cfg.get("path")
        if not path:
            raise ValueError("data.path is required when data.adapter is parquet/csv")
        if Path(str(path)).is_dir():
            LOGGER.info("data.path points to directory; auto-switch to raw_dir loader.")
            adapter = "raw_dir"
        else:
            t0 = time.perf_counter()
            read_res = read_panel(
                path=str(path),
                sanitize=sanitize,
                sanitization_config=PanelSanitizationConfig(duplicate_policy=duplicate_policy),
                return_report=sanitize,
            )
            loader_report["adapter_load_seconds"] = float(time.perf_counter() - t0)
            if sanitize:
                panel, report = read_res
                loader_report["sanitization_report"] = report.__dict__ if hasattr(report, "__dict__") else str(report)
            else:
                panel = read_res
            attach_panel_profile(panel, source="file_io")
            return panel, loader_report

    if adapter == "raw_dir":
        path = data_cfg.get("path")
        if not path:
            raise ValueError("data.path is required when data.adapter is raw_dir")
        t0 = time.perf_counter()
        panel, dir_report = read_panel_directory(
            directory=str(path),
            pattern=str(data_cfg.get("raw_pattern", "*.parquet,*.csv")),
            sanitize=sanitize,
            sanitization_config=PanelSanitizationConfig(duplicate_policy=duplicate_policy),
            return_report=True,
            asset_from_filename=bool(data_cfg.get("raw_asset_from_filename", True)),
        )
        loader_report["adapter_load_seconds"] = float(time.perf_counter() - t0)
        loader_report["directory_report"] = asdict(dir_report)
        attach_panel_profile(panel, source="raw_dir")
        return panel, loader_report

    if adapter not in adapter_registry:
        raise KeyError(f"Unknown data adapter '{adapter}'. Available adapters: {sorted(adapter_registry.keys())}")

    adapter_fn = adapter_registry[adapter]
    adapter_cfg = adapter_cfg or build_adapter_config(data_cfg)
    t0 = time.perf_counter()
    panel = adapter_fn(adapter_cfg)
    loader_report["adapter_load_seconds"] = float(time.perf_counter() - t0)
    loader_report["adapter_registry_size"] = int(len(adapter_registry))
    loader_report["symbols"] = list(adapter_cfg.symbols)
    loader_report["start_date"] = adapter_cfg.start_date
    loader_report["end_date"] = adapter_cfg.end_date
    attach_panel_profile(panel, source="adapter")
    return panel, loader_report


def ensure_mode_shape(panel: pd.DataFrame, data_cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = panel.copy()
    mode = data_cfg["mode"]
    report: dict[str, Any] = {"mode": mode}

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "asset" not in out.columns:
        asset_name = str(data_cfg.get("asset") or "SINGLE_ASSET")
        out["asset"] = asset_name

    out["asset"] = out["asset"].astype(str)
    out = out.dropna(subset=["date"]).sort_values(["date", "asset"]).reset_index(drop=True)
    if mode == "single_asset":
        candidate = data_cfg.get("asset")
        if candidate is None:
            candidate = out["asset"].astype(str).drop_duplicates().iloc[0]
            LOGGER.info("data.mode=single_asset and no asset provided; using first asset: %s", candidate)
        candidate = str(candidate)
        out = out[out["asset"] == candidate].copy()
        report["selected_asset"] = candidate
    report["rows_after_mode"] = int(len(out))
    report["assets_after_mode"] = int(out["asset"].nunique()) if not out.empty else 0
    return out.reset_index(drop=True), report


class DataAdapterWorkflow:
    """Data adapter lifecycle manager: registry, validation, loading, and audit."""

    def __init__(
        self,
        data_cfg: dict[str, Any],
        scope_cfg: dict[str, Any],
        out_dir: Path,
        timings: dict[str, float],
        *,
        logger_name: str = "factorlab.workflows.config_runner",
    ) -> None:
        self.data_cfg = data_cfg
        self.scope_cfg = scope_cfg
        self.out_dir = Path(out_dir).resolve()
        self.timings = timings
        self.logger_name = logger_name

    def _build_adapter_registry(self) -> dict[str, Any]:
        with timed_stage("build_data_adapter_registry", timings=self.timings, logger_name=self.logger_name):
            return build_data_adapter_registry(
                plugin_dirs=self.data_cfg["adapter_plugin_dirs"] if self.data_cfg["adapter_auto_discover"] else [],
                plugin_specs=self.data_cfg["adapter_plugins"],
                on_plugin_error=self.data_cfg["adapter_plugin_on_error"],
                include_defaults=True,
            )

    def _build_adapter_validator_registry(self) -> dict[str, Any]:
        with timed_stage("build_data_adapter_validator_registry", timings=self.timings, logger_name=self.logger_name):
            return build_data_adapter_validator_registry(
                plugin_dirs=self.data_cfg["adapter_plugin_dirs"] if self.data_cfg["adapter_auto_discover"] else [],
                plugin_specs=self.data_cfg["adapter_plugins"],
                on_plugin_error=self.data_cfg["adapter_plugin_on_error"],
                include_defaults=True,
            )

    def _validate_adapter(
        self,
        adapter_registry: dict[str, Any],
        validator_registry: dict[str, Any],
    ) -> tuple[AdapterConfig | None, dict[str, Any]]:
        adapter_cfg: AdapterConfig | None = None
        report: dict[str, Any] = {
            "adapter": self.data_cfg["adapter"],
            "validated": False,
            "validator_found": False,
            "validation_seconds": 0.0,
            "warnings": [],
        }
        with timed_stage("validate_data_adapter_config", timings=self.timings, logger_name=self.logger_name):
            if self.data_cfg["adapter"] in adapter_registry:
                adapter_cfg = build_adapter_config(self.data_cfg)
                report = validate_adapter_config(
                    adapter=self.data_cfg["adapter"],
                    adapter_cfg=adapter_cfg,
                    validator_registry=validator_registry,
                )
        return adapter_cfg, report

    def _load_and_shape(
        self,
        adapter_registry: dict[str, Any],
        adapter_cfg: AdapterConfig | None,
    ) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
        with timed_stage("load_data", timings=self.timings, logger_name=self.logger_name):
            panel, load_report = load_data(
                self.data_cfg,
                self.scope_cfg,
                adapter_registry=adapter_registry,
                adapter_cfg=adapter_cfg,
            )
            panel, mode_report = ensure_mode_shape(panel, data_cfg=self.data_cfg)
        return panel, load_report, mode_report

    def _audit_quality(self, panel: pd.DataFrame, load_report: dict[str, Any]) -> dict[str, str]:
        with timed_stage("adapter_quality_audit", timings=self.timings, logger_name=self.logger_name):
            return write_adapter_quality_audit_tables(
                panel=panel,
                data_cfg=self.data_cfg,
                load_report=load_report,
                out_dir=self.out_dir,
            )

    def run(self) -> DataAdapterWorkflowResult:
        adapter_registry = self._build_adapter_registry()
        validator_registry = self._build_adapter_validator_registry()
        adapter_cfg, adapter_validation_report = self._validate_adapter(
            adapter_registry=adapter_registry,
            validator_registry=validator_registry,
        )
        panel, load_report, mode_report = self._load_and_shape(
            adapter_registry=adapter_registry,
            adapter_cfg=adapter_cfg,
        )
        adapter_audit_tables = self._audit_quality(panel=panel, load_report=load_report)
        return DataAdapterWorkflowResult(
            panel=panel,
            load_report=load_report,
            mode_report=mode_report,
            adapter_validation_report=adapter_validation_report,
            adapter_audit_tables=adapter_audit_tables,
            adapter_registry=adapter_registry,
            adapter_validator_registry=validator_registry,
            adapter_cfg=adapter_cfg,
        )


__all__ = [
    "DataAdapterWorkflow",
    "DataAdapterWorkflowResult",
    "build_adapter_config",
    "ensure_mode_shape",
    "load_data",
    "normalize_adapter_validation_warnings",
    "validate_adapter_config",
    "write_adapter_quality_audit_tables",
]
