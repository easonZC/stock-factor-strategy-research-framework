"""Reusable workflow service for multi-model OOF factor benchmarking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from factorlab.config import NeutralizationConfig, ResearchConfig, UniverseFilterConfig
from factorlab.data import PanelSanitizationConfig, apply_universe_filter, read_panel
from factorlab.factors import apply_factors, default_factor_registry
from factorlab.models import ModelRegistry, OOFSplitConfig, train_oof_model_factor
from factorlab.research import FactorResearchPipeline
from factorlab.utils import get_logger, timed_stage
from factorlab.workflows.runtime import collect_runtime_manifest

LOGGER = get_logger("factorlab.workflows.model_factor_benchmark")


DuplicatePolicy = Literal["last", "first", "raise"]
NeutralizeMode = Literal["none", "size", "industry", "both"]
WinsorizeMode = Literal["quantile", "mad"]

DEFAULT_MODELS = ["lgbm", "mlp"]
DEFAULT_FEATURE_COLS = ["momentum_20", "volatility_20", "liquidity_shock", "size"]
DEFAULT_HORIZONS = [1, 5, 10, 20]
MODEL_ALIASES = {
    "ridge": "ridge",
    "linear": "ridge",
    "linear_regression": "ridge",
    "rf": "rf",
    "random_forest": "rf",
    "randomforest": "rf",
    "mlp": "mlp",
    "nn": "mlp",
    "neural_net": "mlp",
    "neural_network": "mlp",
    "lgbm": "lgbm",
    "lightgbm": "lgbm",
}


@dataclass(slots=True)
class ModelFactorBenchmarkConfig:
    """Config for model-factor OOF benchmark workflow."""

    # Accept list or comma-separated string for friendlier API usage.
    models: list[str] | str | None = field(default_factory=lambda: list(DEFAULT_MODELS))
    factor_prefix: str = "model_factor_oof"
    feature_cols: list[str] | str | None = field(default_factory=lambda: list(DEFAULT_FEATURE_COLS))
    extra_report_factors: list[str] | str | None = field(default_factory=list)
    label_horizon: int = 5

    train_days: int = 252
    valid_days: int = 21
    step_days: int = 21
    embargo_days: int | None = None
    purge_days: int = 0
    split_mode: Literal["rolling", "expanding"] = "rolling"
    min_train_rows: int = 500
    min_valid_rows: int = 100
    scoring_metric: Literal["rank_ic", "mse"] = "rank_ic"
    evaluation_axis: Literal["cross_section", "time"] = "cross_section"
    model_param_grid_dir: str | None = None

    horizons: list[int] | str | None = field(default_factory=lambda: list(DEFAULT_HORIZONS))
    neutralize: NeutralizeMode = "both"
    winsorize: WinsorizeMode = "quantile"
    quantiles: int = 5
    ic_rolling_window: int = 20
    preferred_metric_variant: str = "auto"

    start_date: str | None = None
    end_date: str | None = None
    warmup_days: int = 0
    max_assets: int | None = None

    sanitize: bool = True
    duplicate_policy: DuplicatePolicy = "last"

    apply_universe_filter: bool = False
    universe_filter: UniverseFilterConfig = field(default_factory=UniverseFilterConfig)

    save_model_artifacts: bool = False
    model_artifact_dir: str = "artifacts/models/model_factor_benchmark"


@dataclass(slots=True)
class ModelFactorBenchmarkResult:
    """Output artifact pointers for one benchmark run."""

    out_dir: Path
    index_html: Path
    summary_csv: Path
    comparison_csv: Path
    run_meta_json: Path
    run_manifest_json: Path


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return p


def _parse_model_param_grid(grid_dir: str | None, model_name: str) -> list[dict[str, object]] | None:
    if grid_dir is None:
        return None
    p = Path(grid_dir) / f"{model_name}.json"
    if not p.exists():
        return None
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Model grid file must be list[dict]: {p}")
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Grid item at index {i} is not a dict in file: {p}")
    return payload  # type: ignore[return-value]


def _coerce_name_list(raw: list[str] | str | None, default: list[str]) -> list[str]:
    if raw is None:
        return list(default)
    if isinstance(raw, str):
        parts = [x.strip() for x in raw.split(",")]
        cleaned = [x for x in parts if x]
        return cleaned if cleaned else list(default)
    cleaned = [str(x).strip() for x in raw if str(x).strip()]
    return cleaned if cleaned else list(default)


def _coerce_int_list(raw: list[int] | str | None, default: list[int], min_value: int = 1) -> list[int]:
    if raw is None:
        return list(default)
    vals: list[int] = []
    if isinstance(raw, str):
        tokens = [x.strip() for x in raw.split(",") if x.strip()]
        for t in tokens:
            try:
                v = int(t)
            except ValueError:
                continue
            if v >= min_value:
                vals.append(v)
    else:
        for item in raw:
            try:
                v = int(item)
            except (TypeError, ValueError):
                continue
            if v >= min_value:
                vals.append(v)
    if not vals:
        return list(default)
    return sorted(set(vals))


def _coerce_choice(raw: str, allowed: set[str], default: str, field_name: str) -> str:
    key = str(raw).strip().lower()
    if key in allowed:
        return key
    LOGGER.warning(
        "Invalid %s='%s'. Falling back to default '%s'.",
        field_name,
        raw,
        default,
    )
    return default


def _resolve_models(raw_models: list[str] | str | None) -> list[str]:
    parsed = _coerce_name_list(raw_models, default=DEFAULT_MODELS)
    resolved: list[str] = []
    for original in parsed:
        alias = MODEL_ALIASES.get(str(original).strip().lower(), str(original).strip().lower())
        if alias not in ModelRegistry._defaults:
            LOGGER.warning(
                "Skip unsupported model '%s'. Available models: %s",
                original,
                sorted(ModelRegistry._defaults),
            )
            continue
        if alias not in resolved:
            resolved.append(alias)
    if not resolved:
        raise ValueError("No valid models after normalization. Please provide at least one supported model.")
    return resolved


def _apply_basic_panel_filters(
    panel: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
    max_assets: int | None,
) -> pd.DataFrame:
    out = panel.copy()
    if start_date:
        out = out[out["date"] >= pd.to_datetime(start_date)].copy()
    if end_date:
        out = out[out["date"] <= pd.to_datetime(end_date)].copy()
    if max_assets is not None and int(max_assets) > 0:
        keep_assets = out["asset"].astype(str).drop_duplicates().head(int(max_assets))
        out = out[out["asset"].astype(str).isin(set(keep_assets))].copy()
    return out


def _compute_prefilter_start(start_date: str | None, warmup_days: int) -> str | None:
    if not start_date:
        return None
    if warmup_days <= 0:
        return start_date
    ts = pd.to_datetime(start_date) - pd.tseries.offsets.BDay(int(warmup_days))
    return ts.strftime("%Y-%m-%d")


def _safe_float(value: object) -> float:
    try:
        if pd.isna(value):
            return float("nan")
    except TypeError:
        pass
    return float(value)


def _empty_primary_metrics(variant: str) -> dict[str, float | int | str]:
    return {
        "research_variant": variant,
        "research_horizon": -1,
        "research_rank_ic_mean": float("nan"),
        "research_rank_icir": float("nan"),
        "research_nw_t_rank_ic": float("nan"),
        "research_nw_p_rank_ic": float("nan"),
        "research_ls_mean_ret": float("nan"),
        "research_ls_sharpe": float("nan"),
        "research_ls_alpha_ann": float("nan"),
        "research_rank_autocorr_lag1_mean": float("nan"),
    }


def _extract_primary_metrics(
    summary: pd.DataFrame,
    factor_name: str,
    preferred_horizon: int,
    variant: str = "auto",
) -> dict[str, float | int | str]:
    preferred = str(variant).strip().lower()
    if preferred in {"raw"}:
        variant_order = ["raw", "neutralized"]
    elif preferred in {"neutralized", "neutral"}:
        variant_order = ["neutralized", "raw"]
    else:
        variant_order = ["neutralized", "raw"]

    for v in variant_order:
        subset = summary[(summary["factor"] == factor_name) & (summary["variant"] == v)].copy()
        if subset.empty:
            continue
        ic_rows = subset[subset["rank_ic_mean"].notna()].copy()
        if ic_rows.empty:
            continue
        if int(preferred_horizon) in set(ic_rows["horizon"].astype(int)):
            ic_row = ic_rows[ic_rows["horizon"].astype(int) == int(preferred_horizon)].iloc[0]
        else:
            ic_row = ic_rows.sort_values("horizon").iloc[0]

        chosen_h = int(ic_row["horizon"])
        ls_rows = subset[(subset["horizon"].astype(int) == chosen_h) & (subset["ls_sharpe"].notna())]
        ls_row = ls_rows.iloc[0] if not ls_rows.empty else pd.Series(dtype=float)
        return {
            "research_variant": v,
            "research_horizon": chosen_h,
            "research_rank_ic_mean": _safe_float(ic_row.get("rank_ic_mean", np.nan)),
            "research_rank_icir": _safe_float(ic_row.get("rank_icir", np.nan)),
            "research_nw_t_rank_ic": _safe_float(ic_row.get("nw_t_rank_ic", np.nan)),
            "research_nw_p_rank_ic": _safe_float(ic_row.get("nw_p_rank_ic", np.nan)),
            "research_ls_mean_ret": _safe_float(ls_row.get("ls_mean_ret", np.nan)),
            "research_ls_sharpe": _safe_float(ls_row.get("ls_sharpe", np.nan)),
            "research_ls_alpha_ann": _safe_float(ls_row.get("ls_alpha_ann", np.nan)),
            "research_rank_autocorr_lag1_mean": _safe_float(
                ls_row.get("rank_autocorr_lag1_mean", np.nan)
            ),
        }

    return _empty_primary_metrics(variant=str(variant))


def run_model_factor_benchmark(
    panel_path: str | Path,
    out_dir: str | Path,
    config: ModelFactorBenchmarkConfig,
    repo_root: str | Path | None = None,
) -> ModelFactorBenchmarkResult:
    """Run end-to-end multi-model factor benchmark and produce comparison table."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    models = _resolve_models(config.models)
    feature_cols = _coerce_name_list(config.feature_cols, default=DEFAULT_FEATURE_COLS)
    extra_report_factors = _coerce_name_list(config.extra_report_factors, default=[])
    horizons = _coerce_int_list(config.horizons, default=DEFAULT_HORIZONS, min_value=1)
    neutralize_mode = _coerce_choice(
        raw=str(config.neutralize),
        allowed={"none", "size", "industry", "both"},
        default="both",
        field_name="neutralize",
    )
    winsorize_mode = _coerce_choice(
        raw=str(config.winsorize),
        allowed={"quantile", "mad"},
        default="quantile",
        field_name="winsorize",
    )
    duplicate_policy = _coerce_choice(
        raw=str(config.duplicate_policy),
        allowed={"last", "first", "raise"},
        default="last",
        field_name="duplicate_policy",
    )

    with timed_stage("read_panel", timings=timings, logger_name="factorlab.workflows.model_factor_benchmark"):
        read_result = read_panel(
            panel_path,
            sanitize=config.sanitize,
            sanitization_config=PanelSanitizationConfig(duplicate_policy=duplicate_policy),
            return_report=config.sanitize,
        )
        if config.sanitize:
            panel, sanitization_report = read_result
        else:
            panel = read_result
            sanitization_report = None

    with timed_stage("panel_prepare", timings=timings, logger_name="factorlab.workflows.model_factor_benchmark"):
        prefilter_start = _compute_prefilter_start(config.start_date, config.warmup_days)
        panel = _apply_basic_panel_filters(
            panel=panel,
            start_date=prefilter_start,
            end_date=config.end_date,
            max_assets=config.max_assets,
        )

        required_factor_cols = sorted(set(feature_cols + extra_report_factors))
        registry = default_factor_registry()
        missing = [f for f in required_factor_cols if f not in panel.columns]
        computable = [f for f in missing if f in registry]
        if computable:
            panel = apply_factors(panel, computable, inplace=True)
        unresolved = [f for f in required_factor_cols if f not in panel.columns]
        if unresolved:
            raise KeyError(f"Required factors missing and not computable: {unresolved}")

        universe_report = None
        if config.apply_universe_filter:
            panel, universe_report = apply_universe_filter(panel, config=config.universe_filter)

        if config.start_date:
            panel = panel[panel["date"] >= pd.to_datetime(config.start_date)].copy()
        panel = panel.sort_values(["date", "asset"]).reset_index(drop=True)
        if panel.empty:
            raise RuntimeError("Panel is empty after filtering.")

    oof_cfg = OOFSplitConfig(
        train_days=int(config.train_days),
        valid_days=int(config.valid_days),
        step_days=int(config.step_days),
        embargo_days=(
            int(config.embargo_days) if config.embargo_days is not None else int(config.label_horizon)
        ),
        purge_days=int(config.purge_days),
        split_mode=str(config.split_mode),
        min_train_rows=int(config.min_train_rows),
        min_valid_rows=int(config.min_valid_rows),
    )

    model_rows: list[dict[str, object]] = []
    factor_cols: list[str] = []
    for model_name in models:
        with timed_stage(
            f"train_oof_{model_name}",
            timings=timings,
            logger_name="factorlab.workflows.model_factor_benchmark",
        ):
            LOGGER.info("Training OOF model factor: %s", model_name)
            param_grid = _parse_model_param_grid(config.model_param_grid_dir, model_name=model_name)
            oof_res = train_oof_model_factor(
                panel=panel,
                feature_cols=feature_cols,
                model_name=model_name,
                label_horizon=int(config.label_horizon),
                split_config=oof_cfg,
                param_grid=param_grid,
                scoring_metric=str(config.scoring_metric),
                evaluation_axis=str(config.evaluation_axis),
            )

            factor_col = f"{config.factor_prefix}_{model_name}"
            oof_pred = (
                oof_res.oof_predictions.groupby(["date", "asset"], as_index=False)["pred"].mean().rename(
                    columns={"pred": factor_col}
                )
            )
            if factor_col in panel.columns:
                panel = panel.drop(columns=[factor_col])
            panel = panel.merge(oof_pred, on=["date", "asset"], how="left")
            panel[factor_col] = panel[factor_col].astype(float)
            factor_cols.append(factor_col)

            model_dir = out / f"model_{model_name}"
            model_dir.mkdir(parents=True, exist_ok=True)
            pred_path = model_dir / "oof_predictions.csv"
            folds_path = model_dir / "oof_folds.csv"
            tuning_path = model_dir / "oof_tuning.csv"
            oof_res.oof_predictions.to_csv(pred_path, index=False)
            oof_res.fold_summary.to_csv(folds_path, index=False)
            oof_res.tuning_summary.to_csv(tuning_path, index=False)

            model_path = ""
            if config.save_model_artifacts:
                artifact_path = Path(config.model_artifact_dir) / f"{factor_col}.joblib"
                saved = ModelRegistry.save(
                    model=oof_res.final_model,
                    path=artifact_path,
                    model_name=model_name,
                    metadata={
                        "feature_cols": feature_cols,
                        "label_horizon": int(config.label_horizon),
                        "best_params": oof_res.best_params,
                        "oof_score_metric": str(config.scoring_metric),
                        "oof_evaluation_axis": str(config.evaluation_axis),
                        "best_oof_score": oof_res.best_score,
                    },
                )
                model_path = str(saved)

            coverage = float(panel[factor_col].notna().mean()) if len(panel) > 0 else 0.0
            model_rows.append(
                {
                    "model": model_name,
                    "factor_name": factor_col,
                    "oof_score_metric": str(config.scoring_metric),
                    "oof_evaluation_axis": str(config.evaluation_axis),
                    "best_oof_score": float(oof_res.best_score),
                    "best_oof_rank_ic": (
                        float(oof_res.best_score)
                        if str(config.scoring_metric).strip().lower() == "rank_ic"
                        else float("nan")
                    ),
                    "oof_rows": int(len(oof_res.oof_predictions)),
                    "oof_unique_dates": int(oof_res.oof_predictions["date"].nunique()),
                    "factor_coverage_in_panel": coverage,
                    "best_params_json": json.dumps(_to_jsonable(oof_res.best_params), ensure_ascii=False),
                    "oof_predictions_csv": str(pred_path),
                    "oof_folds_csv": str(folds_path),
                    "oof_tuning_csv": str(tuning_path),
                    "model_path": model_path,
                }
            )

    with timed_stage("research_report", timings=timings, logger_name="factorlab.workflows.model_factor_benchmark"):
        report_factors = [*factor_cols, *extra_report_factors]
        research_cfg = ResearchConfig(
            horizons=horizons,
            quantiles=int(config.quantiles),
            ic_rolling_window=int(config.ic_rolling_window),
            winsorize_method=winsorize_mode,
            neutralization=NeutralizationConfig(mode=neutralize_mode),
        )
        report_outputs = FactorResearchPipeline(research_cfg).run(
            panel=panel,
            factors=report_factors,
            out_dir=out,
        )

    summary = pd.read_csv(report_outputs["summary_csv"])
    rows: list[dict[str, object]] = []
    for item in model_rows:
        fac = str(item["factor_name"])
        metrics = _extract_primary_metrics(
            summary=summary,
            factor_name=fac,
            preferred_horizon=int(config.label_horizon),
            variant=config.preferred_metric_variant,
        )
        row = dict(item)
        row.update(metrics)
        rows.append(row)
    comparison = pd.DataFrame(rows).sort_values(
        ["research_rank_ic_mean", "best_oof_score"],
        ascending=[False, False],
    )
    comparison_path = out / "model_factor_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    run_meta = {
        "panel_path": str(panel_path),
        "config": config,
        "resolved_models": models,
        "resolved_feature_cols": feature_cols,
        "resolved_horizons": horizons,
        "report_factors": report_factors,
        "sanitize_enabled": bool(config.sanitize),
        "sanitization_report": sanitization_report,
        "universe_filter_enabled": bool(config.apply_universe_filter),
        "universe_report": universe_report,
        "rows_after_filters": int(len(panel)),
        "assets_after_filters": int(panel["asset"].nunique()),
        "dates_after_filters": int(panel["date"].nunique()),
        "timings_seconds": timings,
        "outputs": {k: str(v) for k, v in report_outputs.items()},
        "comparison_csv": str(comparison_path),
    }
    run_meta_path = _write_json(out / "run_meta.json", run_meta)
    run_manifest_path = _write_json(
        out / "run_manifest.json",
        collect_runtime_manifest(repo_root=repo_root),
    )

    index_html = Path(report_outputs["index_html"])
    summary_csv = Path(report_outputs["summary_csv"])
    return ModelFactorBenchmarkResult(
        out_dir=out,
        index_html=index_html,
        summary_csv=summary_csv,
        comparison_csv=comparison_path,
        run_meta_json=run_meta_path,
        run_manifest_json=run_manifest_path,
    )
