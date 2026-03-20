"""Research stage orchestration for CS and TS workflows."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from factorlab.config import NeutralizationConfig, ResearchConfig
from factorlab.runtime import OutputContext, coerce_output_context
from factorlab.research import FactorResearchPipeline, TSResearchConfig, TimeSeriesFactorResearchPipeline
from factorlab.utils import get_logger, timed_stage
from factorlab.workflows.config_normalization import (
    as_dict as _as_dict,
    to_bool as _to_bool,
    to_float as _to_float,
)


LOGGER = get_logger("factorlab.workflows.config_runner")


def run_research_stage(
    *,
    panel: pd.DataFrame,
    effective_factors: list[str],
    scope_cfg: dict[str, Any],
    research_cfg: dict[str, Any],
    out_dir: OutputContext | str | Path,
    overview_files: dict[str, Path] | None = None,
    timings: dict[str, float] | None = None,
    logger_name: str = "factorlab.workflows.config_runner",
) -> tuple[dict[str, Path], list[Any]]:
    output_context = coerce_output_context(out_dir)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with timed_stage("research", timings=timings if timings is not None else {}, logger_name=logger_name):
            if scope_cfg["factor_scope"] == "cs":
                wins = _as_dict(research_cfg.get("winsorize"))
                neu = _as_dict(research_cfg.get("neutralize"))
                neutral_mode = str(neu.get("mode", "both")).strip().lower() if _to_bool(neu.get("enabled"), True) else "none"
                if neutral_mode not in {"none", "size", "industry", "both"}:
                    LOGGER.warning("Invalid research.neutralize.mode='%s'. Use 'both'.", neutral_mode)
                    neutral_mode = "both"

                cs_cfg = ResearchConfig(
                    horizons=research_cfg["horizons"],
                    quantiles=int(research_cfg["quantiles"]),
                    ic_rolling_window=int(research_cfg["ic_rolling_window"]),
                    annualization_days=int(research_cfg["annualization_days"]),
                    standardization=scope_cfg["standardization"],  # type: ignore[arg-type]
                    winsorize_enabled=_to_bool(wins.get("enabled"), True),
                    winsorize_method=str(wins.get("method", "quantile")).strip().lower(),
                    lower_q=_to_float(wins.get("lower_q"), 0.01),
                    upper_q=_to_float(wins.get("upper_q"), 0.99),
                    mad_scale=max(1.0, _to_float(wins.get("mad_scale"), 5.0)),
                    missing_policy=str(research_cfg.get("missing_policy", "drop")),
                    preprocess_steps=research_cfg.get("preprocess_steps") or ["winsorize", "standardize", "neutralize"],
                    neutralization=NeutralizationConfig(mode=neutral_mode),  # type: ignore[arg-type]
                )
                outputs = FactorResearchPipeline(cs_cfg).run(
                    panel=panel,
                    factors=effective_factors,
                    out_dir=output_context.root,
                    output_context=output_context,
                    overview_files=overview_files,
                )
            else:
                ts_cfg = TSResearchConfig(
                    horizons=research_cfg["horizons"],
                    quantiles=int(research_cfg["quantiles"]),
                    ic_rolling_window=int(research_cfg["ic_rolling_window"]),
                    annualization_days=int(research_cfg["annualization_days"]),
                    standardization=scope_cfg["standardization"],  # type: ignore[arg-type]
                    ts_standardize_window=int(research_cfg["ts_standardize_window"]),
                    ts_quantile_lookback=int(research_cfg["ts_quantile_lookback"]),
                    ts_signal_lags=list(research_cfg["ts_signal_lags"]),
                )
                outputs = TimeSeriesFactorResearchPipeline(ts_cfg).run(
                    panel=panel,
                    factors=effective_factors,
                    out_dir=output_context.root,
                    output_context=output_context,
                    overview_files=overview_files,
                )

    return outputs, list(caught)


__all__ = ["run_research_stage"]
