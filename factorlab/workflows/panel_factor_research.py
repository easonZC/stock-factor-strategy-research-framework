"""面板因子研究工作流封装。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .config_runner import ConfigRunResult, run_from_config


MissingPolicy = Literal["drop", "fill_zero", "ffill_by_asset", "cs_median_by_date", "keep"]
OnMissingFactor = Literal["raise", "warn_skip"]
NeutralizeMode = Literal["none", "size", "industry", "both"]
WinsorizeMode = Literal["quantile", "mad"]
CSStandardization = Literal["cs_zscore", "cs_rank", "cs_robust_zscore", "none"]


def _to_text_list(value: list[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]


def _to_int_list(value: list[int] | int | str | None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        vals: list[int] = []
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                vals.append(int(token))
            except ValueError:
                continue
        return vals
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


@dataclass(slots=True)
class PanelFactorResearchConfig:
    """面板研究高层配置（供 apps 与外部 API 复用）。"""

    factors: list[str] | str | None = field(default_factory=list)
    horizons: list[int] | int | str | None = field(default_factory=lambda: [1, 5, 10, 20])
    neutralize: NeutralizeMode = "both"
    winsorize: WinsorizeMode = "quantile"
    standardization: CSStandardization = "cs_zscore"
    missing_policy: MissingPolicy = "drop"
    preprocess_steps: list[str] | str | None = field(default_factory=lambda: ["winsorize", "standardize", "neutralize"])
    quantiles: int = 5
    ic_rolling_window: int = 20
    on_missing_factor: OnMissingFactor = "warn_skip"


def build_panel_factor_research_run_config(
    panel_path: str | Path,
    config: PanelFactorResearchConfig,
) -> dict:
    """将高层配置转换为统一工作流配置。"""
    factor_names = _to_text_list(config.factors)
    horizons = [h for h in _to_int_list(config.horizons) if h > 0] or [1, 5, 10, 20]
    preprocess_steps = [x.lower() for x in _to_text_list(config.preprocess_steps)] or [
        "winsorize",
        "standardize",
        "neutralize",
    ]

    cfg: dict = {
        "run": {
            "factor_scope": "cs",
            "eval_axis": "cross_section",
            "standardization": str(config.standardization),
            "stop_after": "research",
            "research_profile": "full",
        },
        "data": {
            "path": str(panel_path),
            "mode": "panel",
            "fields_required": ["date", "asset", "close"],
        },
        "factor": {
            "on_missing": str(config.on_missing_factor),
        },
        "research": {
            "horizons": horizons,
            "quantiles": int(config.quantiles),
            "ic_rolling_window": int(config.ic_rolling_window),
            "missing_policy": str(config.missing_policy),
            "preprocess_steps": preprocess_steps,
            "winsorize": {
                "enabled": True,
                "method": str(config.winsorize),
            },
            "neutralize": {
                "enabled": str(config.neutralize) != "none",
                "mode": str(config.neutralize),
            },
        },
        "backtest": {
            "enabled": False,
        },
    }
    if factor_names:
        cfg["factor"]["names"] = factor_names
    return cfg


def run_panel_factor_research(
    panel_path: str | Path,
    out_dir: str | Path,
    config: PanelFactorResearchConfig,
    repo_root: str | Path | None = None,
    validate_schema: bool = True,
) -> ConfigRunResult:
    """以面板文件为输入运行因子研究。"""
    workflow_cfg = build_panel_factor_research_run_config(panel_path=panel_path, config=config)
    return run_from_config(
        config=workflow_cfg,
        out_dir=out_dir,
        repo_root=repo_root,
        validate_schema=validate_schema,
    )

