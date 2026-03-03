"""工作流聚合导出。

对外暴露配置运行、模型因子基准和运行时审计能力。
"""

from .config_runner import (
    ConfigRunResult,
    apply_config_override,
    compose_run_config,
    compose_run_config_with_alias_report,
    deep_merge_dict,
    load_run_config,
    normalize_run_config_aliases,
    run_from_config,
    validate_run_config_schema,
)
from .model_factor_benchmark import (
    ModelFactorBenchmarkConfig,
    ModelFactorBenchmarkResult,
    run_model_factor_benchmark,
)
from .panel_factor_research import (
    PanelFactorResearchConfig,
    build_panel_factor_research_run_config,
    run_panel_factor_research,
)
from .runtime import collect_runtime_manifest

__all__ = [
    "ConfigRunResult",
    "ModelFactorBenchmarkConfig",
    "ModelFactorBenchmarkResult",
    "PanelFactorResearchConfig",
    "apply_config_override",
    "build_panel_factor_research_run_config",
    "collect_runtime_manifest",
    "compose_run_config",
    "compose_run_config_with_alias_report",
    "deep_merge_dict",
    "load_run_config",
    "normalize_run_config_aliases",
    "run_from_config",
    "run_panel_factor_research",
    "run_model_factor_benchmark",
    "validate_run_config_schema",
]
