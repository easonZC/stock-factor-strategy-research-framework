"""数据读写、适配器注册与股票池过滤模块导出。"""

from .adapters import prepare_sina_panel, prepare_stooq_panel
from .factory import (
    build_data_adapter_registry,
    build_data_adapter_validator_registry,
    default_data_adapter_registry,
    default_data_adapter_validator_registry,
    discover_data_adapter_registry,
    discover_data_adapter_validator_registry,
    load_data_adapter_plugins,
    load_data_adapter_validator_plugins,
)
from .io import (
    PanelDirectoryReadReport,
    PanelSanitizationConfig,
    PanelSanitizationReport,
    read_panel_directory,
    read_panel,
    write_panel,
)
from .synthetic import generate_synthetic_panel
from .universe import UniverseFilterReport, apply_universe_filter

__all__ = [
    "PanelSanitizationConfig",
    "PanelSanitizationReport",
    "PanelDirectoryReadReport",
    "UniverseFilterReport",
    "apply_universe_filter",
    "build_data_adapter_registry",
    "build_data_adapter_validator_registry",
    "default_data_adapter_registry",
    "default_data_adapter_validator_registry",
    "discover_data_adapter_registry",
    "discover_data_adapter_validator_registry",
    "generate_synthetic_panel",
    "load_data_adapter_plugins",
    "load_data_adapter_validator_plugins",
    "prepare_sina_panel",
    "prepare_stooq_panel",
    "read_panel_directory",
    "read_panel",
    "write_panel",
]
