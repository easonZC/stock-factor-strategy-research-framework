"""Data ingestion, synthetic generation, and universe filtering utilities."""

from .adapters import prepare_sina_panel, prepare_stooq_panel
from .factory import (
    build_data_adapter_registry,
    default_data_adapter_registry,
    discover_data_adapter_registry,
    load_data_adapter_plugins,
)
from .io import (
    PanelSanitizationConfig,
    PanelSanitizationReport,
    read_panel,
    write_panel,
)
from .synthetic import generate_synthetic_panel
from .universe import UniverseFilterReport, apply_universe_filter

__all__ = [
    "PanelSanitizationConfig",
    "PanelSanitizationReport",
    "UniverseFilterReport",
    "apply_universe_filter",
    "build_data_adapter_registry",
    "default_data_adapter_registry",
    "discover_data_adapter_registry",
    "generate_synthetic_panel",
    "load_data_adapter_plugins",
    "prepare_sina_panel",
    "prepare_stooq_panel",
    "read_panel",
    "write_panel",
]
