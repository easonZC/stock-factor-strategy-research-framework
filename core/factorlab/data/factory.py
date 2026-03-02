"""Data-adapter factory and plugin registration helpers."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pandas as pd

from factorlab.config import AdapterConfig
from factorlab.data.adapters import prepare_sina_panel, prepare_stooq_panel
from factorlab.utils import get_logger

LOGGER = get_logger("factorlab.data.factory")
DataAdapterFn = Callable[[AdapterConfig], pd.DataFrame]


def default_data_adapter_registry() -> dict[str, DataAdapterFn]:
    """Built-in adapter functions."""
    return {
        "sina": prepare_sina_panel,
        "stooq": prepare_stooq_panel,
    }


def _snake_case(name: str) -> str:
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", str(name))
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", text)
    return text.strip("_").lower()


def _load_module(module_ref: str) -> ModuleType:
    ref = str(module_ref).strip()
    path = Path(ref).expanduser()
    if path.exists() and path.is_file():
        module_name = f"factorlab_data_plugin_{abs(hash(str(path.resolve())))}"
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load data plugin module from file: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module
    return importlib.import_module(ref)


def _adapter_name_for_callable_name(name: str) -> str:
    raw = str(name).strip()
    if raw.startswith("prepare_") and raw.endswith("_panel"):
        raw = raw[len("prepare_") : -len("_panel")]
    return _snake_case(raw)


def _is_adapter_callable(obj: Any) -> bool:
    if not callable(obj):
        return False
    try:
        sig = inspect.signature(obj)
    except Exception:
        return False
    return len(sig.parameters) >= 1


def _registry_from_module(module: ModuleType) -> dict[str, DataAdapterFn]:
    reg: dict[str, DataAdapterFn] = {}
    exported = getattr(module, "DATA_ADAPTER_REGISTRY", None)
    if isinstance(exported, dict):
        for name, fn in exported.items():
            if isinstance(name, str) and _is_adapter_callable(fn):
                reg[name.strip()] = fn

    fn_getter = getattr(module, "get_data_adapter_registry", None)
    if callable(fn_getter):
        maybe = fn_getter()
        if not isinstance(maybe, dict):
            raise TypeError("get_data_adapter_registry() must return dict[str, Callable[[AdapterConfig], DataFrame]]")
        for name, fn in maybe.items():
            if not isinstance(name, str) or not _is_adapter_callable(fn):
                raise TypeError("get_data_adapter_registry() returned invalid adapter entry.")
            reg[name.strip()] = fn

    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if fn.__module__ != module.__name__:
            continue
        if not _is_adapter_callable(fn):
            continue
        if not (name.startswith("prepare_") and name.endswith("_panel")):
            continue
        adapter_name = _adapter_name_for_callable_name(name)
        reg.setdefault(adapter_name, fn)
    return reg


def _merge_registry(
    base: dict[str, DataAdapterFn],
    incoming: dict[str, DataAdapterFn],
    source: str,
    on_error: str,
) -> dict[str, DataAdapterFn]:
    out = dict(base)
    for name, fn in incoming.items():
        if name in out:
            msg = f"Data adapter '{name}' already registered; override by plugin source: {source}"
            if on_error == "raise":
                raise ValueError(msg)
            LOGGER.warning(msg)
        out[name] = fn
    return out


def discover_data_adapter_registry(plugin_dirs: list[str | Path], on_error: str = "raise") -> dict[str, DataAdapterFn]:
    """Discover data adapters from python files under configured plugin directories."""
    reg: dict[str, DataAdapterFn] = {}
    for raw_dir in plugin_dirs:
        path = Path(raw_dir).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists() or not path.is_dir():
            msg = f"Data adapter plugin directory not found: {path}"
            if on_error == "raise":
                raise FileNotFoundError(msg)
            LOGGER.warning(msg)
            continue
        for file in sorted(path.glob("*.py")):
            if file.name.startswith("_"):
                continue
            try:
                mod = _load_module(str(file))
                sub_reg = _registry_from_module(mod)
                reg = _merge_registry(reg, sub_reg, source=str(file), on_error=on_error)
            except Exception as exc:
                msg = f"Failed to load data adapter plugin module '{file}': {exc}"
                if on_error == "raise":
                    raise RuntimeError(msg) from exc
                LOGGER.warning(msg)
    return reg


def load_data_adapter_plugins(plugin_specs: list[Any], on_error: str = "raise") -> dict[str, DataAdapterFn]:
    """Load data adapter plugins by module/callable specification."""
    reg: dict[str, DataAdapterFn] = {}
    for entry in plugin_specs:
        try:
            if isinstance(entry, str):
                mod = _load_module(entry)
                reg = _merge_registry(reg, _registry_from_module(mod), source=entry, on_error=on_error)
                continue

            if not isinstance(entry, dict):
                raise TypeError(f"Unsupported data adapter plugin entry type: {type(entry)}")

            if "callable_path" in entry:
                call_path = str(entry["callable_path"]).strip()
                module_ref, callable_name = call_path.split(":", 1) if ":" in call_path else call_path.rsplit(".", 1)
                mod = _load_module(module_ref)
                fn = getattr(mod, callable_name)
                if not _is_adapter_callable(fn):
                    raise TypeError(f"callable_path is not valid adapter callable: {call_path}")
                name = str(entry.get("name") or _adapter_name_for_callable_name(callable_name)).strip()
                reg = _merge_registry(reg, {name: fn}, source=call_path, on_error=on_error)
                continue

            module_ref = entry.get("module")
            if module_ref is None:
                raise ValueError("Data adapter plugin entry must contain either 'callable_path' or 'module'.")
            mod = _load_module(str(module_ref))
            if "callable" not in entry:
                reg = _merge_registry(reg, _registry_from_module(mod), source=str(module_ref), on_error=on_error)
                continue
            callable_name = str(entry["callable"]).strip()
            fn = getattr(mod, callable_name)
            if not _is_adapter_callable(fn):
                raise TypeError(f"module/callable is not valid adapter callable: {module_ref}.{callable_name}")
            name = str(entry.get("name") or _adapter_name_for_callable_name(callable_name)).strip()
            reg = _merge_registry(reg, {name: fn}, source=f"{module_ref}.{callable_name}", on_error=on_error)
        except Exception as exc:
            msg = f"Failed to load data adapter plugin entry {entry!r}: {exc}"
            if on_error == "raise":
                raise RuntimeError(msg) from exc
            LOGGER.warning(msg)
    return reg


def build_data_adapter_registry(
    plugin_dirs: list[str | Path] | None = None,
    plugin_specs: list[Any] | None = None,
    on_plugin_error: str = "raise",
    include_defaults: bool = True,
) -> dict[str, DataAdapterFn]:
    """Build data adapter registry from built-ins plus optional plugin discovery/loading."""
    registry = default_data_adapter_registry() if include_defaults else {}
    if plugin_dirs:
        registry = _merge_registry(
            registry,
            discover_data_adapter_registry(plugin_dirs=plugin_dirs, on_error=on_plugin_error),
            source="plugin_dirs",
            on_error=on_plugin_error,
        )
    if plugin_specs:
        registry = _merge_registry(
            registry,
            load_data_adapter_plugins(plugin_specs=plugin_specs, on_error=on_plugin_error),
            source="plugin_specs",
            on_error=on_plugin_error,
        )
    return registry

