"""Strategy factory and plugin registration helpers."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from factorlab.strategies.base import Strategy
from factorlab.strategies.implementations import (
    FlexibleLongShortStrategy,
    LongShortQuantileStrategy,
    TopKLongStrategy,
)
from factorlab.utils import get_logger

LOGGER = get_logger("factorlab.strategies.factory")
StrategyCtor = Callable[[], Strategy]


def default_strategy_registry() -> dict[str, StrategyCtor]:
    """Built-in strategy constructors."""
    return {
        "topk": lambda: TopKLongStrategy(name="topk_long", top_k=20),
        "longshort": lambda: LongShortQuantileStrategy(name="long_short_quantile", quantile=0.2),
        "flex": lambda: FlexibleLongShortStrategy(name="flexible_long_short"),
    }


def _snake_case(name: str) -> str:
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", str(name))
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", text)
    return text.strip("_").lower()


def _load_module(module_ref: str) -> ModuleType:
    ref = str(module_ref).strip()
    path = Path(ref).expanduser()
    if path.exists() and path.is_file():
        module_name = f"factorlab_strategy_plugin_{abs(hash(str(path.resolve())))}"
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load strategy plugin module from file: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module
    return importlib.import_module(ref)


def _iter_strategy_classes(module: ModuleType) -> list[type[Strategy]]:
    classes: list[type[Strategy]] = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj is Strategy:
            continue
        if not issubclass(obj, Strategy):
            continue
        if inspect.isabstract(obj):
            continue
        if obj.__module__ != module.__name__:
            continue
        classes.append(obj)
    return classes


def _strategy_name_for_class(cls: type[Strategy]) -> str:
    explicit = getattr(cls, "STRATEGY_NAME", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    name = cls.__name__
    if name.endswith("Strategy"):
        name = name[:-8]
    return _snake_case(name)


def _build_ctor(cls: type[Strategy], strategy_name: str, init_kwargs: dict[str, Any] | None = None) -> StrategyCtor:
    kwargs = dict(init_kwargs or {})
    sig = inspect.signature(cls)
    if "name" in sig.parameters and "name" not in kwargs:
        kwargs["name"] = strategy_name

    def _ctor() -> Strategy:
        obj = cls(**kwargs)
        if not isinstance(obj, Strategy):
            raise TypeError(f"Expected Strategy instance from {cls}, got {type(obj)}")
        return obj

    _ctor()  # fail fast for constructor issues
    return _ctor


def _registry_from_module(module: ModuleType) -> dict[str, StrategyCtor]:
    reg: dict[str, StrategyCtor] = {}
    exported = getattr(module, "STRATEGY_REGISTRY", None)
    if isinstance(exported, dict):
        for name, ctor in exported.items():
            if isinstance(name, str) and callable(ctor):
                reg[name.strip()] = ctor

    fn = getattr(module, "get_strategy_registry", None)
    if callable(fn):
        maybe = fn()
        if not isinstance(maybe, dict):
            raise TypeError("get_strategy_registry() must return dict[str, Callable[[], Strategy]]")
        for name, ctor in maybe.items():
            if not isinstance(name, str) or not callable(ctor):
                raise TypeError("get_strategy_registry() returned invalid registry entry")
            reg[name.strip()] = ctor

    for cls in _iter_strategy_classes(module):
        name = _strategy_name_for_class(cls)
        if name in reg:
            continue
        reg[name] = _build_ctor(cls, strategy_name=name)
    return reg


def _merge_registry(
    base: dict[str, StrategyCtor],
    incoming: dict[str, StrategyCtor],
    source: str,
    on_error: str,
) -> dict[str, StrategyCtor]:
    out = dict(base)
    for name, ctor in incoming.items():
        if name in out:
            msg = f"Strategy '{name}' already registered; override by plugin source: {source}"
            if on_error == "raise":
                raise ValueError(msg)
            LOGGER.warning(msg)
        out[name] = ctor
    return out


def discover_strategy_registry(plugin_dirs: list[str | Path], on_error: str = "raise") -> dict[str, StrategyCtor]:
    """Discover strategy classes from python files under configured plugin directories."""
    reg: dict[str, StrategyCtor] = {}
    for raw_dir in plugin_dirs:
        path = Path(raw_dir).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists() or not path.is_dir():
            msg = f"Strategy plugin directory not found: {path}"
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
                msg = f"Failed to load strategy plugin module '{file}': {exc}"
                if on_error == "raise":
                    raise RuntimeError(msg) from exc
                LOGGER.warning(msg)
    return reg


def load_strategy_plugins(plugin_specs: list[Any], on_error: str = "raise") -> dict[str, StrategyCtor]:
    """Load strategy plugins by module/class specifications."""
    reg: dict[str, StrategyCtor] = {}
    for entry in plugin_specs:
        try:
            if isinstance(entry, str):
                mod = _load_module(entry)
                reg = _merge_registry(reg, _registry_from_module(mod), source=entry, on_error=on_error)
                continue

            if not isinstance(entry, dict):
                raise TypeError(f"Unsupported strategy plugin entry type: {type(entry)}")

            init_kwargs = entry.get("init")
            if init_kwargs is not None and not isinstance(init_kwargs, dict):
                raise TypeError("Strategy plugin entry 'init' must be a dict when provided.")

            if "class_path" in entry:
                class_path = str(entry["class_path"]).strip()
                module_ref, cls_name = class_path.split(":", 1) if ":" in class_path else class_path.rsplit(".", 1)
                mod = _load_module(module_ref)
                cls = getattr(mod, cls_name)
                if not inspect.isclass(cls) or not issubclass(cls, Strategy):
                    raise TypeError(f"class_path is not a Strategy subclass: {class_path}")
                name = str(entry.get("name") or _strategy_name_for_class(cls)).strip()
                reg = _merge_registry(
                    reg,
                    {name: _build_ctor(cls, strategy_name=name, init_kwargs=init_kwargs)},
                    source=class_path,
                    on_error=on_error,
                )
                continue

            module_ref = entry.get("module")
            if module_ref is None:
                raise ValueError("Strategy plugin entry must contain either 'class_path' or 'module'.")
            mod = _load_module(str(module_ref))

            if "class" not in entry:
                reg = _merge_registry(reg, _registry_from_module(mod), source=str(module_ref), on_error=on_error)
                continue

            cls_name = str(entry["class"]).strip()
            cls = getattr(mod, cls_name)
            if not inspect.isclass(cls) or not issubclass(cls, Strategy):
                raise TypeError(f"module/class is not a Strategy subclass: {module_ref}.{cls_name}")
            name = str(entry.get("name") or _strategy_name_for_class(cls)).strip()
            reg = _merge_registry(
                reg,
                {name: _build_ctor(cls, strategy_name=name, init_kwargs=init_kwargs)},
                source=f"{module_ref}.{cls_name}",
                on_error=on_error,
            )
        except Exception as exc:
            msg = f"Failed to load strategy plugin entry {entry!r}: {exc}"
            if on_error == "raise":
                raise RuntimeError(msg) from exc
            LOGGER.warning(msg)
    return reg


def build_strategy_registry(
    plugin_dirs: list[str | Path] | None = None,
    plugin_specs: list[Any] | None = None,
    on_plugin_error: str = "raise",
    include_defaults: bool = True,
) -> dict[str, StrategyCtor]:
    """Build strategy registry from built-ins plus optional plugin discovery/loading."""
    registry = default_strategy_registry() if include_defaults else {}
    if plugin_dirs:
        registry = _merge_registry(
            registry,
            discover_strategy_registry(plugin_dirs=plugin_dirs, on_error=on_plugin_error),
            source="plugin_dirs",
            on_error=on_plugin_error,
        )
    if plugin_specs:
        registry = _merge_registry(
            registry,
            load_strategy_plugins(plugin_specs=plugin_specs, on_error=on_plugin_error),
            source="plugin_specs",
            on_error=on_plugin_error,
        )
    return registry

