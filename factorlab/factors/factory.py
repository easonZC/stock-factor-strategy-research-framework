"""因子工厂与注册辅助。"""

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

from factorlab.factors.base import Factor
from factorlab.factors.simple import (
    LiquidityShockFactor,
    MomentumFactor,
    SizeFactor,
    VolatilityFactor,
    VolumePricePressureFactor,
)
from factorlab.utils import get_logger

LOGGER = get_logger("factorlab.factors.factory")
FactorCtor = Callable[[], Factor]


def default_factor_registry() -> dict[str, FactorCtor]:
    return {
        "momentum_20": lambda: MomentumFactor(name="momentum_20", lookback=20),
        "volatility_20": lambda: VolatilityFactor(name="volatility_20", window=20),
        "liquidity_shock": lambda: LiquidityShockFactor(name="liquidity_shock", window=20),
        "volume_price_pressure_20": lambda: VolumePricePressureFactor(
            name="volume_price_pressure_20",
            window=20,
        ),
        "size": lambda: SizeFactor(name="size"),
    }


def _snake_case(name: str) -> str:
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", str(name))
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", text)
    return text.strip("_").lower()


def _load_module(module_ref: str) -> ModuleType:
    ref = str(module_ref).strip()
    path = Path(ref).expanduser()
    if path.exists() and path.is_file():
        module_name = f"factorlab_plugin_{abs(hash(str(path.resolve())))}"
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load plugin module from file: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module
    return importlib.import_module(ref)


def _iter_factor_classes(module: ModuleType) -> list[type[Factor]]:
    classes: list[type[Factor]] = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj is Factor:
            continue
        if not issubclass(obj, Factor):
            continue
        if inspect.isabstract(obj):
            continue
        if obj.__module__ != module.__name__:
            continue
        classes.append(obj)
    return classes


def _factor_name_for_class(cls: type[Factor]) -> str:
    explicit = getattr(cls, "FACTOR_NAME", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    name = cls.__name__
    if name.endswith("Factor"):
        name = name[:-6]
    return _snake_case(name)


def _build_ctor(cls: type[Factor], factor_name: str, init_kwargs: dict[str, Any] | None = None) -> FactorCtor:
    kwargs = dict(init_kwargs or {})
    sig = inspect.signature(cls)
    if "name" in sig.parameters and "name" not in kwargs:
        kwargs["name"] = factor_name

    def _ctor() -> Factor:
        fac = cls(**kwargs)
        if not isinstance(fac, Factor):
            raise TypeError(f"Expected Factor instance from {cls}, got {type(fac)}")
        return fac

    # 构造参数异常时提前失败，避免运行时再暴露。
    _ctor()
    return _ctor


def _registry_from_module(module: ModuleType) -> dict[str, FactorCtor]:
    reg: dict[str, FactorCtor] = {}
    exported = getattr(module, "FACTOR_REGISTRY", None)
    if isinstance(exported, dict):
        for name, ctor in exported.items():
            if isinstance(name, str) and callable(ctor):
                reg[name.strip()] = ctor

    fn = getattr(module, "get_factor_registry", None)
    if callable(fn):
        maybe = fn()
        if not isinstance(maybe, dict):
            raise TypeError("get_factor_registry() must return dict[str, Callable[[], Factor]]")
        for name, ctor in maybe.items():
            if not isinstance(name, str) or not callable(ctor):
                raise TypeError("get_factor_registry() returned invalid registry entry")
            reg[name.strip()] = ctor

    for cls in _iter_factor_classes(module):
        fac_name = _factor_name_for_class(cls)
        if fac_name in reg:
            continue
        reg[fac_name] = _build_ctor(cls, factor_name=fac_name)
    return reg


def _merge_registry(
    base: dict[str, FactorCtor],
    incoming: dict[str, FactorCtor],
    source: str,
    on_error: str,
) -> dict[str, FactorCtor]:
    out = dict(base)
    for name, ctor in incoming.items():
        if name in out:
            msg = f"Factor '{name}' already registered; plugin source={source}"
            if on_error == "raise":
                raise ValueError(msg)
            LOGGER.warning(msg)
            continue
        out[name] = ctor
    return out


def discover_factor_registry(plugin_dirs: list[str | Path], on_error: str = "raise") -> dict[str, FactorCtor]:
    reg: dict[str, FactorCtor] = {}
    for raw_dir in plugin_dirs:
        path = Path(raw_dir).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists() or not path.is_dir():
            msg = f"Plugin directory not found: {path}"
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
                msg = f"Failed to load plugin module '{file}': {exc}"
                if on_error == "raise":
                    raise RuntimeError(msg) from exc
                LOGGER.warning(msg)
    return reg


def load_factor_plugins(plugin_specs: list[Any], on_error: str = "raise") -> dict[str, FactorCtor]:
    """按模块/类规范加载因子插件。

    支持输入形式：
    - `"pkg.module"` 或 `"/abs/or/rel/path/plugin.py"`
    - `{"module": "pkg.module", "class": "MyFactor", "name": "my_factor", "init": {...}}`
    - `{"class_path": "pkg.module:MyFactor", "name": "my_factor", "init": {...}}`
    """
    reg: dict[str, FactorCtor] = {}
    for entry in plugin_specs:
        try:
            if isinstance(entry, str):
                mod = _load_module(entry)
                sub_reg = _registry_from_module(mod)
                reg = _merge_registry(reg, sub_reg, source=entry, on_error=on_error)
                continue

            if not isinstance(entry, dict):
                raise TypeError(f"Unsupported plugin entry type: {type(entry)}")

            init_kwargs = entry.get("init")
            if init_kwargs is not None and not isinstance(init_kwargs, dict):
                raise TypeError("Plugin entry 'init' must be a dict when provided.")

            if "class_path" in entry:
                class_path = str(entry["class_path"]).strip()
                module_ref, cls_name = class_path.split(":", 1) if ":" in class_path else class_path.rsplit(".", 1)
                mod = _load_module(module_ref)
                cls = getattr(mod, cls_name)
                if not inspect.isclass(cls) or not issubclass(cls, Factor):
                    raise TypeError(f"class_path is not a Factor subclass: {class_path}")
                factor_name = str(entry.get("name") or _factor_name_for_class(cls)).strip()
                reg = _merge_registry(
                    reg,
                    {factor_name: _build_ctor(cls, factor_name=factor_name, init_kwargs=init_kwargs)},
                    source=class_path,
                    on_error=on_error,
                )
                continue

            module_ref = entry.get("module")
            if module_ref is None:
                raise ValueError("Plugin entry must contain either 'class_path' or 'module'.")
            mod = _load_module(str(module_ref))
            if "class" not in entry:
                sub_reg = _registry_from_module(mod)
                reg = _merge_registry(reg, sub_reg, source=str(module_ref), on_error=on_error)
                continue

            cls_name = str(entry["class"]).strip()
            cls = getattr(mod, cls_name)
            if not inspect.isclass(cls) or not issubclass(cls, Factor):
                raise TypeError(f"module/class is not a Factor subclass: {module_ref}.{cls_name}")
            factor_name = str(entry.get("name") or _factor_name_for_class(cls)).strip()
            reg = _merge_registry(
                reg,
                {factor_name: _build_ctor(cls, factor_name=factor_name, init_kwargs=init_kwargs)},
                source=f"{module_ref}.{cls_name}",
                on_error=on_error,
            )
        except Exception as exc:
            msg = f"Failed to load factor plugin entry {entry!r}: {exc}"
            if on_error == "raise":
                raise RuntimeError(msg) from exc
            LOGGER.warning(msg)
    return reg


def build_factor_registry(
    plugin_dirs: list[str | Path] | None = None,
    plugin_specs: list[Any] | None = None,
    on_plugin_error: str = "raise",
) -> dict[str, FactorCtor]:
    registry = default_factor_registry()
    if plugin_dirs:
        registry = _merge_registry(
            registry,
            discover_factor_registry(plugin_dirs=plugin_dirs, on_error=on_plugin_error),
            source="plugin_dirs",
            on_error=on_plugin_error,
        )
    if plugin_specs:
        registry = _merge_registry(
            registry,
            load_factor_plugins(plugin_specs=plugin_specs, on_error=on_plugin_error),
            source="plugin_specs",
            on_error=on_plugin_error,
        )
    return registry


def apply_factors(
    panel: pd.DataFrame,
    factor_names: list[str],
    inplace: bool = True,
    registry: dict[str, FactorCtor] | None = None,
) -> pd.DataFrame:
    """计算指定因子并追加到面板数据。

    参数：
    - panel: 输入面板。
    - factor_names: 需要计算的因子名称列表。
    - inplace: 为 True 时原地写入，减少大表拷贝。
    - registry: 可选因子注册表；为空时使用内置注册表。
    """
    out = panel if inplace else panel.copy()
    reg = registry or default_factor_registry()
    for name in factor_names:
        if name not in reg:
            raise KeyError(f"Unknown factor '{name}'. Available: {list(reg)}")
        factor = reg[name]()
        out[factor.name] = factor.compute(out)
    return out
