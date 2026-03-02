"""模块说明。"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import pandas as pd

from factorlab.preprocess.transforms import apply_cs_standardize, ts_rolling_zscore
from factorlab.utils import get_logger

LOGGER = get_logger("factorlab.preprocess.factory")
TransformFn = Callable[..., pd.Series]


def _clip_transform(
    panel: pd.DataFrame,
    factor_col: str,
    lower: float | None = None,
    upper: float | None = None,
    by_date: bool = False,
) -> pd.Series:
    s = panel[factor_col].astype(float)
    lo = float(lower) if lower is not None else None
    hi = float(upper) if upper is not None else None
    if by_date and "date" in panel.columns:
        return (
            panel.assign(_factor=s)
            .groupby("date", group_keys=False)["_factor"]
            .apply(lambda x: x.clip(lower=lo, upper=hi))
        )
    return s.clip(lower=lo, upper=hi)


def _signed_log1p_transform(panel: pd.DataFrame, factor_col: str) -> pd.Series:
    s = panel[factor_col].astype(float)
    return np.sign(s) * np.log1p(np.abs(s))


def _ts_rolling_zscore_transform(
    panel: pd.DataFrame,
    factor_col: str,
    window: int = 60,
) -> pd.Series:
    required = {"asset", "date", factor_col}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise KeyError(f"ts_rolling_zscore transform requires columns: {missing}")
    return ts_rolling_zscore(panel[["asset", "date", factor_col]].copy(), col=factor_col, window=max(5, int(window)))


def _cs_rank_transform(panel: pd.DataFrame, factor_col: str) -> pd.Series:
    required = {"date", factor_col}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise KeyError(f"cs_rank transform requires columns: {missing}")
    return apply_cs_standardize(panel[["date", factor_col]].copy(), col=factor_col, method="cs_rank")


def _cs_zscore_transform(panel: pd.DataFrame, factor_col: str) -> pd.Series:
    required = {"date", factor_col}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise KeyError(f"cs_zscore transform requires columns: {missing}")
    return apply_cs_standardize(panel[["date", factor_col]].copy(), col=factor_col, method="cs_zscore")


def default_transform_registry() -> dict[str, TransformFn]:
    """中文说明。"""
    return {
        "clip": _clip_transform,
        "signed_log1p": _signed_log1p_transform,
        "ts_rolling_zscore": _ts_rolling_zscore_transform,
        "cs_rank": _cs_rank_transform,
        "cs_zscore": _cs_zscore_transform,
    }


def _snake_case(name: str) -> str:
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", str(name))
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", text)
    return text.strip("_").lower()


def _load_module(module_ref: str) -> ModuleType:
    ref = str(module_ref).strip()
    path = Path(ref).expanduser()
    if path.exists() and path.is_file():
        module_name = f"factorlab_transform_plugin_{abs(hash(str(path.resolve())))}"
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load transform plugin module from file: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module
    return importlib.import_module(ref)


def _iter_transform_functions(module: ModuleType) -> dict[str, TransformFn]:
    out: dict[str, TransformFn] = {}
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if obj.__module__ != module.__name__:
            continue
        if not name.startswith("transform_"):
            continue
        transform_name = _snake_case(name[len("transform_") :])
        if not transform_name:
            continue
        out[transform_name] = obj
    return out


def _registry_from_module(module: ModuleType) -> dict[str, TransformFn]:
    reg: dict[str, TransformFn] = {}
    exported = getattr(module, "TRANSFORM_REGISTRY", None)
    if isinstance(exported, dict):
        for name, fn in exported.items():
            if isinstance(name, str) and callable(fn):
                reg[name.strip()] = fn

    fn_get = getattr(module, "get_transform_registry", None)
    if callable(fn_get):
        maybe = fn_get()
        if not isinstance(maybe, dict):
            raise TypeError("get_transform_registry() must return dict[str, Callable]")
        for name, fn in maybe.items():
            if not isinstance(name, str) or not callable(fn):
                raise TypeError("get_transform_registry() returned invalid registry entry")
            reg[name.strip()] = fn

    auto = _iter_transform_functions(module)
    for name, fn in auto.items():
        reg.setdefault(name, fn)
    return reg


def _merge_registry(
    base: dict[str, TransformFn],
    incoming: dict[str, TransformFn],
    source: str,
    on_error: str,
) -> dict[str, TransformFn]:
    out = dict(base)
    for name, fn in incoming.items():
        if name in out:
            msg = f"Transform '{name}' already registered; plugin source={source}"
            if on_error == "raise":
                raise ValueError(msg)
            LOGGER.warning(msg)
            continue
        out[name] = fn
    return out


def discover_transform_registry(plugin_dirs: list[str | Path], on_error: str = "raise") -> dict[str, TransformFn]:
    """中文说明。"""
    reg: dict[str, TransformFn] = {}
    for raw_dir in plugin_dirs:
        path = Path(raw_dir).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists() or not path.is_dir():
            msg = f"Transform plugin directory not found: {path}"
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
                msg = f"Failed to load transform plugin module '{file}': {exc}"
                if on_error == "raise":
                    raise RuntimeError(msg) from exc
                LOGGER.warning(msg)
    return reg


def _load_callable_path(path_text: str) -> TransformFn:
    raw = str(path_text).strip()
    if ":" in raw:
        mod_ref, fn_name = raw.split(":", 1)
    else:
        mod_ref, fn_name = raw.rsplit(".", 1)
    mod = _load_module(mod_ref)
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise TypeError(f"callable_path is not callable: {raw}")
    return fn


def _resolve_plugin_entry(entry: dict[str, Any]) -> tuple[str, TransformFn]:
    name = str(entry.get("name", "")).strip()
    kwargs = entry.get("kwargs")
    if kwargs is not None and not isinstance(kwargs, dict):
        raise TypeError("Transform plugin entry 'kwargs' must be a dict when provided.")
    kwargs = dict(kwargs or {})

    if "callable_path" in entry:
        fn = _load_callable_path(str(entry["callable_path"]))
        resolved_name = name
        if not resolved_name:
            resolved_name = _snake_case(str(entry["callable_path"]).split(":")[-1].split(".")[-1])
        if kwargs:
            return resolved_name, lambda panel, factor_col, _fn=fn, _kw=kwargs: _fn(panel, factor_col, **_kw)
        return resolved_name, fn

    module_ref = entry.get("module")
    fn_name = entry.get("callable")
    if module_ref is None or fn_name is None:
        raise ValueError("Transform plugin entry must contain either 'callable_path' or ('module' + 'callable').")

    module = _load_module(str(module_ref))
    fn = getattr(module, str(fn_name))
    if not callable(fn):
        raise TypeError(f"module/callable is not callable: {module_ref}.{fn_name}")
    resolved_name = name or _snake_case(str(fn_name))
    if kwargs:
        return resolved_name, lambda panel, factor_col, _fn=fn, _kw=kwargs: _fn(panel, factor_col, **_kw)
    return resolved_name, fn


def load_transform_plugins(plugin_specs: list[Any], on_error: str = "raise") -> dict[str, TransformFn]:
    """中文说明。"""
    reg: dict[str, TransformFn] = {}
    for entry in plugin_specs:
        try:
            if isinstance(entry, str):
                mod = _load_module(entry)
                sub_reg = _registry_from_module(mod)
                reg = _merge_registry(reg, sub_reg, source=entry, on_error=on_error)
                continue
            if not isinstance(entry, dict):
                raise TypeError(f"Unsupported transform plugin entry type: {type(entry)}")
            name, fn = _resolve_plugin_entry(entry)
            reg = _merge_registry(reg, {name: fn}, source=str(entry), on_error=on_error)
        except Exception as exc:
            msg = f"Failed to load transform plugin entry {entry!r}: {exc}"
            if on_error == "raise":
                raise RuntimeError(msg) from exc
            LOGGER.warning(msg)
    return reg


def build_transform_registry(
    plugin_dirs: list[str | Path] | None = None,
    plugin_specs: list[Any] | None = None,
    on_plugin_error: str = "raise",
    include_defaults: bool = True,
) -> dict[str, TransformFn]:
    """中文说明。"""
    registry = default_transform_registry() if include_defaults else {}
    if plugin_dirs:
        registry = _merge_registry(
            registry,
            discover_transform_registry(plugin_dirs=plugin_dirs, on_error=on_plugin_error),
            source="plugin_dirs",
            on_error=on_plugin_error,
        )
    if plugin_specs:
        registry = _merge_registry(
            registry,
            load_transform_plugins(plugin_specs=plugin_specs, on_error=on_plugin_error),
            source="plugin_specs",
            on_error=on_plugin_error,
        )
    return registry
