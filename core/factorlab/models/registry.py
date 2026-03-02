"""模块说明。"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

try:  # pragma: no cover - optional dependency path
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional dependency path
    LGBMRegressor = None

try:  # pragma: no cover - optional dependency path
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency path
    XGBRegressor = None


ModelBuilder = Callable[[dict[str, Any]], Any]
LOGGER = logging.getLogger("factorlab.models.registry")


def _build_lgbm(cfg: dict[str, Any]):
    if LGBMRegressor is None:
        raise ImportError(
            "Model 'lgbm' requested but lightgbm is not installed. "
            "Install lightgbm or choose another model."
        )
    return LGBMRegressor(**cfg)


def _build_xgb(cfg: dict[str, Any]):
    if XGBRegressor is None:
        raise ImportError(
            "Model 'xgb' requested but xgboost is not installed. "
            "Install xgboost or choose another model."
        )
    return XGBRegressor(**cfg)


def default_model_defaults() -> dict[str, dict[str, Any]]:
    """内置模型默认参数。"""
    return {
        "ridge": {"alpha": 1.0, "solver": "svd"},
        "rf": {"n_estimators": 120, "max_depth": 6, "random_state": 42, "n_jobs": 1},
        "mlp": {"hidden_layer_sizes": (64, 32), "activation": "relu", "max_iter": 300, "random_state": 42},
        "lgbm": {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": -1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "n_jobs": 1,
        },
        "xgb": {
            "n_estimators": 220,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": 1,
            "objective": "reg:squarederror",
            "verbosity": 0,
        },
    }


def default_model_registry() -> dict[str, ModelBuilder]:
    """内置模型构造器注册表。"""
    return {
        "ridge": lambda cfg: Ridge(**cfg),
        "rf": lambda cfg: RandomForestRegressor(**cfg),
        "mlp": lambda cfg: MLPRegressor(**cfg),
        "lgbm": _build_lgbm,
        "xgb": _build_xgb,
    }


def _snake_case(name: str) -> str:
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", str(name))
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", text)
    return text.strip("_").lower()


def _load_module(module_ref: str) -> ModuleType:
    ref = str(module_ref).strip()
    path = Path(ref).expanduser()
    if path.exists() and path.is_file():
        module_name = f"factorlab_model_plugin_{abs(hash(str(path.resolve())))}"
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load model plugin module from file: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        return module
    return importlib.import_module(ref)


def _is_model_builder(obj: Any) -> bool:
    if not callable(obj):
        return False
    try:
        inspect.signature(obj)
    except Exception:
        return False
    return True


def _model_name_for_builder_name(name: str) -> str:
    raw = str(name).strip()
    if raw.startswith("build_") and raw.endswith("_model"):
        raw = raw[len("build_") : -len("_model")]
    return _snake_case(raw)


def _normalize_defaults_map(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, dict):
            out[k.strip().lower()] = dict(v)
    return out


def _registry_from_module(module: ModuleType) -> tuple[dict[str, ModelBuilder], dict[str, dict[str, Any]]]:
    reg: dict[str, ModelBuilder] = {}
    defaults: dict[str, dict[str, Any]] = {}

    exported = getattr(module, "MODEL_REGISTRY", None)
    if isinstance(exported, dict):
        for name, builder in exported.items():
            if isinstance(name, str) and _is_model_builder(builder):
                reg[name.strip().lower()] = builder

    fn_get = getattr(module, "get_model_registry", None)
    if callable(fn_get):
        maybe = fn_get()
        if not isinstance(maybe, dict):
            raise TypeError("get_model_registry() must return dict[str, Callable].")
        for name, builder in maybe.items():
            if not isinstance(name, str) or not _is_model_builder(builder):
                raise TypeError("get_model_registry() returned invalid registry entry.")
            reg[name.strip().lower()] = builder

    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if fn.__module__ != module.__name__:
            continue
        if not _is_model_builder(fn):
            continue
        if not (name.startswith("build_") and name.endswith("_model")):
            continue
        model_name = _model_name_for_builder_name(name)
        reg.setdefault(model_name, fn)

    defaults.update(_normalize_defaults_map(getattr(module, "MODEL_DEFAULTS", None)))
    get_defaults = getattr(module, "get_model_defaults", None)
    if callable(get_defaults):
        defaults.update(_normalize_defaults_map(get_defaults()))

    for model_name in reg:
        defaults.setdefault(model_name, {})
    return reg, defaults


def _merge_registry(
    base_reg: dict[str, ModelBuilder],
    base_defaults: dict[str, dict[str, Any]],
    incoming_reg: dict[str, ModelBuilder],
    incoming_defaults: dict[str, dict[str, Any]],
    source: str,
    on_error: str,
) -> tuple[dict[str, ModelBuilder], dict[str, dict[str, Any]]]:
    out_reg = dict(base_reg)
    out_defaults = {k: dict(v) for k, v in base_defaults.items()}
    for name, builder in incoming_reg.items():
        if name in out_reg:
            if on_error == "raise":
                raise ValueError(f"Model '{name}' already registered; plugin source={source}")
            LOGGER.warning(
                "Skip duplicate model plugin registration: name=%s source=%s",
                name,
                source,
            )
            continue
        out_reg[name] = builder
        if name in incoming_defaults:
            out_defaults[name] = dict(incoming_defaults[name])
        else:
            out_defaults.setdefault(name, {})
    return out_reg, out_defaults


def discover_model_registry(
    plugin_dirs: list[str | Path],
    on_error: str = "raise",
) -> tuple[dict[str, ModelBuilder], dict[str, dict[str, Any]]]:
    """从插件目录发现模型构造器。"""
    reg: dict[str, ModelBuilder] = {}
    defaults: dict[str, dict[str, Any]] = {}
    for raw_dir in plugin_dirs:
        path = Path(raw_dir).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists() or not path.is_dir():
            msg = f"Model plugin directory not found: {path}"
            if on_error == "raise":
                raise FileNotFoundError(msg)
            LOGGER.warning(msg)
            continue
        for file in sorted(path.glob("*.py")):
            if file.name.startswith("_"):
                continue
            try:
                mod = _load_module(str(file))
                sub_reg, sub_defaults = _registry_from_module(mod)
                reg, defaults = _merge_registry(
                    reg,
                    defaults,
                    sub_reg,
                    sub_defaults,
                    source=str(file),
                    on_error=on_error,
                )
            except Exception as exc:
                msg = f"Failed to load model plugin module '{file}': {exc}"
                if on_error == "raise":
                    raise RuntimeError(msg) from exc
                LOGGER.warning(msg)
    return reg, defaults


def _load_callable_from_path(path_text: str) -> ModelBuilder:
    raw = str(path_text).strip()
    module_ref, callable_name = raw.split(":", 1) if ":" in raw else raw.rsplit(".", 1)
    mod = _load_module(module_ref)
    fn = getattr(mod, callable_name)
    if not _is_model_builder(fn):
        raise TypeError(f"callable_path is not valid model builder: {raw}")
    return fn


def load_model_plugins(
    plugin_specs: list[Any],
    on_error: str = "raise",
) -> tuple[dict[str, ModelBuilder], dict[str, dict[str, Any]]]:
    """按显式规范加载模型插件。"""
    reg: dict[str, ModelBuilder] = {}
    defaults: dict[str, dict[str, Any]] = {}
    for entry in plugin_specs:
        try:
            if isinstance(entry, str):
                mod = _load_module(entry)
                sub_reg, sub_defaults = _registry_from_module(mod)
                reg, defaults = _merge_registry(
                    reg,
                    defaults,
                    sub_reg,
                    sub_defaults,
                    source=entry,
                    on_error=on_error,
                )
                continue

            if not isinstance(entry, dict):
                raise TypeError(f"Unsupported model plugin entry type: {type(entry)}")

            entry_defaults = dict(entry.get("defaults", {}) or {})
            if "callable_path" in entry:
                call_path = str(entry["callable_path"]).strip()
                builder = _load_callable_from_path(call_path)
                call_name = call_path.split(":")[-1].split(".")[-1]
                name = str(entry.get("name") or _model_name_for_builder_name(call_name)).strip().lower()
                reg, defaults = _merge_registry(
                    reg,
                    defaults,
                    {name: builder},
                    {name: entry_defaults},
                    source=call_path,
                    on_error=on_error,
                )
                continue

            module_ref = entry.get("module")
            if module_ref is None:
                raise ValueError("Model plugin entry must contain 'module' or 'callable_path'.")
            mod = _load_module(str(module_ref))
            if "callable" not in entry:
                sub_reg, sub_defaults = _registry_from_module(mod)
                reg, defaults = _merge_registry(
                    reg,
                    defaults,
                    sub_reg,
                    sub_defaults,
                    source=str(module_ref),
                    on_error=on_error,
                )
                continue

            callable_name = str(entry["callable"]).strip()
            builder = getattr(mod, callable_name)
            if not _is_model_builder(builder):
                raise TypeError(f"module/callable is not valid model builder: {module_ref}.{callable_name}")
            name = str(entry.get("name") or _model_name_for_builder_name(callable_name)).strip().lower()
            reg, defaults = _merge_registry(
                reg,
                defaults,
                {name: builder},
                {name: entry_defaults},
                source=f"{module_ref}.{callable_name}",
                on_error=on_error,
            )
        except Exception as exc:
            msg = f"Failed to load model plugin entry {entry!r}: {exc}"
            if on_error == "raise":
                raise RuntimeError(msg) from exc
            LOGGER.warning(msg)
    return reg, defaults


def build_model_registry(
    plugin_dirs: list[str | Path] | None = None,
    plugin_specs: list[Any] | None = None,
    on_plugin_error: str = "raise",
    include_defaults: bool = True,
) -> tuple[dict[str, ModelBuilder], dict[str, dict[str, Any]]]:
    """构建模型注册表（内置 + 插件）。"""
    registry = default_model_registry() if include_defaults else {}
    defaults = default_model_defaults() if include_defaults else {}
    if plugin_dirs:
        sub_reg, sub_defaults = discover_model_registry(plugin_dirs=plugin_dirs, on_error=on_plugin_error)
        registry, defaults = _merge_registry(
            registry,
            defaults,
            sub_reg,
            sub_defaults,
            source="plugin_dirs",
            on_error=on_plugin_error,
        )
    if plugin_specs:
        sub_reg, sub_defaults = load_model_plugins(plugin_specs=plugin_specs, on_error=on_plugin_error)
        registry, defaults = _merge_registry(
            registry,
            defaults,
            sub_reg,
            sub_defaults,
            source="plugin_specs",
            on_error=on_plugin_error,
        )
    return registry, defaults


def _instantiate_model(builder: ModelBuilder, cfg: dict[str, Any], model_name: str):
    """兼容不同 builder 签名（dict / **kwargs / 无参）。"""
    errors: list[str] = []
    for mode in ("dict", "kwargs", "zero"):
        try:
            if mode == "dict":
                return builder(cfg)
            if mode == "kwargs":
                return builder(**cfg)  # type: ignore[misc]
            return builder()  # type: ignore[misc]
        except TypeError as exc:
            errors.append(f"{mode}: {exc}")
            continue
    raise TypeError(
        f"Failed to instantiate model '{model_name}' via builder signature attempts. "
        f"errors={errors}"
    )


class ModelRegistry:
    """模型工厂 + 持久化注册中心。"""

    _defaults: dict[str, dict[str, Any]] = default_model_defaults()
    _builders: dict[str, ModelBuilder] = default_model_registry()

    @classmethod
    def reset_defaults(cls) -> None:
        """重置到内置模型注册表。"""
        cls._builders = default_model_registry()
        cls._defaults = default_model_defaults()

    @classmethod
    def register(
        cls,
        name: str,
        builder: ModelBuilder,
        defaults: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        key = str(name).strip().lower()
        if not key:
            raise ValueError("Model name cannot be empty.")
        if not _is_model_builder(builder):
            raise TypeError("builder must be callable.")
        if key in cls._builders and not overwrite:
            raise ValueError(f"Model '{key}' already exists.")
        cls._builders[key] = builder
        cls._defaults[key] = dict(defaults or {})

    @classmethod
    def configure_plugins(
        cls,
        plugin_dirs: list[str | Path] | None = None,
        plugin_specs: list[Any] | None = None,
        on_plugin_error: str = "raise",
        include_defaults: bool = True,
    ) -> dict[str, ModelBuilder]:
        """按插件配置重建当前活动模型注册表。"""
        reg, defaults = build_model_registry(
            plugin_dirs=plugin_dirs,
            plugin_specs=plugin_specs,
            on_plugin_error=on_plugin_error,
            include_defaults=include_defaults,
        )
        cls._builders = dict(reg)
        cls._defaults = {k: dict(defaults.get(k, {})) for k in reg}
        return dict(cls._builders)

    @classmethod
    def available_models(cls) -> list[str]:
        """返回当前活动模型名称列表。"""
        return sorted(cls._builders.keys())

    @classmethod
    def create(cls, name: str, params: dict[str, Any] | None = None):
        key = str(name).strip().lower()
        if key not in cls._builders:
            raise KeyError(f"Unknown model: {name}. Available: {cls.available_models()}")

        cfg = dict(cls._defaults.get(key, {}))
        if params:
            cfg.update(params)
        return _instantiate_model(cls._builders[key], cfg=cfg, model_name=key)

    @staticmethod
    def save(
        model,
        path: Path,
        model_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model_name": model_name, "model": model, "metadata": metadata or {}}
        joblib.dump(payload, path)
        return path

    @staticmethod
    def load(
        path: Path,
        model_name: str | None = None,
        return_metadata: bool = False,
    ):
        payload = joblib.load(path)
        if model_name is not None and payload.get("model_name") != model_name:
            raise ValueError(
                f"Model name mismatch: expected {model_name}, found {payload.get('model_name')}"
            )
        model = payload["model"]
        if return_metadata:
            return model, payload.get("metadata", {})
        return model
