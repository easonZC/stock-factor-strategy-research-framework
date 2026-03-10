"""因子目录、元数据与定义导出工具。"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from factorlab.factors.base import Factor, FactorDefinition

FactorCtor = Callable[[], Factor]


def _normalize_text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return tuple(out)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    return str(value)


def factor_definition_from_instance(factor: Factor) -> FactorDefinition:
    """从已实例化因子提取定义信息。"""
    cls = type(factor)
    params: dict[str, Any] = {}
    if is_dataclass(factor):
        for item in fields(factor):
            if item.name == "name":
                continue
            params[item.name] = _jsonable(getattr(factor, item.name))
    else:
        try:
            raw_items = vars(factor).items()
        except TypeError:
            raw_items = (
                (name, getattr(factor, name))
                for name in dir(factor)
                if not name.startswith("_") and name != "name" and not callable(getattr(factor, name))
            )
        for key, value in raw_items:
            if key == "name":
                continue
            params[str(key)] = _jsonable(value)

    module = f"{cls.__module__}.{cls.__name__}"
    origin = "builtin" if cls.__module__.startswith("factorlab.") else "plugin"
    description = str(
        getattr(
            cls,
            "FACTOR_DESCRIPTION",
            f"Factor implementation {module}.",
        )
    ).strip()
    formula = str(getattr(cls, "FACTOR_FORMULA", "")).strip()
    family = str(getattr(cls, "FACTOR_FAMILY", "custom")).strip() or "custom"
    required_columns = _normalize_text_tuple(getattr(cls, "REQUIRED_COLUMNS", ()))
    tags = _normalize_text_tuple(getattr(cls, "FACTOR_TAGS", ()))
    return FactorDefinition(
        name=str(factor.name),
        family=family,
        description=description,
        formula=formula,
        required_columns=required_columns,
        tags=tags,
        parameters=params,
        implementation=module,
        origin=origin,
    )


def describe_factor_registry(
    registry: dict[str, FactorCtor],
    names: list[str] | None = None,
) -> list[FactorDefinition]:
    """描述因子注册表。"""
    selected = list(names) if names is not None else sorted(registry.keys())
    definitions: list[FactorDefinition] = []
    for name in selected:
        ctor = registry.get(name)
        if ctor is None:
            continue
        definitions.append(factor_definition_from_instance(ctor()))
    return definitions


def factor_required_columns(
    registry: dict[str, FactorCtor],
    names: list[str],
) -> set[str]:
    """汇总给定因子的输入列要求。"""
    required: set[str] = set()
    for definition in describe_factor_registry(registry=registry, names=names):
        required.update(str(col) for col in definition.required_columns if str(col).strip())
    return required


def factor_definitions_frame(definitions: list[FactorDefinition]) -> pd.DataFrame:
    """将定义列表展开为可落盘 DataFrame。"""
    rows = []
    for item in definitions:
        rows.append(
            {
                "name": item.name,
                "family": item.family,
                "origin": item.origin,
                "required_columns": ", ".join(item.required_columns),
                "tags": ", ".join(item.tags),
                "description": item.description,
                "formula": item.formula,
                "parameters_json": json.dumps(item.parameters, ensure_ascii=False, sort_keys=True),
                "implementation": item.implementation,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "name",
            "family",
            "origin",
            "required_columns",
            "tags",
            "description",
            "formula",
            "parameters_json",
            "implementation",
        ],
    )


def write_factor_definition_artifacts(
    *,
    out_dir: str | Path,
    definitions: list[FactorDefinition],
) -> dict[str, Path]:
    """将因子定义写入 canonical overview 目录。"""
    root = Path(out_dir)
    overview_dir = root / "tables" / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)
    csv_path = overview_dir / "factor_definitions.csv"
    json_path = overview_dir / "factor_definitions.json"

    frame = factor_definitions_frame(definitions)
    frame.to_csv(csv_path, index=False)
    payload = [
        {
            "name": item.name,
            "family": item.family,
            "origin": item.origin,
            "required_columns": list(item.required_columns),
            "tags": list(item.tags),
            "description": item.description,
            "formula": item.formula,
            "parameters": item.parameters,
            "implementation": item.implementation,
        }
        for item in definitions
    ]
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "factor_definitions_csv": csv_path,
        "factor_definitions_json": json_path,
    }
