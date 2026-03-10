"""策略目录、元数据与定义导出工具。"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from factorlab.strategies.base import Strategy, StrategyDefinition

StrategyCtor = Callable[[], Strategy]


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def strategy_definition_from_instance(strategy: Strategy) -> StrategyDefinition:
    """从已实例化策略提取定义信息。"""
    cls = type(strategy)
    if is_dataclass(strategy):
        params = {
            item.name: _jsonable(getattr(strategy, item.name))
            for item in fields(strategy)
            if item.name != "name"
        }
    else:
        params = {
            key: _jsonable(value)
            for key, value in vars(strategy).items()
            if key != "name"
        }
    module = f"{cls.__module__}.{cls.__name__}"
    return StrategyDefinition(
        name=str(strategy.name),
        family=str(getattr(cls, "STRATEGY_FAMILY", "custom")).strip() or "custom",
        description=str(getattr(cls, "STRATEGY_DESCRIPTION", f"Strategy implementation {module}.")).strip(),
        constraints=_as_tuple(getattr(cls, "STRATEGY_CONSTRAINTS", ())),
        tags=_as_tuple(getattr(cls, "STRATEGY_TAGS", ())),
        parameters=params,
        implementation=module,
        origin="builtin" if cls.__module__.startswith("factorlab.") else "plugin",
    )


def describe_strategy_registry(
    registry: dict[str, StrategyCtor],
    names: list[str] | None = None,
) -> list[StrategyDefinition]:
    """描述策略注册表。"""
    selected = list(names) if names is not None else sorted(registry.keys())
    return [strategy_definition_from_instance(registry[name]()) for name in selected if name in registry]


def strategy_definitions_frame(definitions: list[StrategyDefinition]) -> pd.DataFrame:
    """将定义列表展开为可落盘 DataFrame。"""
    return pd.DataFrame(
        [
            {
                "name": item.name,
                "family": item.family,
                "origin": item.origin,
                "constraints": ", ".join(item.constraints),
                "tags": ", ".join(item.tags),
                "description": item.description,
                "parameters_json": json.dumps(item.parameters, ensure_ascii=False, sort_keys=True),
                "implementation": item.implementation,
            }
            for item in definitions
        ],
        columns=[
            "name",
            "family",
            "origin",
            "constraints",
            "tags",
            "description",
            "parameters_json",
            "implementation",
        ],
    )


def write_strategy_definition_artifacts(
    *,
    out_dir: str | Path,
    definitions: list[StrategyDefinition],
) -> dict[str, Path]:
    """将策略定义写入 canonical overview 目录。"""
    root = Path(out_dir) / "tables" / "overview"
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "strategy_definitions.csv"
    json_path = root / "strategy_definitions.json"
    strategy_definitions_frame(definitions).to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            [
                {
                    "name": item.name,
                    "family": item.family,
                    "origin": item.origin,
                    "constraints": list(item.constraints),
                    "tags": list(item.tags),
                    "description": item.description,
                    "parameters": item.parameters,
                    "implementation": item.implementation,
                }
                for item in definitions
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "strategy_definitions_csv": csv_path,
        "strategy_definitions_json": json_path,
    }
