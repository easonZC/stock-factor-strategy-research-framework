"""Factor preflight, compute, definitions, and transform helpers."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from factorlab.factors import (
    FactorDefinition,
    apply_factor_combinations,
    apply_factor_expressions,
    apply_factors,
    describe_factor_registry,
    extract_expression_dependencies,
    factor_required_columns,
)
from factorlab.workflows.config_normalization import (
    as_dict as _as_dict,
    as_list as _as_list,
    is_forbidden_leakage_name as _is_forbidden_leakage_name,
    to_bool as _to_bool,
)
from factorlab.workflows.plugin_preflight import preflight_requested_components
from factorlab.utils import get_logger


LOGGER = get_logger("factorlab.workflows.config_runner")
PLUGIN_LOGGER_NAME = "factorlab.workflows.config_runner"


def resolve_required_fields(
    scope_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    factor_names: list[str],
    research_cfg: dict[str, Any],
    factor_registry: dict[str, Any],
) -> list[str]:
    required = set(str(x).strip() for x in data_cfg["fields_required"] if str(x).strip())
    required.update({"date", "close"})
    if scope_cfg["factor_scope"] == "cs":
        required.add("asset")
    required.update(factor_required_columns(factor_registry, factor_names))

    if scope_cfg["factor_scope"] == "cs":
        neutral = _as_dict(research_cfg.get("neutralize"))
        steps = {str(x).strip().lower() for x in _as_list(research_cfg.get("preprocess_steps")) if str(x).strip()}
        neutralize_in_pipeline = ("neutralize" in steps) if steps else True
        if _to_bool(neutral.get("enabled"), True) and neutralize_in_pipeline:
            mode = str(neutral.get("mode", "both")).strip().lower()
            if mode in {"size", "both"}:
                required.add("mkt_cap")
            if mode in {"industry", "both"}:
                required.add("industry")
    return sorted(required)


def filter_factors_by_available_columns(
    panel: pd.DataFrame,
    factor_names: list[str],
    on_missing: str,
    factor_registry: dict[str, Any],
) -> tuple[list[str], list[str]]:
    if not factor_names:
        return [], []
    if on_missing != "warn_skip":
        return factor_names, []

    selected: list[str] = []
    skipped: list[str] = []
    cols = set(panel.columns)
    required_map = {
        item.name: set(item.required_columns)
        for item in describe_factor_registry(factor_registry, names=factor_names)
    }
    for name in factor_names:
        required = required_map.get(name, set())
        missing = sorted(required - cols)
        if missing:
            LOGGER.warning(
                "Skip factor '%s': required input columns missing=%s and factor.on_missing=warn_skip",
                name,
                missing,
            )
            skipped.append(name)
            continue
        selected.append(name)
    return selected, skipped


def build_effective_factor_definitions(
    factor_names: list[str],
    factor_registry: dict[str, Any],
    expressions: dict[str, str],
    combinations: list[dict[str, Any]],
    panel_columns: list[str],
) -> list[FactorDefinition]:
    """Build the resolved factor definitions for the current run."""
    registry_defs = {
        item.name: item
        for item in describe_factor_registry(
            factor_registry,
            names=[name for name in factor_names if name in factor_registry],
        )
    }
    combination_by_name = {
        str(item.get("name")).strip(): item
        for item in combinations
        if str(item.get("name", "")).strip()
    }
    panel_col_set = set(str(col) for col in panel_columns)

    definitions: list[FactorDefinition] = []
    for name in factor_names:
        if name in registry_defs:
            definitions.append(registry_defs[name])
            continue
        if name in expressions:
            expr = str(expressions[name]).strip()
            deps = tuple(sorted(extract_expression_dependencies(expr)))
            definitions.append(
                FactorDefinition(
                    name=name,
                    family="expression",
                    description="Factor produced by expression composition in the current run.",
                    formula=expr,
                    required_columns=deps,
                    tags=("expression",),
                    parameters={"expression": expr},
                    implementation="factorlab.expression",
                    origin="expression",
                )
            )
            continue
        if name in combination_by_name:
            spec = combination_by_name[name]
            weights = spec.get("weights", {}) if isinstance(spec.get("weights"), dict) else {}
            orthogonalize_to = [str(x).strip() for x in spec.get("orthogonalize_to", []) if str(x).strip()]
            dependencies = tuple(sorted({str(k).strip() for k in weights if str(k).strip()} | set(orthogonalize_to)))
            definitions.append(
                FactorDefinition(
                    name=name,
                    family="combination",
                    description="Factor produced by weighted combination / orthogonalization in the current run.",
                    formula=f"weighted_combination({json.dumps(weights, ensure_ascii=False, sort_keys=True)})",
                    required_columns=dependencies,
                    tags=("combination",),
                    parameters={"weights": weights, "orthogonalize_to": orthogonalize_to},
                    implementation="factorlab.combiner",
                    origin="combination",
                )
            )
            continue
        if name in panel_col_set:
            definitions.append(
                FactorDefinition(
                    name=name,
                    family="input_column",
                    description="Factor provided directly by the input panel rather than computed by the registry.",
                    formula=name,
                    required_columns=(),
                    tags=("panel",),
                    parameters={},
                    implementation="input_panel",
                    origin="panel",
                )
            )
            continue
        definitions.append(
            FactorDefinition(
                name=name,
                family="unresolved",
                description="Factor requested in the run but no formal definition could be resolved.",
                formula="",
                required_columns=(),
                tags=("unresolved",),
                parameters={},
                implementation="unknown",
                origin="unknown",
            )
        )
    return definitions


def preflight_factor_candidates(
    factor_names: list[str],
    panel: pd.DataFrame,
    factor_registry: dict[str, Any],
    expression_outputs: set[str],
    combination_outputs: set[str],
    on_missing: str,
) -> dict[str, Any]:
    """Run unified factor candidate preflight."""
    forbidden = [x for x in factor_names if _is_forbidden_leakage_name(x)]
    requested = [x for x in factor_names if x not in set(forbidden)]
    available = sorted(set(str(x) for x in panel.columns) | set(factor_registry.keys()) | expression_outputs | combination_outputs)
    report = preflight_requested_components(
        kind="factor",
        requested=requested,
        available=available,
        on_missing=on_missing,
        logger_name=PLUGIN_LOGGER_NAME,
    )
    out = report.to_dict()
    out["skipped_forbidden_candidates"] = forbidden
    return out


def validate_required_fields(panel: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise KeyError(f"Data missing required fields: {missing}")


def compute_factors(
    panel: pd.DataFrame,
    factor_names: list[str],
    on_missing: str,
    registry: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = panel.copy()
    if not factor_names:
        return out, [], []
    missing = [f for f in factor_names if f not in out.columns]
    computable = [f for f in missing if f in registry]
    if computable:
        out = apply_factors(out, computable, inplace=True, registry=registry)
    unresolved = [f for f in factor_names if f not in out.columns]
    if unresolved:
        if on_missing == "warn_skip":
            LOGGER.warning("Skip unresolved factors due to factor.on_missing=warn_skip: %s", unresolved)
        else:
            raise KeyError(f"Factors missing and not computable: {unresolved}")
    selected = [f for f in factor_names if f not in unresolved]
    return out, computable, selected


def call_transform_fn(fn: Any, panel: pd.DataFrame, factor_col: str, kwargs: dict[str, Any]) -> pd.Series:
    try:
        value = fn(panel=panel, factor_col=factor_col, **kwargs)
    except TypeError:
        value = fn(panel, factor_col, **kwargs)

    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise TypeError(
                f"Transform output for '{factor_col}' must be a Series or single-column DataFrame, got shape={value.shape}."
            )
        value = value.iloc[:, 0]

    if not isinstance(value, pd.Series):
        if isinstance(value, np.ndarray):
            value = pd.Series(value, index=panel.index, dtype=float)
        else:
            raise TypeError(f"Transform output for '{factor_col}' must be a pandas Series, got {type(value)}.")

    if len(value) != len(panel):
        raise ValueError(
            f"Transform output length mismatch for '{factor_col}': expected {len(panel)} rows, got {len(value)}."
        )
    return value.reindex(panel.index)


def apply_custom_transforms(
    panel: pd.DataFrame,
    factor_names: list[str],
    transform_specs: list[dict[str, Any]],
    transform_registry: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = panel.copy()
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for fac in factor_names:
        if fac not in out.columns:
            errors.append({"factor": fac, "transform": None, "error": "factor column missing on panel before transform stage"})
            continue

        for spec in transform_specs:
            name = str(spec.get("name", "")).strip()
            kwargs = dict(spec.get("kwargs", {}) or {})
            on_error = str(spec.get("on_error", "raise")).strip().lower()
            fn = transform_registry.get(name)
            if fn is None:
                msg = f"custom transform '{name}' is not registered"
                if on_error == "warn_skip":
                    LOGGER.warning("Skip transform for factor '%s': %s", fac, msg)
                    skipped.append({"factor": fac, "transform": name, "reason": msg})
                    continue
                raise KeyError(msg)
            try:
                out[fac] = call_transform_fn(fn=fn, panel=out, factor_col=fac, kwargs=kwargs)
                applied.append({"factor": fac, "transform": name, "kwargs": kwargs})
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                if on_error == "warn_skip":
                    LOGGER.warning(
                        "Skip transform '%s' for factor '%s' due to on_error=warn_skip: %s",
                        name,
                        fac,
                        msg,
                    )
                    skipped.append({"factor": fac, "transform": name, "reason": msg})
                    continue
                raise

    return out, {"applied": applied, "skipped": skipped, "errors": errors}


def preflight_custom_transforms(
    transform_specs: list[dict[str, Any]],
    transform_registry: dict[str, Any],
) -> dict[str, Any]:
    """Check custom transforms resolve cleanly before execution."""
    policy_by_name: dict[str, str] = {}
    ordered_names: list[str] = []
    for spec in transform_specs:
        name = str(spec.get("name", "")).strip()
        if not name:
            continue
        on_error = str(spec.get("on_error", "raise")).strip().lower()
        prev = policy_by_name.get(name)
        cur = "warn_skip" if on_error == "warn_skip" else "raise"
        if prev is None:
            policy_by_name[name] = cur
            ordered_names.append(name)
        elif prev == "warn_skip" and cur == "raise":
            policy_by_name[name] = "raise"

    strict_names = [x for x in ordered_names if policy_by_name.get(x) == "raise"]
    warn_names = [x for x in ordered_names if policy_by_name.get(x) == "warn_skip"]
    available = sorted(transform_registry.keys())

    report_strict = preflight_requested_components(
        kind="transform",
        requested=strict_names,
        available=available,
        on_missing="raise",
        logger_name=PLUGIN_LOGGER_NAME,
    )
    report_warn = preflight_requested_components(
        kind="transform",
        requested=warn_names,
        available=available,
        on_missing="warn_skip",
        logger_name=PLUGIN_LOGGER_NAME,
    )
    return {
        "kind": "transform",
        "requested": ordered_names,
        "available": available,
        "resolved": [*report_strict.resolved, *[x for x in report_warn.resolved if x not in set(report_strict.resolved)]],
        "missing": [*report_strict.missing, *[x for x in report_warn.missing if x not in set(report_strict.missing)]],
        "on_missing": "mixed",
        "alias_hits": {**report_strict.alias_hits, **report_warn.alias_hits},
        "policy_by_name": policy_by_name,
    }


__all__ = [
    "apply_custom_transforms",
    "build_effective_factor_definitions",
    "compute_factors",
    "filter_factors_by_available_columns",
    "preflight_custom_transforms",
    "preflight_factor_candidates",
    "resolve_required_fields",
    "validate_required_fields",
]
