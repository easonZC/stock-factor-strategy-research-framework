"""组合因子构建与正交化工具。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factorlab.preprocess import apply_cs_standardize
from factorlab.utils import get_logger

LOGGER = get_logger("factorlab.factors.combiner")

_ALLOWED_STANDARDIZATION = {"none", "cs_zscore", "cs_rank", "cs_robust_zscore"}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [value]


def _normalize_combination_entry(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise TypeError(f"Factor combination entry must be a dict, got {type(entry)}")

    name = str(entry.get("name", "")).strip()
    if not name:
        raise ValueError("Factor combination entry requires non-empty 'name'.")

    weights_map_raw = entry.get("weights")
    factor_weights: dict[str, float] = {}
    if isinstance(weights_map_raw, dict):
        for fac, w in weights_map_raw.items():
            fac_name = str(fac).strip()
            if not fac_name:
                continue
            factor_weights[fac_name] = float(w)
    else:
        factors = [str(x).strip() for x in _as_list(entry.get("factors")) if str(x).strip()]
        weights = _as_list(entry.get("weight_values"))
        if not factors:
            raise ValueError("Factor combination requires 'weights' dict or non-empty 'factors' list.")
        if not weights:
            weights = [1.0] * len(factors)
        if len(weights) != len(factors):
            raise ValueError(
                f"Factor combination '{name}' has mismatched factors/weight_values length: "
                f"{len(factors)} vs {len(weights)}."
            )
        factor_weights = {f: float(w) for f, w in zip(factors, weights)}

    if not factor_weights:
        raise ValueError(f"Factor combination '{name}' resolved empty factor weights.")

    standardization = str(entry.get("standardization", "none")).strip().lower()
    if standardization not in _ALLOWED_STANDARDIZATION:
        raise ValueError(
            f"Factor combination '{name}' invalid standardization='{standardization}'. "
            f"Allowed={sorted(_ALLOWED_STANDARDIZATION)}"
        )
    orthogonalize_to = [str(x).strip() for x in _as_list(entry.get("orthogonalize_to")) if str(x).strip()]

    return {
        "name": name,
        "weights": factor_weights,
        "standardization": standardization,
        "orthogonalize_to": orthogonalize_to,
    }


def normalize_factor_combinations(raw: Any, strict: bool = False) -> list[dict[str, Any]]:
    if raw is None:
        return []
    entries: list[Any]
    if isinstance(raw, dict):
        entries = [{**spec, "name": name} for name, spec in raw.items()]
    elif isinstance(raw, list):
        entries = raw
    else:
        if strict:
            raise TypeError(f"factor.combinations must be list/dict, got {type(raw)}")
        LOGGER.warning("Ignore invalid factor.combinations type: %s", type(raw))
        return []

    out: list[dict[str, Any]] = []
    for item in entries:
        try:
            out.append(_normalize_combination_entry(item))
        except Exception as exc:
            if strict:
                raise
            LOGGER.warning("Skip invalid factor combination entry %r: %s", item, exc)
    return out


def _orthogonalize_by_date(
    panel: pd.DataFrame,
    target_col: str,
    against_cols: list[str],
) -> pd.Series:
    out = pd.Series(np.nan, index=panel.index, dtype=float)
    if not against_cols:
        return pd.to_numeric(panel[target_col], errors="coerce").astype(float)

    cols = ["date", target_col, *against_cols]
    for _, grp in panel[cols].groupby("date"):
        g = grp.copy()
        for c in [target_col, *against_cols]:
            g[c] = pd.to_numeric(g[c], errors="coerce")
        g = g.dropna(subset=[target_col, *against_cols])
        if len(g) < len(against_cols) + 2:
            continue
        y = g[target_col].to_numpy(dtype=float)
        x = g[against_cols].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        out.loc[g.index] = resid
    return out


def apply_factor_combinations(
    panel: pd.DataFrame,
    combinations: list[dict[str, Any]],
    on_error: str = "raise",
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    out = panel.copy()
    computed: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    for spec in combinations:
        try:
            name = str(spec["name"])
            weights = dict(spec["weights"])
            standardization = str(spec.get("standardization", "none")).strip().lower()
            orth_cols = [str(x).strip() for x in spec.get("orthogonalize_to", []) if str(x).strip()]

            required = sorted(set(weights) | set(orth_cols))
            missing = [c for c in required if c not in out.columns]
            if missing:
                raise KeyError(f"missing required columns: {missing}")

            combo = pd.Series(0.0, index=out.index, dtype=float)
            for fac, w in weights.items():
                combo = combo + float(w) * pd.to_numeric(out[fac], errors="coerce").astype(float)

            combo_df = out[["date"]].copy()
            combo_df[name] = combo.astype(float)
            if orth_cols:
                tmp = out[["date", *orth_cols]].copy()
                tmp[name] = combo_df[name]
                combo_df[name] = _orthogonalize_by_date(tmp, target_col=name, against_cols=orth_cols)

            if standardization != "none":
                combo_df[name] = apply_cs_standardize(combo_df[["date", name]], col=name, method=standardization)

            out[name] = pd.to_numeric(combo_df[name], errors="coerce").astype(float)
            computed.append(name)
        except Exception as exc:
            msg = f"{spec!r}: {exc}"
            errors.append(msg)
            if on_error == "raise":
                raise RuntimeError(f"Factor combination failed for {name if 'name' in spec else spec}: {exc}") from exc
            skipped.append(str(spec.get("name", "unknown")))
            LOGGER.warning("Skip factor combination due to on_error=warn_skip: %s", msg)
    return out, computed, skipped, errors

