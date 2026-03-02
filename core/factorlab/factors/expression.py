"""Safe factor-expression utilities."""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd

from factorlab.utils import get_logger

LOGGER = get_logger("factorlab.factors.expression")

_ALLOWED_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}

_ALLOWED_UNARY_OPS = {
    ast.UAdd: lambda x: +x,
    ast.USub: lambda x: -x,
}
_ALLOWED_FUNCTIONS = {"abs", "log1p", "exp", "sqrt", "clip"}


def _to_series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(index)
    if np.isscalar(value):
        return pd.Series(float(value), index=index, dtype=float)
    arr = np.asarray(value)
    if arr.ndim != 1 or len(arr) != len(index):
        raise ValueError("Expression evaluation produced invalid shape.")
    return pd.Series(arr, index=index, dtype=float)


def _safe_call(name: str, args: list[Any]) -> Any:
    if name == "abs" and len(args) == 1:
        x = args[0]
        return x.abs() if hasattr(x, "abs") else np.abs(x)
    if name == "log1p" and len(args) == 1:
        return np.log1p(args[0])
    if name == "exp" and len(args) == 1:
        return np.exp(args[0])
    if name == "sqrt" and len(args) == 1:
        return np.sqrt(args[0])
    if name == "clip" and len(args) == 3:
        x, lo, hi = args
        if isinstance(x, pd.Series):
            return x.clip(lower=lo, upper=hi)
        return np.clip(x, lo, hi)
    raise ValueError(f"Unsupported function call: {name}")


def _validate_node(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _validate_node(node.body)
        return
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        return
    if isinstance(node, ast.Name):
        return
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _ALLOWED_BIN_OPS:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        _validate_node(node.left)
        _validate_node(node.right)
        return
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED_UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        _validate_node(node.operand)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed in factor expressions.")
        fn_name = node.func.id
        if fn_name not in _ALLOWED_FUNCTIONS:
            raise ValueError(f"Unsupported function call: {fn_name}")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported in factor expressions.")
        for arg in node.args:
            _validate_node(arg)
        return
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def validate_factor_expression(expression: str) -> set[str]:
    """Validate expression safety and return identifier dependencies."""
    expr = str(expression).strip()
    if not expr:
        raise ValueError("Expression cannot be empty.")
    tree = ast.parse(expr, mode="eval")
    _validate_node(tree)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    function_names = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    return names - function_names


def _eval_node(node: ast.AST, context: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, context)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    if isinstance(node, ast.Name):
        if node.id not in context:
            raise KeyError(f"Unknown identifier in expression: {node.id}")
        return context[node.id]
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BIN_OPS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        left = _eval_node(node.left, context)
        right = _eval_node(node.right, context)
        return _ALLOWED_BIN_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        val = _eval_node(node.operand, context)
        return _ALLOWED_UNARY_OPS[op_type](val)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only direct function calls are allowed in factor expressions.")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported in factor expressions.")
        fn_name = node.func.id
        args = [_eval_node(arg, context) for arg in node.args]
        return _safe_call(fn_name, args)
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def extract_expression_dependencies(expression: str) -> set[str]:
    """Extract identifier dependencies from expression string."""
    return validate_factor_expression(expression)


def evaluate_factor_expression(panel: pd.DataFrame, expression: str) -> pd.Series:
    """Evaluate one safe factor expression against panel columns."""
    deps = validate_factor_expression(expression)
    tree = ast.parse(str(expression).strip(), mode="eval")
    missing = [name for name in sorted(deps) if name not in panel.columns]
    if missing:
        raise KeyError(f"Missing expression dependencies: {missing}")

    context: dict[str, Any] = {}
    for col in deps:
        context[col] = pd.to_numeric(panel[col], errors="coerce")
    raw = _eval_node(tree, context)
    return _to_series(raw, panel.index).astype(float)


def apply_factor_expressions(
    panel: pd.DataFrame,
    expressions: dict[str, str],
    on_error: str = "raise",
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Compute expression factors and append columns to panel."""
    out = panel.copy()
    computed: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []
    mode = str(on_error).strip().lower()
    if mode not in {"raise", "warn_skip"}:
        raise ValueError("on_error must be one of: raise, warn_skip")

    for fac_name, expr in expressions.items():
        try:
            out[fac_name] = evaluate_factor_expression(out, expr)
            computed.append(fac_name)
        except Exception as exc:
            msg = f"{fac_name}: {exc}"
            if mode == "raise":
                raise RuntimeError(f"Failed to evaluate factor expression '{fac_name}': {exc}") from exc
            LOGGER.warning("Skip expression factor due to expression_on_error=warn_skip. %s", msg)
            skipped.append(fac_name)
            errors.append(msg)
    return out, computed, skipped, errors
