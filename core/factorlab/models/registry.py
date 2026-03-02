"""Model registry for factor modeling experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

try:  # pragma: no cover - optional dependency path
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional dependency path
    LGBMRegressor = None


class ModelRegistry:
    """Factory + persistence registry for ML models."""

    _defaults: dict[str, dict[str, Any]] = {
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
    }

    @classmethod
    def create(cls, name: str, params: dict[str, Any] | None = None):
        key = str(name).strip().lower()
        if key not in cls._defaults:
            raise KeyError(f"Unknown model: {name}. Available: {list(cls._defaults)}")

        cfg = dict(cls._defaults[key])
        if params:
            cfg.update(params)

        if key == "ridge":
            return Ridge(**cfg)
        if key == "rf":
            return RandomForestRegressor(**cfg)
        if key == "mlp":
            return MLPRegressor(**cfg)
        if key == "lgbm":
            if LGBMRegressor is None:
                raise ImportError(
                    "Model 'lgbm' requested but lightgbm is not installed. "
                    "Install lightgbm or choose ridge/rf/mlp."
                )
            return LGBMRegressor(**cfg)
        raise KeyError(f"Unknown model: {name}. Available: {list(cls._defaults)}")

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
