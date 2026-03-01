"""Simple model registry for factor modeling experiments."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


class ModelRegistry:
    """Factory + persistence registry for scikit-learn models."""

    _factories = {
        "ridge": lambda: Ridge(alpha=1.0, random_state=None),
        "rf": lambda: RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            n_jobs=1,
        ),
    }

    @classmethod
    def create(cls, name: str):
        if name not in cls._factories:
            raise KeyError(f"Unknown model: {name}. Available: {list(cls._factories)}")
        return cls._factories[name]()

    @staticmethod
    def save(model, path: Path, model_name: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model_name": model_name, "model": model}
        joblib.dump(payload, path)
        return path

    @staticmethod
    def load(path: Path, model_name: str):
        payload = joblib.load(path)
        if payload.get("model_name") != model_name:
            raise ValueError(
                f"Model name mismatch: expected {model_name}, found {payload.get('model_name')}"
            )
        return payload["model"]
