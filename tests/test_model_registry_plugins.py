"""相关功能测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from factorlab.models import ModelRegistry, build_model_registry


def _write_model_plugin(plugin_dir: Path) -> Path:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    path = plugin_dir / "my_model_plugin.py"
    path.write_text(
        """
from sklearn.linear_model import Ridge

MODEL_DEFAULTS = {
    "tiny_ridge": {"alpha": 0.2, "solver": "svd"}
}

def build_tiny_ridge_model(params):
    return Ridge(**params)
""".strip(),
        encoding="utf-8",
    )
    return path


def _write_duplicate_model_plugin(plugin_dir: Path) -> Path:
    plugin_dir.mkdir(parents=True, exist_ok=True)
    path = plugin_dir / "duplicate_model_plugin.py"
    path.write_text(
        """
from sklearn.linear_model import Ridge

MODEL_DEFAULTS = {
    "ridge": {"alpha": 999.0, "solver": "svd"}
}

def build_ridge_model(params):
    return Ridge(**params)
""".strip(),
        encoding="utf-8",
    )
    return path


def test_build_model_registry_with_plugin_dir(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    _write_model_plugin(plugin_dir)

    reg, defaults = build_model_registry(
        plugin_dirs=[plugin_dir],
        include_defaults=False,
        on_plugin_error="raise",
    )
    assert "tiny_ridge" in reg
    assert "tiny_ridge" in defaults
    model = reg["tiny_ridge"](dict(defaults["tiny_ridge"]))
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_model_registry_configure_plugins_supports_custom_model(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    _write_model_plugin(plugin_dir)

    ModelRegistry.reset_defaults()
    try:
        active = ModelRegistry.configure_plugins(
            plugin_dirs=[plugin_dir],
            include_defaults=True,
            on_plugin_error="raise",
        )
        assert "tiny_ridge" in active
        model = ModelRegistry.create("tiny_ridge")
        x = pd.DataFrame({"a": np.linspace(0.0, 1.0, 12), "b": np.linspace(1.0, 2.0, 12)})
        y = x["a"] * 0.3 + x["b"] * 0.1
        model.fit(x, y)
        pred = model.predict(x)
        assert len(pred) == len(x)
    finally:
        ModelRegistry.reset_defaults()


def test_model_registry_warn_skip_keeps_builtin_on_duplicate(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins"
    _write_duplicate_model_plugin(plugin_dir)

    ModelRegistry.reset_defaults()
    try:
        ModelRegistry.configure_plugins(
            plugin_dirs=[plugin_dir],
            include_defaults=True,
            on_plugin_error="warn_skip",
        )
        ridge = ModelRegistry.create("ridge")
        assert float(getattr(ridge, "alpha")) == 1.0
    finally:
        ModelRegistry.reset_defaults()
