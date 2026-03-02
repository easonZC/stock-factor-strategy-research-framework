"""插件预检工具测试。"""

from __future__ import annotations

import pytest

from factorlab.workflows.plugin_preflight import preflight_requested_components


def test_preflight_supports_alias_and_case_preserve() -> None:
    report = preflight_requested_components(
        kind="model",
        requested=["Random_Forest", "MLP"],
        available=["ridge", "rf", "mlp"],
        on_missing="raise",
        alias_map={"random_forest": "rf"},
    )
    assert report.resolved == ["rf", "mlp"]
    assert report.alias_hits["Random_Forest"] == "rf"


def test_preflight_warn_skip_for_missing_entries() -> None:
    report = preflight_requested_components(
        kind="transform",
        requested=["known", "unknown"],
        available=["known"],
        on_missing="warn_skip",
    )
    assert report.resolved == ["known"]
    assert report.missing == ["unknown"]


def test_preflight_raise_for_missing_entries() -> None:
    with pytest.raises(KeyError, match="strategy preflight"):
        preflight_requested_components(
            kind="strategy",
            requested=["custom_strategy"],
            available=["topk", "longshort"],
            on_missing="raise",
        )

