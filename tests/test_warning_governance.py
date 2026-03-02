"""模块说明。"""

from __future__ import annotations

import warnings

from factorlab.utils import summarize_captured_warnings


def test_warning_summary_classifies_benign_runtime_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("invalid value encountered in divide", RuntimeWarning)
    summary = summarize_captured_warnings(caught, logger_name="factorlab.test.warning")
    assert summary["total_count"] == 1
    assert summary["benign_count"] == 1
    assert summary["actionable_count"] == 0


def test_warning_summary_keeps_unknown_as_actionable() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("something unexpected happened", UserWarning)
    summary = summarize_captured_warnings(caught, logger_name="factorlab.test.warning")
    assert summary["total_count"] == 1
    assert summary["actionable_count"] == 1
    assert summary["actionable_samples"]

