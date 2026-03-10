"""运行期告警捕获与汇总。"""

from __future__ import annotations

from collections import Counter
from typing import Any

from factorlab.utils.logging_utils import get_logger


def _warning_level(category_name: str, message: str) -> tuple[str, str]:
    text = str(message).strip().lower()
    cat = str(category_name).strip()

    if "constantinputwarning" in cat.lower() and "correlation coefficient is not defined" in text:
        return "benign", "constant_correlation"
    if "invalid value encountered in divide" in text:
        return "benign", "divide_by_zero_corr"
    if "linalgwarning" in cat.lower() and "ill-conditioned matrix" in text:
        return "benign", "ill_conditioned_linear_system"
    return "actionable", "unknown"


def summarize_captured_warnings(
    records: list[Any],
    logger_name: str = "factorlab",
    max_actionable_logs: int = 5,
) -> dict[str, Any]:
    logger = get_logger(logger_name)
    benign = Counter()
    actionable = Counter()
    actionable_samples: list[dict[str, str]] = []

    for rec in records:
        category_name = getattr(getattr(rec, "category", None), "__name__", "Warning")
        message = str(getattr(rec, "message", "")).strip()
        level, reason = _warning_level(category_name=category_name, message=message)
        key = f"{category_name}: {reason}"
        if level == "benign":
            benign[key] += 1
            continue
        actionable[key] += 1
        if len(actionable_samples) < int(max_actionable_logs):
            actionable_samples.append(
                {
                    "category": category_name,
                    "message": message,
                    "source": f"{getattr(rec, 'filename', '')}:{getattr(rec, 'lineno', '')}",
                }
            )

    if actionable_samples:
        logger.warning(
            "Actionable warnings captured=%s (showing up to %s).",
            int(sum(actionable.values())),
            int(max_actionable_logs),
        )
        for row in actionable_samples:
            logger.warning(
                "[%s] %s @ %s",
                row["category"],
                row["message"],
                row["source"],
            )

    return {
        "total_count": int(len(records)),
        "benign_count": int(sum(benign.values())),
        "actionable_count": int(sum(actionable.values())),
        "benign_breakdown": dict(benign),
        "actionable_breakdown": dict(actionable),
        "actionable_samples": actionable_samples,
    }

