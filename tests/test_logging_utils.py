"""相关功能测试。"""

from __future__ import annotations

from pathlib import Path

from factorlab.utils import configure_logging, get_logger


def test_configure_logging_writes_file(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    configure_logging(level="INFO", log_file=log_path, force=True)
    logger = get_logger("factorlab.test.logging")
    logger.info("logging smoke message")
    text = log_path.read_text(encoding="utf-8")
    assert "logging smoke message" in text
    assert "factorlab.test.logging" in text

