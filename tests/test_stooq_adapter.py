"""Adapter behavior tests for public Stooq ingestion."""

from __future__ import annotations

from typing import Any

from factorlab.config import AdapterConfig
from factorlab.data.adapters import prepare_stooq_panel


class _DummyResponse:
    def __init__(self, payload: str) -> None:
        self._payload = payload.encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_DummyResponse":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


def test_stooq_adapter_basic(monkeypatch) -> None:
    payload = (
        "Date,Open,High,Low,Close,Volume\n"
        "2024-01-02,10,11,9,10.5,1000\n"
        "2024-01-03,10.5,11.2,10.1,10.9,1200\n"
        "2024-01-04,10.9,11.5,10.7,11.1,900\n"
    )

    def _fake_urlopen(url: str, timeout: int):  # noqa: ANN001
        assert "stooq.com" in url
        assert timeout == 20
        return _DummyResponse(payload)

    monkeypatch.setattr("factorlab.data.adapters.urlopen", _fake_urlopen)
    panel = prepare_stooq_panel(
        AdapterConfig(
            symbols=("aapl", "msft"),
            min_rows_per_asset=2,
            request_timeout_sec=20,
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
    )
    assert not panel.empty
    assert {"date", "asset", "close"}.issubset(panel.columns)
    assert set(panel["asset"].unique()) == {"AAPL", "MSFT"}

