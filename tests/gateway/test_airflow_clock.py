"""Airflow clock: atelier restart does not execute classify."""

from __future__ import annotations

import pytest

from atelier import gateway
from atelier.config import AtelierConfig
from atelier.gateway import _federation_clock, _maybe_auto_start_classify


def test_default_clock_is_airflow() -> None:
    assert AtelierConfig().classify_clock == "airflow"
    assert _federation_clock(AtelierConfig())
    assert not _federation_clock(AtelierConfig(classify_clock="gateway"))


def test_airflow_clock_does_not_call_fsm_start(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def _spy(*, source_id=None, **kw):
        calls.append({"source_id": source_id, **kw})
        return {"run_id": "spy", "started": True}

    monkeypatch.setattr(gateway, "fsm_start", _spy)
    monkeypatch.setenv("ATELIER_CLASSIFY_CLOCK", "airflow")
    monkeypatch.setenv("ATELIER_CLASSIFY_AUTO_START", "true")
    monkeypatch.setenv("ATELIER_CLASSIFY_CONNECTION", "hive-poc")
    monkeypatch.setenv("ATELIER_CLASSIFY_DATABASE", "default")
    _maybe_auto_start_classify()
    assert calls == []
