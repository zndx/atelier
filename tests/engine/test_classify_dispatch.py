"""Callers of fsm_start treat a declaration as a declaration."""

from __future__ import annotations

import json

import pytest

from atelier.engine.activity_declare import DeclareError


def test_pipeline_task_records_a_declaration(monkeypatch) -> None:
    class _Dao:
        def __init__(self) -> None:
            raise RuntimeError("no db in this test")

    monkeypatch.setattr("atelier.db.dao.AtelierDao", _Dao)

    def _start(source_id=None):
        assert source_id == "sdg-corpora/pin_macbook"
        return {
            "started": False,
            "scheduled": True,
            "activity_id": "act-1",
            "clock": "airflow",
        }

    monkeypatch.setattr("atelier.gateway.fsm_start", _start)
    from atelier.task_handlers import _handle_pipeline

    result = _handle_pipeline({"source_id": "sdg-corpora/pin_macbook"})
    assert result.skipped is False
    assert result.outcome["scheduled"] is True
    assert result.outcome["started"] is False
    assert result.outcome["activity_id"] == "act-1"


def test_pipeline_task_fails_a_refused_declare(monkeypatch) -> None:
    class _Dao:
        def __init__(self) -> None:
            raise RuntimeError("no db in this test")

    monkeypatch.setattr("atelier.db.dao.AtelierDao", _Dao)
    monkeypatch.setattr(
        "atelier.gateway.fsm_start",
        lambda source_id=None: {
            "started": False, "scheduled": False, "error": "refused",
        },
    )
    from atelier.task_handlers import _handle_pipeline

    with pytest.raises(DeclareError, match="refused"):
        _handle_pipeline({"source_id": "sdg-corpora/pin_macbook"})


def test_apply_and_rerun_names_a_declaration(tmp_path, monkeypatch, capsys) -> None:
    proposal = {
        "run_id": "r1",
        "overlay": {},
        "rationale": "",
        "expected_effect": "",
        "trigger": "",
    }
    (tmp_path / "proposed_overlay.json").write_text(json.dumps(proposal))

    class _Cfg:
        overwatch_autonomy = "autonomous"

    monkeypatch.setattr("atelier.config.load_config", lambda: _Cfg())
    monkeypatch.setattr(
        "atelier.overwatch.apply_and_rerun._results_dir", lambda _run_id: tmp_path,
    )
    monkeypatch.setattr("atelier.config_overlay.set_overlay", lambda overlay: None)
    monkeypatch.setattr(
        "atelier.gateway.fsm_start",
        lambda source_id=None: {
            "started": False,
            "scheduled": True,
            "activity_id": "act-9",
        },
    )
    from atelier.overwatch.apply_and_rerun import main

    code = main(["r1", "--source-id", "sdg-corpora/pin_macbook"])
    assert code == 0
    body = json.loads(capsys.readouterr().out)
    assert body["status"] == "applied_and_scheduled"
    assert body["fsm_start_result"]["activity_id"] == "act-9"


def test_apply_and_rerun_refuses_a_dark_scheduler(tmp_path, monkeypatch, capsys) -> None:
    proposal = {"run_id": "r1", "overlay": {}}
    (tmp_path / "proposed_overlay.json").write_text(json.dumps(proposal))

    class _Cfg:
        overwatch_autonomy = "autonomous"

    monkeypatch.setattr("atelier.config.load_config", lambda: _Cfg())
    monkeypatch.setattr(
        "atelier.overwatch.apply_and_rerun._results_dir", lambda _run_id: tmp_path,
    )
    applied: list[dict] = []
    monkeypatch.setattr(
        "atelier.config_overlay.set_overlay", lambda overlay: applied.append(overlay),
    )
    monkeypatch.setattr(
        "atelier.gateway.fsm_start",
        lambda source_id=None: {
            "started": False, "scheduled": False, "error": "dark",
        },
    )
    from atelier.overwatch.apply_and_rerun import main

    assert main(["r1"]) == 5
    err = capsys.readouterr().err
    assert "dark" in err
    assert applied == [{}]
