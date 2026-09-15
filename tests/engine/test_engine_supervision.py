"""Atelier EngineSupervision: Subscribe, Hello, empty replay, project mismatch."""
from __future__ import annotations

from atelier.engine.supervision_bus import init_bus
from atelier.engine.supervision_servicer import PROJECT, EngineSupervisionServicer
from zndx.supervision.v1 import supervision_pb2 as sv


class _Ctx:
    def __init__(self, active: int = 2):
        self._n = 0
        self._active = active
        self.aborted = None

    def is_active(self) -> bool:
        self._n += 1
        return self._n <= self._active

    def abort(self, code, details):
        self.aborted = (code, details)
        raise RuntimeError(f"{code} {details}")


def _collect(sub: sv.Subscribe, extra=None) -> list[sv.EngineEvent]:
    init_bus()
    svc = EngineSupervisionServicer()

    def msgs():
        yield sv.SupervisorMessage(subscribe=sub)
        if extra:
            yield extra

    return list(svc.Supervise(msgs(), _Ctx(active=1)))


def test_subscribe_hello_and_replay_complete():
    events = _collect(sv.Subscribe(supervisor_id="nautilus:atelier@test", project="atelier"))
    kinds = [ev.WhichOneof("event") for ev in events]
    assert "hello" in kinds
    assert "replay_complete" in kinds
    hello = next(ev.hello for ev in events if ev.HasField("hello"))
    assert hello.project == PROJECT
    assert "replay:ring" in list(hello.capabilities)
    assert "directive:observe_only" in list(hello.capabilities)
    rc = next(ev.replay_complete for ev in events if ev.HasField("replay_complete"))
    assert rc.clamped is False


def test_project_mismatch_goodbye():
    events = _collect(sv.Subscribe(supervisor_id="x", project="gaius"))
    assert events
    assert events[0].HasField("goodbye")
    assert events[0].goodbye.reason == sv.GOODBYE_REASON_PROJECT_MISMATCH


def test_backlog_empty():
    init_bus()
    svc = EngineSupervisionServicer()
    resp = svc.Backlog(sv.BacklogRequest(), _Ctx())
    assert resp.project == PROJECT
    assert list(resp.rows) == []


def test_first_message_must_be_subscribe():
    init_bus()
    svc = EngineSupervisionServicer()

    def msgs():
        yield sv.SupervisorMessage(heartbeat=sv.SupervisorHeartbeat(at_unix_ms=1))

    ctx = _Ctx()
    try:
        list(svc.Supervise(msgs(), ctx))
    except RuntimeError as e:
        assert "NOSUBSCRIBE" in str(e)
    else:
        raise AssertionError("expected NOSUBSCRIBE")
