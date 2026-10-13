"""The owner starts Metaflow from a RUNNING activity and releases it."""

from __future__ import annotations

import inspect
import sys
import time
from concurrent import futures

import grpc
import pytest

from atelier.engine.activity_owner import ActivityOwner, flow_command
from atelier.engine.flow_processes import flow_processes
from atelier.engine.workload_catalog import KIND, PEER
from atelier.flows.execution_topology import (
    classify_topology,
    configuration_from_reason,
    reason_for,
)
from zndx.scheduler.v1 import scheduler_pb2 as spb
from zndx.scheduler.v1 import scheduler_pb2_grpc as spbg


class _Proc:
    def __init__(self, cmd: list[str]) -> None:
        self.cmd = cmd
        self.pid = None
        self._code: int | None = None

    def poll(self) -> int | None:
        return self._code

    def finish(self, code: int) -> None:
        self._code = code


class _Signals(spbg.SchedulerServicer):
    def __init__(self, activities=None, *, accept_release=True, unimplemented=False):
        self.activities = list(activities or [])
        self.accept_release = accept_release
        self.unimplemented = unimplemented
        self.listed = 0
        self.renewed: list = []
        self.released: list = []

    def ListActivities(self, request, context):
        self.listed += 1
        if self.unimplemented:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "no ListActivities")
        assert request.peer == PEER
        assert request.kind == KIND
        assert request.active_only is True
        return spb.ListActivitiesResponse(activities=self.activities)

    def RenewActivity(self, request, context):
        self.renewed.append(request)
        return spb.ActivityResponse(accepted=True)

    def ReleaseActivity(self, request, context):
        self.released.append(request)
        if not self.accept_release:
            return spb.ActivityResponse(accepted=False, error="busy")
        return spb.ActivityResponse(accepted=True)


def _serve(servicer):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    spbg.add_SchedulerServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, f"127.0.0.1:{port}"


def _activity(activity_id: str, *, state: int, reason: str, horizon_ns: int):
    from zndx.engine.v1 import engine_pb2 as zpb

    return zpb.Activity(
        activity_id=activity_id,
        kind=KIND,
        peer=PEER,
        state=state,
        reason=reason,
        horizon_ns=horizon_ns,
    )


def _reason() -> str:
    return reason_for(classify_topology(
        source_id="sdg-corpora/pin_macbook",
        target="collection:clinic",
    ))


def _owner(addr: str, procs: list[_Proc]) -> ActivityOwner:
    def _popen(cmd):
        proc = _Proc(list(cmd))
        procs.append(proc)
        return proc

    return ActivityOwner(addr, popen=_popen, interval_s=3600)


def test_running_activity_starts_the_declared_flow_and_renews() -> None:
    from zndx.engine.v1 import engine_pb2 as zpb

    aid = "act-owner-run"
    horizon = time.time_ns() + 3_600_000_000_000
    sig = _Signals([_activity(
        aid, state=zpb.ACTIVITY_RUNNING, reason=_reason(), horizon_ns=horizon,
    )])
    server, addr = _serve(sig)
    procs: list[_Proc] = []
    owner = _owner(addr, procs)
    try:
        spawned = owner.poll_once()
        assert spawned == [aid]
        again = owner.poll_once()
        assert again == []
        assert len(procs) == 1
        cmd = procs[0].cmd
        assert cmd[0] == sys.executable
        assert cmd[1:4] == ["-m", "atelier.flows.run_classify", "run"]
        assert "--source-id=sdg-corpora/pin_macbook" in cmd
        assert "--target=collection:clinic" in cmd
        blob = next(a for a in cmd if a.startswith("--run-configuration="))
        conf = configuration_from_reason(blob.split("=", 1)[1])
        assert conf["sweep_capability"] == "decisions"
        assert conf["sweep_model"] == "vllm-sr/d3-flash"
        assert conf["runner"] == "metaflow"
        assert "--with" not in cmd
        assert len(sig.renewed) == 2
        assert sig.renewed[0].activity_id == aid
        assert sig.renewed[0].horizon_ns == horizon
        assert flow_processes().get(aid) is not None
    finally:
        flow_processes().unregister(aid)
        server.stop(0)


def test_queued_or_unreadable_activity_does_not_start() -> None:
    from zndx.engine.v1 import engine_pb2 as zpb

    horizon = time.time_ns() + 3_600_000_000_000
    sig = _Signals([
        _activity(
            "act-queued", state=zpb.ACTIVITY_QUEUED,
            reason=_reason(), horizon_ns=horizon,
        ),
        _activity(
            "act-bad", state=zpb.ACTIVITY_RUNNING,
            reason="a person clicked start", horizon_ns=horizon,
        ),
    ])
    server, addr = _serve(sig)
    procs: list[_Proc] = []
    owner = _owner(addr, procs)
    try:
        assert owner.poll_once() == []
        assert procs == []
        assert sig.renewed == []
    finally:
        server.stop(0)


def test_finished_child_is_released_and_not_started_again() -> None:
    from zndx.engine.v1 import engine_pb2 as zpb

    aid = "act-owner-done"
    horizon = time.time_ns() + 3_600_000_000_000
    sig = _Signals([_activity(
        aid, state=zpb.ACTIVITY_RUNNING, reason=_reason(), horizon_ns=horizon,
    )])
    server, addr = _serve(sig)
    procs: list[_Proc] = []
    owner = _owner(addr, procs)
    try:
        owner.poll_once()
        procs[0].finish(0)
        assert owner.poll_once() == []
        assert sig.released[0].outcome == "completed: metaflow exit 0"
        assert sig.released[0].peer == PEER
        assert flow_processes().get(aid) is None
        assert owner.poll_once() == []
        assert len(procs) == 1
    finally:
        flow_processes().unregister(aid)
        server.stop(0)


def test_failed_exit_releases_as_no_effect() -> None:
    from zndx.engine.v1 import engine_pb2 as zpb

    aid = "act-owner-fail"
    horizon = time.time_ns() + 3_600_000_000_000
    sig = _Signals([_activity(
        aid, state=zpb.ACTIVITY_RUNNING, reason=_reason(), horizon_ns=horizon,
    )])
    server, addr = _serve(sig)
    procs: list[_Proc] = []
    owner = _owner(addr, procs)
    try:
        owner.poll_once()
        procs[0].finish(2)
        owner.poll_once()
        assert sig.released[0].outcome == "failed: metaflow exit 2"
    finally:
        flow_processes().unregister(aid)
        server.stop(0)


def test_refused_release_does_not_start_a_second_flow() -> None:
    from zndx.engine.v1 import engine_pb2 as zpb

    aid = "act-owner-retry"
    horizon = time.time_ns() + 3_600_000_000_000
    sig = _Signals(
        [_activity(aid, state=zpb.ACTIVITY_RUNNING, reason=_reason(), horizon_ns=horizon)],
        accept_release=False,
    )
    server, addr = _serve(sig)
    procs: list[_Proc] = []
    owner = _owner(addr, procs)
    try:
        owner.poll_once()
        procs[0].finish(0)
        owner.poll_once()
        owner.poll_once()
        assert len(procs) == 1
        assert len(sig.released) == 2
        assert flow_processes().get(aid) is not None
    finally:
        flow_processes().unregister(aid)
        server.stop(0)


def test_unimplemented_list_starts_nothing() -> None:
    sig = _Signals(unimplemented=True)
    server, addr = _serve(sig)
    procs: list[_Proc] = []
    owner = _owner(addr, procs)
    try:
        assert owner.poll_once() == []
        assert owner.poll_once() == []
        assert procs == []
        assert sig.listed == 2
    finally:
        server.stop(0)


def test_flow_command_uses_the_activity_reason() -> None:
    cmd = flow_command(_reason())
    assert cmd[1:4] == ["-m", "atelier.flows.run_classify", "run"]
    conf = configuration_from_reason(
        next(a for a in cmd if a.startswith("--run-configuration=")).split("=", 1)[1]
    )
    assert conf["source_id"] == "sdg-corpora/pin_macbook"
    assert conf["target"] == "collection:clinic"
    with pytest.raises(ValueError, match="not the run configuration"):
        flow_command("not-json")


def test_serve_starts_and_stops_the_owner(monkeypatch) -> None:
    from atelier.engine.server import _start_activity_owner, serve

    started: dict[str, str] = {}

    def _init(target=None):
        started["target"] = target
        return "owner"

    monkeypatch.setattr(
        "atelier.engine.activity_owner.init_activity_owner", _init,
    )
    monkeypatch.setattr(
        "atelier.engine.workload_sync.signals_sync_target",
        lambda: "127.0.0.1:9",
    )
    assert _start_activity_owner() == "owner"
    assert started["target"] == "127.0.0.1:9"

    def _dark():
        raise RuntimeError("no contract")

    monkeypatch.setattr(
        "atelier.engine.workload_sync.signals_sync_target", _dark,
    )
    assert _start_activity_owner() is None
    src = inspect.getsource(serve)
    assert "owner = _start_activity_owner()" in src
    assert "owner.stop()" in src
