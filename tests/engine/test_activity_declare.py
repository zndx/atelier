"""DeclareActivity carries the execution topology. No local pipeline."""

from __future__ import annotations

from concurrent import futures

import grpc
import pytest

from atelier.engine import activity_declare as ad
from atelier.engine.workload_catalog import DAG_ID, KIND
from atelier.flows.execution_topology import (
    classify_topology,
    configuration_from_reason,
)
from atelier.gateway import fsm_start
from zndx.scheduler.v1 import scheduler_pb2 as spb
from zndx.scheduler.v1 import scheduler_pb2_grpc as spbg


class _Signals(spbg.SchedulerServicer):
    def __init__(self, *, accept=True, unimplemented=False):
        self.requests: list = []
        self.accept = accept
        self.unimplemented = unimplemented

    def DeclareActivity(self, request, context):
        if self.unimplemented:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "no DeclareActivity")
        self.requests.append(request)
        if not self.accept:
            return spb.ActivityResponse(accepted=False, error="refused")
        return spb.ActivityResponse(accepted=True, activity=_activity(request))


def _activity(request):
    from zndx.engine.v1 import engine_pb2 as zpb

    return zpb.Activity(
        activity_id="act-test",
        kind=request.kind,
        peer=request.peer,
        dag_id=DAG_ID,
        run_id="manual__act-test",
        state=zpb.ACTIVITY_QUEUED,
        reason=request.reason,
    )


def _serve(servicer):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    spbg.add_SchedulerServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, f"127.0.0.1:{port}"


def test_declare_sends_the_topology_as_the_activity_reason() -> None:
    sig = _Signals()
    server, addr = _serve(sig)
    topo = classify_topology(source_id="sdg-corpora/pin_macbook", target="collection:clinic")
    try:
        result = ad.declare_topology(topo, target=addr)
    finally:
        server.stop(0)
    assert result["scheduled"] is True
    assert result["started"] is False
    assert result["activity_id"] == "act-test"
    assert result["dag_id"] == DAG_ID
    req = sig.requests[0]
    assert req.peer == "atelier"
    assert req.kind == KIND
    assert [(c.leaf, c.gpu) for c in req.claims] == list(topo.claims)
    assert list(req.precludes) == list(topo.precludes)
    echoed = configuration_from_reason(req.reason)
    assert echoed == result["configuration"]
    assert echoed["source_id"] == "sdg-corpora/pin_macbook"
    assert echoed["sweep_capability"] == "decisions"
    assert echoed["runner"] == "metaflow"
    assert echoed["trigger"] == "imperative"
    assert echoed["cron"] == ""


def test_refused_declare_does_not_look_like_a_started_run() -> None:
    server, addr = _serve(_Signals(accept=False))
    topo = classify_topology(source_id="sdg-corpora/pin_macbook")
    try:
        with pytest.raises(ad.DeclareError, match="refused"):
            ad.declare_topology(topo, target=addr)
    finally:
        server.stop(0)


def test_declare_failure_uses_a_free_coordination_code() -> None:
    assert ad.GURU_NODECLARE == "#CO.0000000A.NODECLARE"


def test_interpret_dispatch_keeps_a_declaration_distinct_from_a_thread() -> None:
    declared = ad.interpret_dispatch({
        "started": False, "scheduled": True, "activity_id": "act-1",
    })
    assert declared["status"] == "scheduled"
    assert declared["started"] is False
    started = ad.interpret_dispatch({"started": True, "run_id": "r"})
    assert started["status"] == "started"
    with pytest.raises(ad.DeclareError, match="refused"):
        ad.interpret_dispatch({
            "started": False, "scheduled": False, "error": "refused",
        })


def test_fsm_start_on_the_airflow_clock_declares_and_does_not_run(monkeypatch) -> None:
    calls: list[str] = []

    def _boom(*_a, **_k):
        calls.append("pipeline")
        raise AssertionError("in-process pipeline")

    def _declare(source_id):
        calls.append(source_id or "")
        return {"started": False, "scheduled": True, "clock": "airflow", "activity_id": "act-1"}

    monkeypatch.setattr(
        "atelier.classify.pipeline.run_classification_pipeline", _boom,
    )
    monkeypatch.setattr(ad, "schedule_classify", _declare)
    monkeypatch.setattr(
        "atelier.gateway._queue_refusal", lambda: None,
    )
    result = fsm_start(source_id="sdg-corpora/pin_macbook")
    assert result["scheduled"] is True
    assert result["started"] is False
    assert calls == ["sdg-corpora/pin_macbook"]
