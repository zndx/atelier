"""WRK occupancy intent for Scheduler/RequestQueueShare."""

from __future__ import annotations

from concurrent import futures

import grpc
import pytest

from atelier.engine.queue_share import (
    GURU_SHAREFAIL,
    GPU_KEY,
    INSTRUCT,
    PEER,
    list_queue_share_requests,
    notify_admit,
    request_queue_share,
    share_for_class,
)
from atelier.engine.s2s import declared_queues, local_response
from zndx.engine.v1 import engine_pb2 as zpb
from zndx.scheduler.v1 import scheduler_pb2 as spb
from zndx.scheduler.v1 import scheduler_pb2_grpc as spbg


def test_instruct_share_requests_gpu_floor_4() -> None:
    req = share_for_class("instruct", INSTRUCT)
    assert req.peer == PEER
    assert req.workloads[0].wrk == "instruct"
    assert req.workloads[0].queue == INSTRUCT.queue
    assert req.shares[0].queue == INSTRUCT.queue
    assert req.shares[0].guaranteed.quantities[GPU_KEY] == 4
    assert req.shares[0].max.quantities[GPU_KEY] == 4
    assert req.shares[0].max_applications == 1


def test_referee_occupies_instruct_leaf() -> None:
    req = share_for_class("referee", INSTRUCT)
    assert req.workloads[0].wrk == "referee"
    assert req.workloads[0].queue == INSTRUCT.queue
    assert req.workloads[0].resource_class == INSTRUCT.name


def test_queue_hint_is_declared_shape_not_occupancy() -> None:
    hints = declared_queues()
    assert len(hints) == 1
    h = hints[0]
    assert h.path == INSTRUCT.queue
    assert h.gpu_guarantee == 0
    assert h.gpu_max == 4
    q = local_response(zpb.SERVER_QUERY_KIND_QUEUES)
    assert [x.path for x in q.queues] == [INSTRUCT.queue]


def test_module_never_writes_queues_yaml() -> None:
    from pathlib import Path

    src = Path("src/atelier/engine/queue_share.py").read_text()
    assert "do not write queues.yaml" in src
    assert "open(" not in src
    assert "write_text" not in src


def _serve(servicer) -> tuple[grpc.Server, str]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    spbg.add_SchedulerServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, f"127.0.0.1:{port}"


def test_unimplemented_is_signals_not_yet(monkeypatch) -> None:
    server, addr = _serve(spbg.SchedulerServicer())
    monkeypatch.setenv("SIGNALS_ENGINE_TARGET", addr)
    try:
        assert request_queue_share("instruct", INSTRUCT) is False
        assert list_queue_share_requests() == []
    finally:
        server.stop(grace=0)


class _Reject(spbg.SchedulerServicer):
    def RequestQueueShare(self, request, context):
        return spb.QueueShareResponse(
            accepted=False, request_id=request.request_id, error="cannot persist"
        )


def test_rejected_persist_is_sharefail(monkeypatch) -> None:
    server, addr = _serve(_Reject())
    monkeypatch.setenv("SIGNALS_ENGINE_TARGET", addr)
    try:
        with pytest.raises(RuntimeError, match=r"#YK\.00000007\.SHAREFAIL"):
            request_queue_share("instruct", INSTRUCT)
    finally:
        server.stop(grace=0)


class _Unavailable(spbg.SchedulerServicer):
    def RequestQueueShare(self, request, context):
        context.abort(grpc.StatusCode.UNAVAILABLE, "engine down")

    def ListQueueShareRequests(self, request, context):
        context.abort(grpc.StatusCode.UNAVAILABLE, "engine down")


def test_other_grpc_error_is_sharefail(monkeypatch) -> None:
    server, addr = _serve(_Unavailable())
    monkeypatch.setenv("SIGNALS_ENGINE_TARGET", addr)
    try:
        with pytest.raises(RuntimeError, match=r"#YK\.00000007\.SHAREFAIL"):
            request_queue_share("instruct", INSTRUCT)
        with pytest.raises(RuntimeError, match=r"#YK\.00000007\.SHAREFAIL"):
            list_queue_share_requests()
    finally:
        server.stop(grace=0)


class _Record(spbg.SchedulerServicer):
    def __init__(self) -> None:
        self.seen = []

    def RequestQueueShare(self, request, context):
        self.seen.append(request)
        return spb.QueueShareResponse(
            accepted=True,
            request_id=request.request_id,
            state=spb.QUEUE_SHARE_RECORDED,
        )

    def ListQueueShareRequests(self, request, context):
        recs = [
            spb.QueueShareRecord(request=r, recorded_at_ns=1, state=spb.QUEUE_SHARE_RECORDED)
            for r in self.seen
        ]
        return spb.ListQueueShareRequestsResponse(records=recs)


def test_recorded_and_list(monkeypatch) -> None:
    svc = _Record()
    server, addr = _serve(svc)
    monkeypatch.setenv("SIGNALS_ENGINE_TARGET", addr)
    try:
        assert request_queue_share("referee", INSTRUCT) is True
        recs = list_queue_share_requests()
        assert len(recs) == 1
        assert recs[0].request.workloads[0].wrk == "referee"
    finally:
        server.stop(grace=0)


def test_notify_admit_propagates_sharefail(monkeypatch) -> None:
    server, addr = _serve(_Reject())
    monkeypatch.setenv("SIGNALS_ENGINE_TARGET", addr)
    try:
        with pytest.raises(RuntimeError, match="SHAREFAIL"):
            notify_admit("instruct")
    finally:
        server.stop(grace=0)


def test_notify_admit_unimplemented_ok(monkeypatch) -> None:
    server, addr = _serve(spbg.SchedulerServicer())
    monkeypatch.setenv("SIGNALS_ENGINE_TARGET", addr)
    try:
        assert notify_admit("instruct") is False
    finally:
        server.stop(grace=0)
