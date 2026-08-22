"""WRK occupancy intent for Scheduler/RequestQueueShare."""

from __future__ import annotations

import uuid
from concurrent import futures

import grpc
import pytest

from atelier.engine.queue_share import (
    GURU_SHAREFAIL,
    GPU_KEY,
    HEAVY,
    LIGHT,
    MEDIUM,
    PEER,
    leaf_for_gpu_tokens,
    list_queue_share_requests,
    mint_uuid7,
    notify_admit,
    notify_release,
    request_queue_share,
    require_uuid7,
    share_for_class,
)
from atelier.engine.s2s import declared_queues, local_response
from zndx.engine.v1 import engine_pb2 as zpb
from zndx.scheduler.v1 import scheduler_pb2 as spb
from zndx.scheduler.v1 import scheduler_pb2_grpc as spbg


def test_uuidv7_required_never_v4() -> None:
    s = mint_uuid7()
    u = uuid.UUID(s)
    assert u.version == 7
    assert require_uuid7(s) == s
    with pytest.raises(RuntimeError, match=r"UUIDv7"):
        require_uuid7("")
    with pytest.raises(RuntimeError, match=r"not v4"):
        require_uuid7(str(uuid.uuid4()))


def test_share_mints_uuidv7_every_request() -> None:
    a = share_for_class("instruct", HEAVY)
    b = share_for_class("instruct", HEAVY)
    assert uuid.UUID(a.request_id).version == 7
    assert uuid.UUID(b.request_id).version == 7
    assert a.request_id != b.request_id


def test_tp4_occupies_heavy_not_instruct() -> None:
    req = share_for_class("instruct", HEAVY)
    assert req.peer == PEER
    assert req.workloads[0].wrk == "instruct"
    assert req.workloads[0].queue == "root.internal.inference.heavy"
    assert req.workloads[0].resource_class == "internal.inference.heavy"
    assert req.shares[0].queue == HEAVY.queue
    assert req.shares[0].guaranteed.quantities[GPU_KEY] == 4
    assert req.shares[0].max.quantities[GPU_KEY] == 4
    assert "instruct" not in req.shares[0].queue
    ref = share_for_class("referee", HEAVY)
    assert ref.workloads[0].wrk == "referee"
    assert ref.shares[0].queue == HEAVY.queue


def test_zero_floor_valid_until_on_end() -> None:
    req = share_for_class(
        "instruct", HEAVY, gpu=0, valid_until_ns=123, supersedes_request_id="x", applications=0
    )
    assert req.shares[0].guaranteed.quantities[GPU_KEY] == 0
    assert req.valid_until_ns == 123
    assert req.workloads[0].applications == 0
    assert req.supersedes_request_id == "x"
    assert uuid.UUID(req.request_id).version == 7


def test_leaf_from_tp_pp_share() -> None:
    assert leaf_for_gpu_tokens(1) is LIGHT
    assert leaf_for_gpu_tokens(2) is MEDIUM
    assert leaf_for_gpu_tokens(4) is HEAVY
    assert leaf_for_gpu_tokens(8) is HEAVY
    assert "tp" not in HEAVY.queue
    assert "pp" not in HEAVY.queue
    assert "instruct" not in HEAVY.queue


def test_queue_hint_is_declared_leaf_shape() -> None:
    hints = declared_queues()
    assert [h.role for h in hints] == ["light", "medium", "heavy"]
    assert [h.path for h in hints] == [LIGHT.queue, MEDIUM.queue, HEAVY.queue]
    heavy = hints[2]
    assert heavy.gpu_guarantee == 4
    assert heavy.gpu_max == 4
    q = local_response(zpb.SERVER_QUERY_KIND_QUEUES)
    assert [x.path for x in q.queues] == [LIGHT.queue, MEDIUM.queue, HEAVY.queue]


def test_workloads_advertise_model_tp_pp_not_queue_name(monkeypatch) -> None:
    from atelier.engine.config import EngineConfig, ModelSpec
    from atelier.engine.s2s import declared_workloads

    cfg = EngineConfig()
    cfg.capabilities = {
        "instruct": ModelSpec(model="Qwen/Qwen3.6-35B-A3B-FP8", tensor_parallel_size=4, pipeline_parallel_size=1),
        "referee": ModelSpec(model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", tensor_parallel_size=4),
    }
    monkeypatch.setattr("atelier.engine.config.load_engine_config", lambda: cfg)
    rows = declared_workloads()
    by_wrk = {w.wrk: w for w in rows}
    assert by_wrk["instruct"].model == "Qwen/Qwen3.6-35B-A3B-FP8"
    assert by_wrk["instruct"].tensor_parallel == 4
    assert by_wrk["instruct"].pipeline_parallel == 1
    assert by_wrk["instruct"].gpu_tokens == 4
    assert "heavy" not in by_wrk["instruct"].wrk
    assert list(by_wrk["instruct"].capabilities) == ["instruct"]
    q = local_response(zpb.SERVER_QUERY_KIND_WORKLOADS)
    assert {w.wrk for w in q.workloads} == {"instruct", "referee"}


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
        assert request_queue_share("instruct", HEAVY) is False
        assert list_queue_share_requests() == []
    finally:
        server.stop(grace=0)


class _Reject(spbg.SchedulerServicer):
    def RequestQueueShare(self, request, context):
        return spb.QueueShareResponse(
            accepted=False,
            request_id=request.request_id,
            state=spb.QUEUE_SHARE_REJECTED,
            error="over parent max",
        )


def test_rejected_does_not_admit(monkeypatch) -> None:
    server, addr = _serve(_Reject())
    monkeypatch.setenv("SIGNALS_ENGINE_TARGET", addr)
    try:
        with pytest.raises(RuntimeError, match=r"#YK\.00000007\.SHAREFAIL"):
            request_queue_share("instruct", HEAVY)
        with pytest.raises(RuntimeError, match="SHAREFAIL"):
            notify_admit("instruct")
    finally:
        server.stop(grace=0)


class _RejectSilent(spbg.SchedulerServicer):
    def RequestQueueShare(self, request, context):
        return spb.QueueShareResponse(
            accepted=False,
            request_id=request.request_id,
            state=spb.QUEUE_SHARE_REJECTED,
        )


def test_rejected_without_error_still_blocks_admit(monkeypatch) -> None:
    server, addr = _serve(_RejectSilent())
    monkeypatch.setenv("SIGNALS_ENGINE_TARGET", addr)
    try:
        with pytest.raises(RuntimeError, match=r"REJECTED"):
            notify_admit("instruct")
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
            request_queue_share("instruct", HEAVY)
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


def test_admit_then_zero_floor_end(monkeypatch) -> None:
    svc = _Record()
    server, addr = _serve(svc)
    monkeypatch.setenv("SIGNALS_ENGINE_TARGET", addr)
    try:
        assert notify_admit("referee") is True
        assert notify_release("referee") is True
        assert len(svc.seen) == 2
        admit, end = svc.seen
        assert uuid.UUID(admit.request_id).version == 7
        assert uuid.UUID(end.request_id).version == 7
        assert admit.request_id != end.request_id
        assert end.supersedes_request_id == admit.request_id
        assert end.shares[0].guaranteed.quantities[GPU_KEY] == 0
        assert end.valid_until_ns > 0
        assert end.workloads[0].applications == 0
        recs = list_queue_share_requests()
        assert len(recs) == 2
    finally:
        server.stop(grace=0)


def test_notify_admit_unimplemented_ok(monkeypatch) -> None:
    server, addr = _serve(spbg.SchedulerServicer())
    monkeypatch.setenv("SIGNALS_ENGINE_TARGET", addr)
    try:
        assert notify_admit("instruct") is False
    finally:
        server.stop(grace=0)
