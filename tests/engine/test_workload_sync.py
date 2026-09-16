"""SyncWorkloads submits the paused sdg_classify catalogue as a replace."""

from __future__ import annotations

from concurrent import futures

import grpc
import pytest

from atelier.engine import workload_catalog as wc
from atelier.engine import workload_sync as ws
from zndx.scheduler.v1 import scheduler_pb2 as spb
from zndx.scheduler.v1 import scheduler_pb2_grpc as spbg


class _Signals(spbg.SchedulerServicer):
    def __init__(self, *, accept=True, unimplemented=False):
        self.requests: list = []
        self.accept = accept
        self.unimplemented = unimplemented

    def SyncWorkloads(self, request, context):
        if self.unimplemented:
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "no SyncWorkloads")
        self.requests.append(request)
        if not self.accept:
            return spb.SyncWorkloadsResponse(accepted=False, error="refused")
        resp = spb.SyncWorkloadsResponse(accepted=True)
        for h in request.workloads:
            rec = resp.records.add()
            rec.workload.CopyFrom(h)
            rec.peer = request.peer
            rec.dag_id = h.airflow_dag_id
            rec.state = "materialized" if h.enabled else "paused"
        return resp


def _serve(servicer):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    spbg.add_SchedulerServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, f"127.0.0.1:{port}"


def test_sync_submits_paused_sdg_classify_as_replace() -> None:
    sig = _Signals()
    server, addr = _serve(sig)
    try:
        sync = ws.WorkloadSync(addr)
        records = sync.sync_once()
    finally:
        server.stop(0)
    assert len(sig.requests) == 1
    req = sig.requests[0]
    assert req.peer == "atelier" and req.replace is True
    assert [h.kind for h in req.workloads] == [wc.KIND]
    assert req.workloads[0].enabled is False
    assert req.workloads[0].airflow_dag_id == wc.DAG_ID
    assert records[0]["state"] == "paused"
    assert sync.syncs == 1 and sync.last_error == ""


def test_refused_sync_is_an_error() -> None:
    server, addr = _serve(_Signals(accept=False))
    try:
        sync = ws.WorkloadSync(addr)
        with pytest.raises(RuntimeError, match="refused"):
            sync.sync_once()
        assert "refused" in (sync.last_error or "refused")
    finally:
        server.stop(0)


def test_unimplemented_is_nosync() -> None:
    server, addr = _serve(_Signals(unimplemented=True))
    try:
        sync = ws.WorkloadSync(addr)
        with pytest.raises(ws._Unimplemented):
            sync.sync_once()
    finally:
        server.stop(0)
