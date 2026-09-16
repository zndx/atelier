"""sdg_classify catalogue: AgentRTC-shaped claims, paused, Yield unknown id."""

from __future__ import annotations

from zndx.engine.v1 import engine_pb2 as zpb

from atelier.engine.flow_processes import SpawnedFlow, flow_processes
from atelier.engine.queue_share import LIGHT
from atelier.engine.s2s import local_response
from atelier.engine.sentinel_yield import yield_workload
from atelier.engine.workload_catalog import (
    DAG_ID,
    KIND,
    classify_yk_claims,
    entries,
    schedule_hints,
)


def test_classify_claims_are_light_not_heavy_or_unbudgeted_embedding() -> None:
    claims = classify_yk_claims()
    assert claims == ((LIGHT.queue, 1),)
    assert all(leaf != "root.internal.inference.heavy" for leaf, _ in claims)
    assert all("embedding" not in leaf for leaf, _ in claims)


def test_catalogue_paused_metaflow_dag() -> None:
    [entry] = entries()
    assert entry.kind == KIND
    assert entry.enabled is False
    assert entry.airflow_dag_id == DAG_ID
    assert entry.runner == "metaflow"
    assert entry.cron == "0 8 * * *"
    hint = schedule_hints()[0]
    assert hint.enabled is False
    assert hint.source == "airflow"
    assert hint.airflow_dag_id == DAG_ID
    assert {(c.leaf, c.gpu) for c in hint.claims} == set(entry.claims)


def test_server_query_schedules_publishes_catalogue() -> None:
    resp = local_response(zpb.SERVER_QUERY_KIND_SCHEDULES)
    assert resp.project == "atelier"
    assert [s.kind for s in resp.schedules] == [KIND]
    assert resp.schedules[0].enabled is False


def test_yield_unknown_id_is_idempotent_ok() -> None:
    resp = yield_workload(zpb.YieldRequest(workload_id="missing-run"))
    assert resp.ok
    assert resp.process_ended is False
    empty = yield_workload(zpb.YieldRequest(workload_id=""))
    assert empty.ok
    assert empty.process_ended is False


def test_yield_registered_child() -> None:
    table = flow_processes()
    wid = "sdg-classify-test"
    table.register(SpawnedFlow(workload_id=wid, kind="sdg-classify", pid=None))
    try:
        resp = yield_workload(zpb.YieldRequest(workload_id=wid))
        assert resp.ok
        assert resp.process_ended is True
        again = yield_workload(zpb.YieldRequest(workload_id=wid))
        assert again.ok
        assert again.process_ended is False
    finally:
        table.unregister(wid)
