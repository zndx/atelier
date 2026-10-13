"""Declare one imperative Metaflow run. Signals is the Airflow hop.

Fail closed. A dark Scheduler is an error, not an in-process pipeline.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from atelier.engine.queue_share import mint_uuid7, require_uuid7
from atelier.engine.workload_sync import signals_sync_target
from atelier.flows.execution_topology import (
    ExecutionTopology,
    classify_topology,
    configuration,
    reason_for,
)

logger = logging.getLogger("atelier.engine.activity_declare")

# 0D is Gaius MISSTICK. Declare refusal is the next free coordination code.
GURU_NODECLARE = "#CO.0000000A.NODECLARE"
RPC_TIMEOUT_S = 10.0


class DeclareError(RuntimeError):
    """Signals refused the activity or could not be reached."""


def build_request(topo: ExecutionTopology, *, request_id: str | None = None):
    """DeclareActivityRequest whose conf is the execution topology."""
    from zndx.scheduler.v1 import scheduler_pb2 as spb

    rid = require_uuid7(request_id or mint_uuid7())
    req = spb.DeclareActivityRequest(
        peer=topo.peer,
        request_id=rid,
        kind=topo.kind,
        owner=topo.peer,
        horizon_ns=time.time_ns() + int(topo.horizon_s) * 1_000_000_000,
        precludes=list(topo.precludes),
        reason=reason_for(topo),
    )
    for leaf, gpu in topo.claims:
        req.claims.add(leaf=str(leaf), gpu=int(gpu))
    return req


def declare_topology(
    topo: ExecutionTopology,
    *,
    target: str | None = None,
    timeout_s: float = RPC_TIMEOUT_S,
) -> dict[str, Any]:
    """One Signals DeclareActivity. Does not start Metaflow or a local thread."""
    import grpc
    from zndx.scheduler.v1 import scheduler_pb2_grpc as spb_grpc

    req = build_request(topo)
    addr = (target or "").strip() or signals_sync_target()
    channel = grpc.insecure_channel(addr)
    try:
        stub = spb_grpc.SchedulerStub(channel)
        try:
            resp = stub.DeclareActivity(req, timeout=float(timeout_s))
        except grpc.RpcError as exc:
            raise DeclareError(
                f"{GURU_NODECLARE} DeclareActivity {exc.code().name}: {exc.details()}"
            ) from exc
    finally:
        channel.close()
    if not resp.accepted:
        raise DeclareError(resp.error or f"{GURU_NODECLARE} DeclareActivity refused")
    activity = resp.activity
    logger.info(
        "declared %s activity_id=%s dag_id=%s state=%s",
        topo.kind,
        activity.activity_id,
        activity.dag_id or topo.dag_id,
        activity.state,
    )
    return {
        "started": False,
        "scheduled": True,
        "clock": "airflow",
        "activity_id": activity.activity_id,
        "dag_id": activity.dag_id or topo.dag_id,
        "run_id": activity.run_id,
        "state": int(activity.state),
        "request_id": req.request_id,
        "configuration": configuration(topo),
    }


def schedule_classify(source_id: str | None) -> dict[str, Any]:
    """User trigger: declare the classify activity. Airflow runs the flow."""
    sid = (source_id or "").strip()
    if not sid:
        from atelier.sdg.sample import current_sample_source_id

        sid = current_sample_source_id()
    if not sid:
        return {
            "started": False,
            "scheduled": False,
            "clock": "airflow",
            "error": (
                "no sdg-corpora sample id; run `just sdg-sample` "
                "before declaring the classify activity"
            ),
        }
    topo = classify_topology(source_id=sid, trigger="imperative")
    try:
        return declare_topology(topo)
    except Exception as exc:  # noqa: BLE001 — the HTTP caller needs the refusal
        logger.warning("classify declare failed: %s", exc)
        return {
            "started": False,
            "scheduled": False,
            "clock": "airflow",
            "dag_id": topo.dag_id,
            "error": str(exc),
        }


def interpret_dispatch(result: dict[str, Any]) -> dict[str, Any]:
    """Tell callers whether fsm_start declared an activity or started a thread.

    A refusal is an error. A federation declare is scheduled and not
    started. The CAI clock is started.
    """
    scheduled = bool(result.get("scheduled"))
    started = bool(result.get("started"))
    error = str(result.get("error") or "")
    if error and not scheduled and not started:
        raise DeclareError(error)
    if scheduled and not started:
        status = "scheduled"
    elif started:
        status = "started"
    else:
        status = "idle"
    return {
        "scheduled": scheduled,
        "started": started,
        "activity_id": str(result.get("activity_id") or ""),
        "status": status,
    }
