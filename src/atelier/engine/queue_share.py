"""Request YK queue guarantee floors over zndx.scheduler.v1.

WRK (instruct, referee Complete) occupies a resource-class leaf.
Guarantees must move over time so YK preemption can fire. Signals
records RequestQueueShare; applying queues.yaml is Signals later.
UNIMPLEMENTED means Signals is behind the proto — admit still proceeds.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass

log = logging.getLogger("atelier.engine.queue_share")

GURU_SHAREFAIL = "#YK.00000007.SHAREFAIL"
GPU_KEY = "federation.zndx.org/gpu"
PEER = "atelier"


@dataclass(frozen=True)
class ResourceClass:
    """Leaf resource class. ``name`` is the stamp; ``queue`` is the YK path."""

    name: str
    queue: str
    gpu_tokens: int
    max_applications: int = 1


# TP=4 instruct / referee Complete. Occupancy intent is RequestQueueShare;
# QueueHint stays the declared leaf (default guarantee 0, max 4).
INSTRUCT = ResourceClass(
    name="internal.inference.instruct",
    queue="root.internal.inference.instruct",
    gpu_tokens=4,
    max_applications=1,
)

_KIND_CLASS: dict[str, ResourceClass] = {
    "instruct": INSTRUCT,
    "referee": INSTRUCT,
}


def resource_class_for(kind: str) -> ResourceClass:
    try:
        return _KIND_CLASS[kind.replace("_", "-")]
    except KeyError as e:
        raise KeyError(
            f"no resource class for kind={kind!r}; "
            "instruct/referee occupy internal.inference.instruct"
        ) from e


def _request_id() -> str:
    gen = getattr(uuid, "uuid7", None)
    return str(gen() if callable(gen) else uuid.uuid4())


def _addr() -> str:
    return (
        os.environ.get("SIGNALS_ENGINE_GRPC")
        or os.environ.get("SIGNALS_ENGINE_TARGET")
        or "127.0.0.1:50551"
    )


def share_for_class(kind: str, rc: ResourceClass) -> object:
    """Build a QueueShareRequest for one WRK occupying ``rc``."""
    from zndx.scheduler.v1 import scheduler_pb2 as spb

    gpu = int(rc.gpu_tokens)
    max_gpu = gpu if gpu >= 2 else (2 if gpu else 0)
    share = spb.QueueShare(
        queue=rc.queue,
        guaranteed=spb.ResourceMap(quantities={GPU_KEY: gpu}),
        max=spb.ResourceMap(quantities={GPU_KEY: max_gpu}),
        max_applications=int(rc.max_applications),
    )
    wrk = spb.WorkloadIntent(
        wrk=kind.replace("_", "-"),
        queue=rc.queue,
        resource_class=rc.name,
        applications=1,
    )
    return spb.QueueShareRequest(
        peer=PEER,
        request_id=_request_id(),
        valid_from_ns=time.time_ns(),
        reason=f"{kind} occupies {rc.queue} (guarantee gpu={gpu})",
        workloads=[wrk],
        shares=[share],
    )


def request_queue_share(kind: str, rc: ResourceClass) -> bool:
    """Tell Signals the occupancy intent. True if recorded.

    ``UNIMPLEMENTED``: proto is ahead of Signals — log, do not fail admit.
    Any other gRPC error: fail-fast SHAREFAIL.
    """
    import grpc
    from zndx.scheduler.v1 import scheduler_pb2_grpc as spb_grpc

    req = share_for_class(kind, rc)
    channel = grpc.insecure_channel(_addr())
    try:
        stub = spb_grpc.SchedulerStub(channel)
        resp = stub.RequestQueueShare(req, timeout=5.0)
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNIMPLEMENTED:
            log.info(
                "RequestQueueShare UNIMPLEMENTED (Signals behind proto) wrk=%s queue=%s",
                kind,
                rc.queue,
            )
            return False
        raise RuntimeError(
            f"{GURU_SHAREFAIL} RequestQueueShare failed: {e.code()} {e.details()}\n"
            "  Signals Scheduler on SIGNALS_ENGINE_TARGET; do not write queues.yaml"
        ) from e
    finally:
        channel.close()
    if not resp.accepted and (resp.error or "").strip():
        raise RuntimeError(
            f"{GURU_SHAREFAIL} {resp.error}\n"
            "  Signals Scheduler on SIGNALS_ENGINE_TARGET; do not write queues.yaml"
        )
    log.info(
        "RequestQueueShare accepted=%s state=%s wrk=%s queue=%s id=%s",
        resp.accepted,
        resp.state,
        kind,
        rc.queue,
        resp.request_id or req.request_id,
    )
    return bool(resp.accepted)


def list_queue_share_requests(
    *,
    peer: str = PEER,
    queue: str = "",
    since_ns: int = 0,
    limit: int = 0,
) -> list:
    """History SoR. UNIMPLEMENTED → empty (Signals-not-yet). Else SHAREFAIL."""
    import grpc
    from zndx.scheduler.v1 import scheduler_pb2 as spb
    from zndx.scheduler.v1 import scheduler_pb2_grpc as spb_grpc

    channel = grpc.insecure_channel(_addr())
    try:
        stub = spb_grpc.SchedulerStub(channel)
        resp = stub.ListQueueShareRequests(
            spb.ListQueueShareRequestsRequest(
                peer=peer, queue=queue, since_ns=since_ns, limit=limit
            ),
            timeout=5.0,
        )
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNIMPLEMENTED:
            log.info(
                "ListQueueShareRequests UNIMPLEMENTED (Signals behind proto) peer=%s",
                peer,
            )
            return []
        raise RuntimeError(
            f"{GURU_SHAREFAIL} ListQueueShareRequests failed: {e.code()} {e.details()}\n"
            "  Signals Scheduler on SIGNALS_ENGINE_TARGET; do not write queues.yaml"
        ) from e
    finally:
        channel.close()
    return list(resp.records)


def notify_admit(kind: str) -> bool:
    """Call RequestQueueShare when a WRK admits. SHAREFAIL fails fast."""
    rc = resource_class_for(kind)
    try:
        return request_queue_share(kind, rc)
    except RuntimeError as e:
        if "SHAREFAIL" in str(e):
            raise
        log.warning("RequestQueueShare skipped: %s", e)
        return False
    except Exception as e:
        log.warning("RequestQueueShare skipped: %s", e)
        return False
