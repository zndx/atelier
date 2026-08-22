"""Request YK queue guarantee floors over zndx.scheduler.v1.

WRK (instruct, referee Complete) occupies a resource-class leaf.
Guarantees must move over time so YK preemption can fire. Signals
records RequestQueueShare; applying queues.yaml is Signals later.
UNIMPLEMENTED means Signals is behind the proto — admit still proceeds.
REJECTED means do not admit. QueueHint stays declared leaf shape.

Every QueueShareRequest mints an RFC 9562 UUIDv7 request_id (required;
never omit, never v4). Admit sends occupancy; WRK end sends a zero-floor
with valid_until_ns and supersedes_request_id. Pick heavy/medium/light
from the GPU tokens the packing requirement needs (tp×pp); never put
model/tp/pp in the queue name. Model + capabilities + tp/pp live on
ServerQuery WORKLOADS.
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

# Last admit request_id per WRK kind (release supersedes).
_ADMIT_IDS: dict[str, str] = {}


@dataclass(frozen=True)
class ResourceClass:
    """Leaf resource class. ``name`` is the stamp; ``queue`` is the YK path."""

    name: str
    queue: str
    gpu_tokens: int
    max_applications: int = 1


# Default TP=4 occupies the same leaf as Gaius/Ægir thinking.
HEAVY = ResourceClass(
    name="internal.inference.heavy",
    queue="root.internal.inference.heavy",
    gpu_tokens=4,
    max_applications=1,
)
MEDIUM = ResourceClass(
    name="internal.inference.medium",
    queue="root.internal.inference.medium",
    gpu_tokens=2,
    max_applications=1,
)
LIGHT = ResourceClass(
    name="internal.inference.light",
    queue="root.internal.inference.light",
    gpu_tokens=1,
    max_applications=2,
)


def leaf_for_gpu_tokens(n: int) -> ResourceClass:
    """heavy/medium/light from the GPU share the packing requirement needs.

    Queue names stay resource-class FQNs — never encode model or tp/pp here.
    """
    tokens = int(n)
    if tokens >= 4:
        return HEAVY
    if tokens >= 2:
        return MEDIUM
    if tokens >= 1:
        return LIGHT
    return HEAVY


def resource_class_for_capability(
    capability: str,
    tp: int | None = None,
    pp: int | None = None,
) -> ResourceClass:
    """Map a capability + tp/pp packing requirement to a declared leaf."""
    if tp is None or pp is None:
        spec = None
        try:
            from atelier.engine.config import load_engine_config

            spec = load_engine_config().capabilities.get(capability)
        except Exception:
            spec = None
        if spec is not None:
            if tp is None:
                tp = int(spec.tensor_parallel_size)
            if pp is None:
                pp = int(getattr(spec, "pipeline_parallel_size", 1) or 1)
        else:
            if tp is None:
                tp = 4
            if pp is None:
                pp = 1
    return leaf_for_gpu_tokens(int(tp) * int(pp))


def resource_class_for(kind: str) -> ResourceClass:
    return resource_class_for_capability(kind)


def mint_uuid7() -> str:
    """RFC 9562 UUIDv7. Required on every RequestQueueShare — never omit, never v4."""
    gen = getattr(uuid, "uuid7", None)
    if callable(gen):
        u = gen()
    else:
        ts_ms = time.time_ns() // 1_000_000
        ts_ms &= (1 << 48) - 1
        rnd = int.from_bytes(os.urandom(10), "big")
        rand_a = (rnd >> 62) & 0xFFF
        rand_b = rnd & ((1 << 62) - 1)
        u = uuid.UUID(int=(ts_ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b)
    if u.version != 7:
        raise RuntimeError(
            f"{GURU_SHAREFAIL} request_id must be RFC 9562 UUIDv7, got v{u.version}"
        )
    return str(u)


def require_uuid7(value: str) -> str:
    s = (value or "").strip()
    if not s:
        raise RuntimeError(
            f"{GURU_SHAREFAIL} request_id omitted; RFC 9562 UUIDv7 required"
        )
    try:
        u = uuid.UUID(s)
    except ValueError as e:
        raise RuntimeError(
            f"{GURU_SHAREFAIL} request_id is not a UUID: {s!r}"
        ) from e
    if u.version != 7:
        raise RuntimeError(
            f"{GURU_SHAREFAIL} request_id must be RFC 9562 UUIDv7, not v{u.version}"
        )
    return s


def _addr() -> str:
    return (
        os.environ.get("SIGNALS_ENGINE_GRPC")
        or os.environ.get("SIGNALS_ENGINE_TARGET")
        or "127.0.0.1:50551"
    )


def share_for_class(
    kind: str,
    rc: ResourceClass,
    *,
    gpu: int | None = None,
    valid_until_ns: int = 0,
    supersedes_request_id: str = "",
    applications: int | None = None,
) -> object:
    """Build a QueueShareRequest for one WRK occupying ``rc``."""
    from zndx.scheduler.v1 import scheduler_pb2 as spb

    tokens = int(rc.gpu_tokens if gpu is None else gpu)
    max_gpu = int(rc.gpu_tokens) if int(rc.gpu_tokens) >= 2 else (2 if int(rc.gpu_tokens) else 0)
    apps = 0 if tokens == 0 else (1 if applications is None else int(applications))
    share = spb.QueueShare(
        queue=rc.queue,
        guaranteed=spb.ResourceMap(quantities={GPU_KEY: tokens}),
        max=spb.ResourceMap(quantities={GPU_KEY: max_gpu}),
        max_applications=int(rc.max_applications),
    )
    wrk = spb.WorkloadIntent(
        wrk=kind.replace("_", "-"),
        queue=rc.queue,
        resource_class=rc.name,
        applications=apps,
    )
    return spb.QueueShareRequest(
        peer=PEER,
        request_id=mint_uuid7(),
        valid_from_ns=time.time_ns(),
        valid_until_ns=int(valid_until_ns),
        reason=(
            f"{kind} ended on {rc.queue} (zero floor)"
            if tokens == 0
            else f"{kind} occupies {rc.queue} (guarantee gpu={tokens})"
        ),
        supersedes_request_id=supersedes_request_id,
        workloads=[wrk],
        shares=[share],
    )


def _send(req) -> object:
    import grpc
    from zndx.scheduler.v1 import scheduler_pb2 as spb
    from zndx.scheduler.v1 import scheduler_pb2_grpc as spb_grpc

    require_uuid7(req.request_id)
    channel = grpc.insecure_channel(_addr())
    try:
        stub = spb_grpc.SchedulerStub(channel)
        resp = stub.RequestQueueShare(req, timeout=5.0)
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNIMPLEMENTED:
            log.info(
                "RequestQueueShare UNIMPLEMENTED (Signals behind proto) wrk=%s queue=%s",
                req.workloads[0].wrk if req.workloads else "",
                req.shares[0].queue if req.shares else "",
            )
            return None
        raise RuntimeError(
            f"{GURU_SHAREFAIL} RequestQueueShare failed: {e.code()} {e.details()}\n"
            "  Signals Scheduler on SIGNALS_ENGINE_TARGET; do not write queues.yaml"
        ) from e
    finally:
        channel.close()
    if resp.state == spb.QUEUE_SHARE_REJECTED:
        raise RuntimeError(
            f"{GURU_SHAREFAIL} {resp.error or 'REJECTED'}\n"
            "  Signals Scheduler on SIGNALS_ENGINE_TARGET; do not write queues.yaml"
        )
    if not resp.accepted and (resp.error or "").strip():
        raise RuntimeError(
            f"{GURU_SHAREFAIL} {resp.error}\n"
            "  Signals Scheduler on SIGNALS_ENGINE_TARGET; do not write queues.yaml"
        )
    log.info(
        "RequestQueueShare accepted=%s state=%s wrk=%s queue=%s id=%s",
        resp.accepted,
        resp.state,
        req.workloads[0].wrk if req.workloads else "",
        req.shares[0].queue if req.shares else "",
        resp.request_id or req.request_id,
    )
    return resp


def request_queue_share(kind: str, rc: ResourceClass) -> bool:
    """Tell Signals the occupancy intent. True if recorded.

    ``UNIMPLEMENTED``: proto is ahead of Signals — log, do not fail admit.
    ``REJECTED`` or any other gRPC/persist error: fail-fast SHAREFAIL, do not admit.
    """
    req = share_for_class(kind, rc)
    resp = _send(req)
    if resp is None:
        return False
    _ADMIT_IDS[kind.replace("_", "-")] = req.request_id
    return bool(resp.accepted)


def request_queue_share_end(kind: str, rc: ResourceClass) -> bool:
    """Zero-floor occupancy and close the window when the WRK ends."""
    key = kind.replace("_", "-")
    prior = _ADMIT_IDS.pop(key, "")
    now = time.time_ns()
    req = share_for_class(
        kind,
        rc,
        gpu=0,
        valid_until_ns=now,
        supersedes_request_id=prior,
        applications=0,
    )
    resp = _send(req)
    return bool(resp and resp.accepted)


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


def notify_admit(kind: str, tp: int | None = None, pp: int | None = None) -> bool:
    """Call RequestQueueShare when a WRK admits. REJECTED/SHAREFAIL fails fast."""
    rc = resource_class_for_capability(kind, tp, pp)
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


def notify_release(kind: str, tp: int | None = None, pp: int | None = None) -> bool:
    """Zero-floor + valid_until when the WRK ends. Does not block local stop."""
    rc = resource_class_for_capability(kind, tp, pp)
    try:
        return request_queue_share_end(kind, rc)
    except RuntimeError as e:
        log.warning("RequestQueueShare end skipped: %s", e)
        return False
    except Exception as e:
        log.warning("RequestQueueShare end skipped: %s", e)
        return False
