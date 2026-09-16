"""Engine/Yield: end a registered classification child. Unknown id is ok."""

from __future__ import annotations

from atelier.engine.flow_processes import flow_processes
from zndx.engine.v1 import engine_pb2 as zpb


def yield_workload(request: zpb.YieldRequest) -> zpb.YieldResponse:
    wid = (request.workload_id or "").strip()
    if not wid:
        return zpb.YieldResponse(
            ok=True,
            process_ended=False,
            restore_started=False,
            message="empty workload_id",
        )
    table = flow_processes()
    if table.get(wid) is None:
        return zpb.YieldResponse(
            ok=True,
            process_ended=False,
            restore_started=False,
            message=f"no host process for workload_id={wid}",
        )
    ended, msg = table.yield_one(wid)
    return zpb.YieldResponse(
        ok=True,
        process_ended=ended,
        restore_started=False,
        message=msg,
    )
