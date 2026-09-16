"""Registry of spawned classification children for Engine/Yield.

Unknown workload_id is idempotent (ok, not ended). A missing host process
is not a failure.
"""

from __future__ import annotations

import logging
import os
import signal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock

log = logging.getLogger("atelier.engine.flow_processes")


@dataclass
class SpawnedFlow:
    workload_id: str
    kind: str
    pid: int | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FlowProcessTable:
    def __init__(self) -> None:
        self._mu = Lock()
        self._rows: dict[str, SpawnedFlow] = {}

    def register(self, row: SpawnedFlow) -> None:
        with self._mu:
            self._rows[row.workload_id] = row
        log.info(
            "registered flow kind=%s workload_id=%s pid=%s",
            row.kind,
            row.workload_id,
            row.pid,
        )

    def unregister(self, workload_id: str) -> SpawnedFlow | None:
        with self._mu:
            return self._rows.pop(workload_id, None)

    def get(self, workload_id: str) -> SpawnedFlow | None:
        with self._mu:
            return self._rows.get(workload_id)

    def yield_one(self, workload_id: str) -> tuple[bool, str]:
        row = self.unregister(workload_id)
        if row is None:
            return False, f"no spawned flow for workload_id={workload_id or '(empty)'}"
        ended = _terminate_pid(row.pid)
        msg = (
            f"ended {row.kind} pid={row.pid} workload_id={workload_id}"
            if ended
            else f"{row.kind} pid={row.pid} already gone workload_id={workload_id}"
        )
        log.info("yield %s", msg)
        return ended, msg


def _terminate_pid(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    return True


_TABLE = FlowProcessTable()


def flow_processes() -> FlowProcessTable:
    return _TABLE
