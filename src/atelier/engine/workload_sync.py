"""Submit Atelier's WORKLOAD CATALOGUE to Signals (Scheduler/SyncWorkloads).

Fail-open toward boot: a dark or older Signals never gates the engine.
UNIMPLEMENTED logs #CO.0000000B.NOSYNC once and retries; other failures
are #CO.0000000C.SYNCFAIL. replace=True is the whole catalogue.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from atelier.engine import workload_catalog

logger = logging.getLogger("atelier.engine.workload_sync")

GURU_NOSYNC = "#CO.0000000B.NOSYNC"
GURU_SYNCFAIL = "#CO.0000000C.SYNCFAIL"

SYNC_INTERVAL_S = 30 * 60.0
RETRY_S = 300.0
RPC_TIMEOUT_S = 30.0


class _Unimplemented(Exception):
    pass


class WorkloadSync:
    """Periodic Scheduler/SyncWorkloads; last outcome kept for /workloads."""

    def __init__(
        self,
        target: str,
        *,
        project: str = workload_catalog.PEER,
        interval_s: float = SYNC_INTERVAL_S,
    ) -> None:
        self.target = target
        self.project = project
        self.interval_s = float(interval_s)
        self.last_sync_ms: int = 0
        self.last_error: str = ""
        self.last_records: list[dict[str, Any]] = []
        self.syncs = 0
        self._unimplemented_logged = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="atelier-workload-sync", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            delay = self.interval_s
            try:
                self.sync_once()
            except _Unimplemented:
                if not self._unimplemented_logged:
                    logger.warning(
                        "%s Signals at %s has no Scheduler/SyncWorkloads — "
                        "catalogue not submitted; retry every %.0fs",
                        GURU_NOSYNC, self.target, RETRY_S,
                    )
                    self._unimplemented_logged = True
                delay = RETRY_S
            except Exception as e:  # noqa: BLE001 — never kill the loop
                self.last_error = str(e)
                logger.warning("%s SyncWorkloads to %s: %s", GURU_SYNCFAIL, self.target, e)
            self._stop.wait(delay)

    def sync_once(self) -> list[dict[str, Any]]:
        import grpc

        from zndx.scheduler.v1 import scheduler_pb2 as spb
        from zndx.scheduler.v1 import scheduler_pb2_grpc as spb_grpc

        hints = workload_catalog.schedule_hints()
        req = spb.SyncWorkloadsRequest(
            peer=self.project,
            workloads=hints,
            replace=True,
            engine_build=_engine_build(),
        )
        channel = grpc.insecure_channel(self.target)
        try:
            stub = spb_grpc.SchedulerStub(channel)
            try:
                resp = stub.SyncWorkloads(req, timeout=RPC_TIMEOUT_S)
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                    raise _Unimplemented() from e
                raise RuntimeError(f"{e.code().name} {e.details()}") from e
        finally:
            channel.close()
        if not resp.accepted:
            raise RuntimeError(resp.error or "SyncWorkloads refused")
        records = [
            {
                "id": r.workload.id,
                "kind": r.workload.kind,
                "dag_id": r.dag_id,
                "state": r.state,
                "error": r.error,
            }
            for r in resp.records
        ]
        self.last_records = records
        self.last_sync_ms = int(time.time() * 1000)
        self.last_error = ""
        self.syncs += 1
        self._unimplemented_logged = False
        by_state: dict[str, int] = {}
        for r in records:
            by_state[r["state"] or "?"] = by_state.get(r["state"] or "?", 0) + 1
        logger.info(
            "workload catalogue submitted to %s: %d entry → %s",
            self.target, len(hints),
            ", ".join(f"{k}={v}" for k, v in sorted(by_state.items())) or "no records",
        )
        return records

    def status(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "last_sync_ms": self.last_sync_ms,
            "last_error": self.last_error,
            "syncs": self.syncs,
            "records": list(self.last_records),
        }


def _engine_build() -> str:
    try:
        from atelier.engine.s2s import advertised_head

        return advertised_head() or ""
    except Exception:  # noqa: BLE001
        return ""


def signals_sync_target(environ: dict[str, str] | None = None) -> str:
    """Discover Signals Scheduler (same engine as Metaflow discovery)."""
    from atelier.flows.platform_metaflow import load_contract, signals_engine_addr

    env = environ if environ is not None else __import__("os").environ
    explicit = (env.get("SIGNALS_ENGINE_GRPC") or "").strip()
    if explicit:
        return explicit
    doc = load_contract(env)
    return signals_engine_addr(doc, env)


_SYNC: WorkloadSync | None = None


def init_workload_sync(target: str, **kw: Any) -> WorkloadSync:
    global _SYNC
    if _SYNC is None:
        _SYNC = WorkloadSync(target, **kw)
    return _SYNC


def get_workload_sync() -> WorkloadSync | None:
    return _SYNC
