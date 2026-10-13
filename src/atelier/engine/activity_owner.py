"""Start ClassificationFlow when this engine's classify activity is RUNNING.

The activity reason is the run configuration. The flow is
``python -m atelier.flows.run_classify``, which talks to Signals Metaflow
and does not open Airflow. While the child is alive this owner renews the
lease (Signals TTL is 180s). A finished child is released. A refused
release stays registered so the next poll retries and does not start a
second flow.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from atelier.engine.flow_processes import SpawnedFlow, flow_processes
from atelier.engine.workload_catalog import HORIZON_S, KIND, PEER
from atelier.engine.workload_sync import signals_sync_target
from atelier.flows.execution_topology import (
    argv_from_configuration,
    configuration_from_reason,
)

logger = logging.getLogger("atelier.engine.activity_owner")

POLL_S = 30.0
RPC_TIMEOUT_S = 10.0
# Same coordination codes Gaius uses for these two RPCs.
GURU_RENEWFAIL = "#CO.00000006.RENEWFAIL"
GURU_RELEASEFAIL = "#CO.00000007.RELEASEFAIL"

Popen = Callable[..., Any]


@dataclass
class _Child:
    proc: Any
    horizon_ns: int


def flow_command(reason: str) -> list[str]:
    """Metaflow command line taken from the activity reason.

    Model, claims, and trigger stay the declared document. The catalogue
    is not read again here.
    """
    conf = configuration_from_reason(reason)
    if str(conf.get("kind") or KIND) != KIND:
        raise ValueError(f"activity kind {conf.get('kind')!r} is not {KIND}")
    if not str(conf.get("source_id") or "").strip():
        raise ValueError("activity configuration has no source_id")
    return [
        sys.executable,
        "-m",
        "atelier.flows.run_classify",
        *argv_from_configuration(conf),
    ]


def _outcome(code: int) -> str:
    """Release text. A nonzero exit is ``failed:`` so Signals records no effect."""
    if int(code) == 0:
        return "completed: metaflow exit 0"
    return f"failed: metaflow exit {int(code)}"


def _renew_horizon(activity_horizon: int) -> int | None:
    """Heartbeat horizon. A passed horizon is not extended."""
    now = time.time_ns()
    if int(activity_horizon) > now:
        return int(activity_horizon)
    if int(activity_horizon) <= 0:
        return now + int(HORIZON_S) * 1_000_000_000
    return None


class ActivityOwner:
    """Poll Signals for this peer's classify activity and run its flow."""

    def __init__(
        self,
        target: str,
        *,
        popen: Popen | None = None,
        interval_s: float = POLL_S,
    ) -> None:
        self.target = target
        self.popen = popen or subprocess.Popen
        self.interval_s = float(interval_s)
        self._children: dict[str, _Child] = {}
        self._finished: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._unimplemented = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="atelier-activity-owner", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:  # noqa: BLE001 — a dark Signals must not kill the engine
                logger.warning("activity owner poll failed: %s", exc)
            self._stop.wait(self.interval_s)

    def poll_once(self) -> list[str]:
        self._reap()
        listed = self._list_active()
        if listed is None:
            return []
        spawned: list[str] = []
        for activity in listed:
            if activity.peer != PEER or activity.kind != KIND:
                continue
            if int(activity.state) != _running():
                continue
            if activity.activity_id in self._finished:
                continue
            child = self._children.get(activity.activity_id)
            if child is not None:
                if child.proc.poll() is None:
                    self._renew(activity.activity_id, int(activity.horizon_ns))
                continue
            if flow_processes().get(activity.activity_id) is not None:
                continue
            try:
                cmd = flow_command(activity.reason)
            except ValueError as exc:
                logger.warning(
                    "classify activity %s has no topology: %s",
                    activity.activity_id, exc,
                )
                continue
            proc = self.popen(cmd)
            self._children[activity.activity_id] = _Child(
                proc, int(activity.horizon_ns),
            )
            flow_processes().register(SpawnedFlow(
                workload_id=activity.activity_id,
                kind=KIND,
                pid=getattr(proc, "pid", None),
            ))
            self._renew(activity.activity_id, int(activity.horizon_ns))
            logger.info("started %s for activity %s", cmd[0:4], activity.activity_id)
            spawned.append(activity.activity_id)
        return spawned

    def _reap(self) -> None:
        for activity_id, child in list(self._children.items()):
            code = child.proc.poll()
            if code is None:
                continue
            try:
                _release(self.target, activity_id, _outcome(int(code)))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "%s release %s failed: %s", GURU_RELEASEFAIL, activity_id, exc,
                )
                continue
            self._children.pop(activity_id, None)
            self._finished.add(activity_id)
            flow_processes().unregister(activity_id)

    def _renew(self, activity_id: str, horizon_ns: int) -> None:
        horizon = _renew_horizon(horizon_ns)
        if horizon is None:
            logger.warning(
                "%s renew %s skipped: horizon passed", GURU_RENEWFAIL, activity_id,
            )
            return
        try:
            _renew(self.target, activity_id, horizon)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "%s renew %s failed: %s", GURU_RENEWFAIL, activity_id, exc,
            )

    def _list_active(self) -> list[Any] | None:
        """RUNNING-capable list. None means Signals has no ListActivities."""
        import grpc
        from zndx.scheduler.v1 import scheduler_pb2 as spb
        from zndx.scheduler.v1 import scheduler_pb2_grpc as spb_grpc

        req = spb.ListActivitiesRequest(
            peer=PEER, kind=KIND, active_only=True,
        )
        channel = grpc.insecure_channel(self.target)
        try:
            stub = spb_grpc.SchedulerStub(channel)
            try:
                resp = stub.ListActivities(req, timeout=RPC_TIMEOUT_S)
            except grpc.RpcError as exc:
                if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
                    if not self._unimplemented:
                        logger.warning(
                            "Signals has no ListActivities; owner will not "
                            "start classify",
                        )
                        self._unimplemented = True
                    return None
                raise RuntimeError(f"{exc.code().name} {exc.details()}") from exc
        finally:
            channel.close()
        self._unimplemented = False
        return list(resp.activities)


def _running() -> int:
    from zndx.engine.v1 import engine_pb2 as zpb

    return int(zpb.ACTIVITY_RUNNING)


def _renew(target: str, activity_id: str, horizon_ns: int) -> None:
    import grpc
    from zndx.scheduler.v1 import scheduler_pb2 as spb
    from zndx.scheduler.v1 import scheduler_pb2_grpc as spb_grpc

    req = spb.RenewActivityRequest(
        peer=PEER, activity_id=activity_id, horizon_ns=int(horizon_ns),
    )
    channel = grpc.insecure_channel(target)
    try:
        stub = spb_grpc.SchedulerStub(channel)
        resp = stub.RenewActivity(req, timeout=RPC_TIMEOUT_S)
    finally:
        channel.close()
    if not resp.accepted:
        raise RuntimeError(
            f"{GURU_RENEWFAIL} {resp.error or 'RenewActivity refused'}"
        )


def _release(target: str, activity_id: str, outcome: str) -> None:
    import grpc
    from zndx.scheduler.v1 import scheduler_pb2 as spb
    from zndx.scheduler.v1 import scheduler_pb2_grpc as spb_grpc

    req = spb.ReleaseActivityRequest(
        peer=PEER, activity_id=activity_id, outcome=outcome,
    )
    channel = grpc.insecure_channel(target)
    try:
        stub = spb_grpc.SchedulerStub(channel)
        resp = stub.ReleaseActivity(req, timeout=RPC_TIMEOUT_S)
    finally:
        channel.close()
    if not resp.accepted:
        raise RuntimeError(
            f"{GURU_RELEASEFAIL} {resp.error or 'ReleaseActivity refused'}"
        )


def init_activity_owner(target: str | None = None) -> ActivityOwner:
    addr = (target or "").strip() or signals_sync_target()
    owner = ActivityOwner(addr)
    owner.start()
    logger.info("activity-owner started target=%s", addr)
    return owner
