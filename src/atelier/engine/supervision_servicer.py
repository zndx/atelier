"""zndx.supervision.v1.EngineSupervision — Atelier lattice port, observe-only.

Nautilus dials Supervise on :50251. Sync gRPC (this engine is not aio).
Follows Hermes: ring replay, no warehouse Backlog, directives acknowledged
but not applied. Project mismatch → GOODBYE.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Iterator

import grpc

from atelier.engine.supervision_bus import (
    KIND_DIRECTIVE_RESULT,
    KIND_GOODBYE,
    KIND_SERVING,
    SupervisionEvent,
    get_bus,
    init_bus,
    now_ms,
)
from atelier.engine.supervision_spec import load_spec
from zndx.supervision.v1 import supervision_pb2 as sv
from zndx.supervision.v1 import supervision_pb2_grpc as sv_grpc

log = logging.getLogger("atelier.engine.supervision")

PROJECT = "atelier"
GURU_NOSUBSCRIBE = "#SV.00000003.NOSUBSCRIBE"
GURU_OBSERVE = "#SV.ATELIER.OBSERVE"
HEARTBEAT_S = 30.0
CAPABILITIES = (
    "replay:ring",
    "directive:observe_only",
    "serving:status",
)


def _engine_build() -> str:
    try:
        from atelier.engine.s2s import advertised_head

        sha = advertised_head()
        if sha:
            return str(sha)[:12]
    except Exception:
        pass
    return "trunk"


def _epoch() -> tuple[str, str, str]:
    spec = load_spec()
    version = spec.spec_version if spec else ""
    sha = spec.sha256 if spec else ""
    rev = _engine_build()
    return f"{version}+{rev}", version, sha


def event_to_proto(ev: SupervisionEvent, epoch: str) -> sv.EngineEvent:
    out = sv.EngineEvent(seq=ev.seq, at_unix_ms=ev.at_unix_ms, epoch=epoch, replayed=ev.replayed)
    p = ev.payload
    if ev.kind == KIND_SERVING:
        out.serving.CopyFrom(
            sv.ServingEvent(
                phase=sv.SERVING_PHASE_SETTLED,
                generation=int(p.get("generation") or 0),
                capability=str(p.get("capability") or "referee"),
                alias=str(p.get("alias") or "atelier"),
                model=str(p.get("model") or ""),
                actual=sv.SERVING_STATUS_SERVING,
                warmup_seconds=int(p.get("warmup_seconds") or 30),
                settled_unix_ms=int(p.get("settled_unix_ms") or ev.at_unix_ms),
            )
        )
    elif ev.kind == KIND_DIRECTIVE_RESULT:
        out.directive_result.CopyFrom(
            sv.DirectiveResult(
                directive_id=str(p.get("directive_id") or ""),
                accepted=bool(p.get("accepted")),
                applied=bool(p.get("applied")),
                note=str(p.get("note") or "")[:500],
                forecast_id=str(p.get("forecast_id") or ""),
                judge_status=str(p.get("judge_status") or "not_invoked"),
                at_unix_ms=ev.at_unix_ms,
            )
        )
    elif ev.kind == KIND_GOODBYE:
        out.goodbye.CopyFrom(
            sv.Goodbye(
                reason=int(p.get("reason") or sv.GOODBYE_REASON_UNSPECIFIED),
                note=str(p.get("note") or ""),
            )
        )
    return out


class EngineSupervisionServicer(sv_grpc.EngineSupervisionServicer):
    def __init__(self) -> None:
        self._sessions: dict[str, queue.Queue] = {}

    def Supervise(self, request_iterator, context) -> Iterator[sv.EngineEvent]:
        try:
            first = next(request_iterator)
        except StopIteration:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"{GURU_NOSUBSCRIBE} stream closed before Subscribe",
            )
            return
        if first.WhichOneof("message") != "subscribe":
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"{GURU_NOSUBSCRIBE} first message must be Subscribe",
            )
            return
        sub = first.subscribe
        epoch, spec_version, spec_sha = _epoch()
        if sub.project and sub.project != PROJECT:
            yield sv.EngineEvent(
                seq=0,
                at_unix_ms=now_ms(),
                epoch=epoch,
                goodbye=sv.Goodbye(
                    reason=sv.GOODBYE_REASON_PROJECT_MISMATCH,
                    note=f"engine project is {PROJECT}",
                ),
            )
            return

        bus = get_bus() or init_bus()
        q = bus.subscribe()
        old = self._sessions.get(sub.supervisor_id)
        if old is not None and old is not q:
            bus.publish_to(
                old,
                KIND_GOODBYE,
                reason=sv.GOODBYE_REASON_SUPERSEDED,
                note=f"superseded by a newer Subscribe from {sub.supervisor_id}",
            )
        self._sessions[sub.supervisor_id] = q
        stop = threading.Event()

        def _pump() -> None:
            try:
                for msg in request_iterator:
                    if stop.is_set():
                        return
                    which = msg.WhichOneof("message")
                    if which in (None, "ack", "heartbeat", "subscribe"):
                        continue
                    if which in ("reclaim_orphan", "escalation", "backlog_transition"):
                        bus.publish_to(
                            q,
                            KIND_DIRECTIVE_RESULT,
                            directive_id=getattr(msg, "directive_id", "") or "",
                            accepted=True,
                            applied=False,
                            note=f"{GURU_OBSERVE} directives=observe; no actuation",
                            judge_status="not_invoked",
                        )
            except Exception as e:  # noqa: BLE001 — reader end is session end
                log.info("supervision: reader ended: %s", e)
            finally:
                stop.set()

        reader = threading.Thread(target=_pump, name="atelier-supervise-pump", daemon=True)
        try:
            yield sv.EngineEvent(
                seq=0,
                at_unix_ms=now_ms(),
                epoch=epoch,
                hello=sv.EngineHello(
                    project=PROJECT,
                    engine_build=_engine_build(),
                    spec_version=spec_version,
                    spec_sha256=spec_sha,
                    boot_unix_ms=int(bus.session * 1000),
                    session=int(bus.session),
                    engine_grpc="127.0.0.1:50251",
                    capabilities=list(CAPABILITIES),
                ),
            )
            replayed: list[SupervisionEvent] = []
            tables: list[str] = []
            if sub.resume_session and int(sub.resume_session) == int(bus.session):
                replayed = bus.ring_since(int(sub.resume_seq))
                tables = ["ring"]
            for ev in replayed:
                yield event_to_proto(ev, epoch)
            yield sv.EngineEvent(
                seq=bus.seq,
                at_unix_ms=now_ms(),
                epoch=epoch,
                replay_complete=sv.ReplayComplete(
                    since_unix_ms=int(sub.since_unix_ms or 0),
                    through_seq=bus.seq,
                    events=len(replayed),
                    tables=tables,
                    clamped=False,
                ),
            )
            bus.publish(
                KIND_SERVING,
                capability="referee",
                alias="atelier",
                actual="serving",
                generation=0,
                settled_unix_ms=now_ms(),
            )
            reader.start()
            while context.is_active() and not stop.is_set():
                try:
                    ev = q.get(timeout=HEARTBEAT_S)
                except queue.Empty:
                    yield sv.EngineEvent(
                        seq=bus.seq,
                        at_unix_ms=now_ms(),
                        epoch=epoch,
                        heartbeat=sv.EngineHeartbeat(
                            at_unix_ms=now_ms(), last_seq=bus.seq, claimed_tasks=0
                        ),
                    )
                    continue
                yield event_to_proto(ev, epoch)
                if ev.kind == KIND_GOODBYE:
                    break
        finally:
            stop.set()
            bus.unsubscribe(q)
            if self._sessions.get(sub.supervisor_id) is q:
                self._sessions.pop(sub.supervisor_id, None)
            log.info("supervision: session closed id=%s", sub.supervisor_id)

    def Backlog(self, request, context) -> sv.BacklogResponse:
        epoch, _, _ = _epoch()
        return sv.BacklogResponse(
            project=PROJECT,
            epoch=epoch,
            supervisor_connected=bool(self._sessions),
        )
