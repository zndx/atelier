"""In-process ring for EngineSupervision.Supervise. No Postgres.

Sync (Atelier engine is ThreadPoolExecutor, not grpc.aio). publish() never
raises. Slow subscribers drop oldest (#SV.00000002.BUSDROP).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("atelier.engine.supervision_bus")

GURU_BUSDROP = "#SV.00000002.BUSDROP"
QUEUE_MAXSIZE = 512
RING_MAXLEN = 4096

KIND_SERVING = "serving"
KIND_DIRECTIVE_RESULT = "directive_result"
KIND_GOODBYE = "goodbye"
KIND_ACTIVITY = "activity"
KIND_POSITION = "position"

_lock = threading.Lock()
_bus: SupervisionBus | None = None


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class SupervisionEvent:
    seq: int
    at_unix_ms: int
    kind: str
    payload: dict[str, Any]
    replayed: bool = False


@dataclass
class SupervisionBus:
    _seq: int = 0
    _subs: dict[int, queue.Queue] = field(default_factory=dict)
    _ring: deque = field(default_factory=lambda: deque(maxlen=RING_MAXLEN))
    session: int = field(default_factory=lambda: int(time.time()))

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._subs[id(q)] = q
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        self._subs.pop(id(q), None)

    def publish(self, kind: str, **payload: Any) -> SupervisionEvent:
        self._seq += 1
        ev = SupervisionEvent(
            seq=self._seq,
            at_unix_ms=now_ms(),
            kind=kind,
            payload=dict(payload),
        )
        self._ring.append(ev)
        for q in list(self._subs.values()):
            try:
                q.put_nowait(ev)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(ev)
                    log.warning("%s dropped oldest for a slow Supervise subscriber", GURU_BUSDROP)
                except queue.Empty:
                    pass
        return ev

    def publish_to(self, q: queue.Queue, kind: str, **payload: Any) -> None:
        self._seq += 1
        ev = SupervisionEvent(
            seq=self._seq, at_unix_ms=now_ms(), kind=kind, payload=dict(payload)
        )
        try:
            q.put_nowait(ev)
        except queue.Full:
            pass

    def ring_since(self, seq: int) -> list[SupervisionEvent]:
        return [e for e in self._ring if e.seq > seq]

    @property
    def seq(self) -> int:
        return self._seq


def init_bus() -> SupervisionBus:
    global _bus
    with _lock:
        if _bus is None:
            _bus = SupervisionBus()
        return _bus


def get_bus() -> SupervisionBus | None:
    return _bus
