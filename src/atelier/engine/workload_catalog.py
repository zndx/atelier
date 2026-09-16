"""Atelier WORKLOAD CATALOGUE — sdg_classify, AgentRTC-shaped claims.

One class. enabled=False until a K8s Metaflow run is proven on the
discovered Signals instance. Live YK: ``root.internal.inference.embedding``
has GPU max 0 (no resources.max), so ColBERT-Zero and ModernBERT serialize
on ``light`` (max 2, standing floor 0 — arbiter raises while the Activity
is in force, same shape as agent-rtc). thinking/instruct Complete uses
standing heavy occupancy — not a claim here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atelier.engine.queue_share import LIGHT

PEER = "atelier"
KIND = "sdg_classify"
DAG_ID = "atelier_sdg_classify"
RUNNER_METAFLOW = "metaflow"
CRON = "0 8 * * *"
HORIZON_S = 14400  # 4h — textproto net_seconds


def classify_yk_claims() -> tuple[tuple[str, int], ...]:
    """YK leaves this workflow expects. Phase RequestQueueShare may zero a floor.

    One light GPU: ColBERT-Zero encode and ModernBERT NHSVM fit take turns.
    Do not claim the embedding leaf — live policy has GPU max 0 there.
    """
    return ((LIGHT.queue, 1),)


@dataclass(frozen=True)
class WorkloadEntry:
    kind: str
    cron: str
    description: str
    runner: str = RUNNER_METAFLOW
    horizon_s: int = HORIZON_S
    enabled: bool = False
    airflow_dag_id: str = ""
    claims: tuple[tuple[str, int], ...] = field(default_factory=tuple)

    @property
    def id(self) -> str:
        return f"task.{self.kind}"

    @property
    def source(self) -> str:
        return "airflow" if self.enabled else "engine"


WORKLOAD_CATALOG: tuple[WorkloadEntry, ...] = (
    WorkloadEntry(
        kind=KIND,
        cron=CRON,
        description=(
            "SDG classification: probe-gated ColBERT-Zero MaxSim + ModernBERT "
            "NHSVM + DST; Complete instruct/thinking"
        ),
        runner=RUNNER_METAFLOW,
        horizon_s=HORIZON_S,
        enabled=False,
        airflow_dag_id=DAG_ID,
        claims=classify_yk_claims(),
    ),
)


def entries() -> tuple[WorkloadEntry, ...]:
    return WORKLOAD_CATALOG


def to_hint(entry: WorkloadEntry) -> Any:
    from zndx.engine.v1 import engine_pb2 as zpb

    hint = zpb.ScheduleHint(
        id=entry.id,
        cron=entry.cron,
        airflow_dag_id=entry.airflow_dag_id or f"{PEER}_{entry.kind}",
        source=entry.source,
        enabled=bool(entry.enabled),
        kind=entry.kind,
        horizon_s=int(entry.horizon_s),
        description=entry.description,
        runner=entry.runner,
    )
    for leaf, gpu in entry.claims:
        hint.claims.add(leaf=str(leaf), gpu=int(gpu))
    return hint


def schedule_hints() -> list[Any]:
    return [to_hint(e) for e in WORKLOAD_CATALOG]
