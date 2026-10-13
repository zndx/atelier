"""Run configuration for one Airflow-triggered Metaflow workflow.

Signals materialises the workload catalogue as Airflow DAGs and is the
only process that talks to Airflow. An imperative run is one activity:
the product engine declares it, Airflow triggers it, and the Metaflow
flow reads this topology. Process endpoints (database, ports) stay in
HOCON. The run's source, model, capability, claims, and trigger do not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from atelier.engine.workload_catalog import (
    DAG_ID,
    HORIZON_S,
    KIND,
    PEER,
    RUNNER_METAFLOW,
    classify_yk_claims,
    decisions_sweep_record,
    entries,
)


@dataclass(frozen=True)
class ExecutionTopology:
    """One Airflow execution of a Metaflow workflow.

    ``trigger="imperative"`` is a user declare (``schedule=None``).
    ``trigger="schedule"`` is the catalogue cron. Both carry the same
    run configuration.
    """

    peer: str
    kind: str
    dag_id: str
    runner: str
    trigger: str
    cron: str
    source_id: str
    target: str
    reference_id: str
    only_precondition: bool
    sweep_capability: str
    sweep_model: str
    sweep_width: int
    revisit_capability: str
    revisit_model: str
    claims: tuple[tuple[str, int], ...]
    precludes: tuple[str, ...]
    horizon_s: int

    def __post_init__(self) -> None:
        if self.runner != RUNNER_METAFLOW:
            raise ValueError(
                f"workflow {self.kind} runner is {self.runner!r}; "
                "Signals schedules Metaflow"
            )
        if self.trigger == "imperative" and self.cron:
            raise ValueError(
                "an imperative Metaflow run has no cron; Airflow triggers it"
            )
        if self.trigger not in {"imperative", "schedule"}:
            raise ValueError(f"unknown trigger {self.trigger!r}")
        if not self.claims:
            raise ValueError(f"workflow {self.kind} has no queue claims")


def classify_topology(
    *,
    source_id: str,
    target: str = "",
    reference_id: str = "",
    only_precondition: bool = False,
    trigger: str = "imperative",
) -> ExecutionTopology:
    """Classify topology from the catalogue entry plus the trigger arguments.

    The sweep record and ``classify_yk_claims`` are that entry's data.
    A caller does not pass model names or queue leaves of its own.
    """
    entry = entries()[0]
    if entry.kind != KIND:
        raise ValueError(f"catalogue head is {entry.kind}, expected {KIND}")
    record = decisions_sweep_record()
    sweep = record["sweep"]
    revisit = record["revisit"]
    replicas = list(sweep.get("replicas") or [])
    cron = "" if trigger == "imperative" else str(entry.cron or "")
    return ExecutionTopology(
        peer=PEER,
        kind=entry.kind,
        dag_id=entry.airflow_dag_id or DAG_ID,
        runner=entry.runner,
        trigger=trigger,
        cron=cron,
        source_id=str(source_id or "").strip(),
        target=str(target or "").strip(),
        reference_id=str(reference_id or "").strip(),
        only_precondition=bool(only_precondition),
        sweep_capability=str(sweep["capability"]),
        sweep_model=str(sweep["model"]),
        sweep_width=len(replicas) or 1,
        revisit_capability=str(revisit["capability"]),
        revisit_model=str(revisit["model"]),
        claims=tuple(entry.claims or classify_yk_claims()),
        precludes=tuple(entry.precludes),
        horizon_s=int(entry.horizon_s or HORIZON_S),
    )


def configuration(topo: ExecutionTopology) -> dict[str, Any]:
    """JSON object Airflow stores as the activity and Metaflow reads back.

    Keys are the run. Adding a knob means adding it here, not reading a
    second settings overlay inside a step.
    """
    return {
        "peer": topo.peer,
        "kind": topo.kind,
        "dag_id": topo.dag_id,
        "runner": topo.runner,
        "trigger": topo.trigger,
        "cron": topo.cron,
        "source_id": topo.source_id,
        "target": topo.target,
        "reference_id": topo.reference_id,
        "only_precondition": topo.only_precondition,
        "sweep_capability": topo.sweep_capability,
        "sweep_model": topo.sweep_model,
        "sweep_width": topo.sweep_width,
        "revisit_capability": topo.revisit_capability,
        "revisit_model": topo.revisit_model,
        "claims": [[leaf, gpu] for leaf, gpu in topo.claims],
        "precludes": list(topo.precludes),
        "horizon_s": topo.horizon_s,
    }


def reason_for(topo: ExecutionTopology) -> str:
    """Activity ``reason``: the run configuration, sorted for a stable conf."""
    return json.dumps(configuration(topo), sort_keys=True, separators=(",", ":"))


def configuration_from_reason(reason: str) -> dict[str, Any]:
    """Parse the configuration Airflow echoed on the activity."""
    try:
        doc = json.loads(reason or "")
    except json.JSONDecodeError as exc:
        raise ValueError("activity reason is not the run configuration") from exc
    if not isinstance(doc, dict) or doc.get("runner") != RUNNER_METAFLOW:
        raise ValueError("activity reason is not a Metaflow execution topology")
    return doc


def argv_from_configuration(conf: dict[str, Any]) -> list[str]:
    """Metaflow argv for one frozen configuration document.

    ``run_classify`` inserts ``--with kubernetes``. ``--run-configuration``
    is the activity reason, so the flow does not re-read the catalogue.
    """
    argv = [
        "run",
        f"--source-id={conf.get('source_id') or ''}",
        f"--target={conf.get('target') or ''}",
        f"--reference-id={conf.get('reference_id') or ''}",
    ]
    if conf.get("only_precondition"):
        argv.append("--only-precondition=true")
    blob = json.dumps(conf, sort_keys=True, separators=(",", ":"))
    argv.append(f"--run-configuration={blob}")
    return argv


def metaflow_argv(topo: ExecutionTopology) -> list[str]:
    """Arguments the owner passes when the activity is RUNNING.

    The flow does not open Airflow.
    """
    return argv_from_configuration(configuration(topo))
