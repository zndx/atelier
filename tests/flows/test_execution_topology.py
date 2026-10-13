"""Imperative classify configuration is the Airflow execution topology."""

from __future__ import annotations

import json

import pytest

from atelier.engine.queue_share import LIGHT
from atelier.engine.workload_catalog import (
    DAG_ID,
    DECISIONS_MODEL,
    KIND,
    THINKING_MODEL,
    classify_yk_claims,
)
from atelier.flows.execution_topology import (
    classify_topology,
    configuration,
    configuration_from_reason,
    metaflow_argv,
    reason_for,
)
from atelier.flows.platform_metaflow import argv_with_kubernetes


def test_imperative_topology_matches_the_catalogue() -> None:
    topo = classify_topology(source_id="sdg-corpora/pin_macbook", target="collection:clinic")
    assert topo.trigger == "imperative"
    assert topo.cron == ""
    assert topo.runner == "metaflow"
    assert topo.kind == KIND
    assert topo.dag_id == DAG_ID
    assert topo.sweep_model == DECISIONS_MODEL
    assert topo.sweep_capability == "decisions"
    assert topo.sweep_width == 2
    assert topo.revisit_model == THINKING_MODEL
    assert topo.revisit_capability == "thinking"
    assert topo.claims == classify_yk_claims()
    assert topo.claims == ((LIGHT.queue, 1),)
    assert topo.precludes == (
        "root.internal.inference.agent-rtc",
        "root.internal.inference.medium",
        "root.internal.inference.extract",
    )
    conf = configuration(topo)
    assert conf["source_id"] == "sdg-corpora/pin_macbook"
    assert conf["target"] == "collection:clinic"
    assert configuration_from_reason(reason_for(topo)) == conf


def test_scheduled_trigger_keeps_an_empty_catalogue_cron() -> None:
    topo = classify_topology(source_id="sdg-corpora/pin_macbook", trigger="schedule")
    assert topo.trigger == "schedule"
    assert topo.cron == ""


def test_imperative_with_a_cron_is_refused() -> None:
    from atelier.flows.execution_topology import ExecutionTopology

    with pytest.raises(ValueError, match="no cron"):
        ExecutionTopology(
            peer="atelier",
            kind=KIND,
            dag_id=DAG_ID,
            runner="metaflow",
            trigger="imperative",
            cron="0 8 * * *",
            source_id="s",
            target="",
            reference_id="",
            only_precondition=False,
            sweep_capability="decisions",
            sweep_model=DECISIONS_MODEL,
            sweep_width=2,
            revisit_capability="thinking",
            revisit_model=THINKING_MODEL,
            claims=(("root.internal.inference.light", 1),),
            precludes=(),
            horizon_s=10,
        )


def test_metaflow_argv_is_the_topology_and_kubernetes_is_the_decorator() -> None:
    topo = classify_topology(
        source_id="sdg-corpora/pin_macbook",
        target="table:billing",
        reference_id="ref-1",
        only_precondition=True,
    )
    argv = metaflow_argv(topo)
    assert argv[:5] == [
        "run",
        "--source-id=sdg-corpora/pin_macbook",
        "--target=table:billing",
        "--reference-id=ref-1",
        "--only-precondition=true",
    ]
    assert argv[5].startswith("--run-configuration=")
    blob = argv[5].split("=", 1)[1]
    assert configuration_from_reason(blob) == configuration(topo)
    decorated = argv_with_kubernetes(["run_classify.py", *argv])
    assert decorated[:3] == ["run_classify.py", "--with", "kubernetes"]
    assert "--source-id=sdg-corpora/pin_macbook" in decorated


def test_reason_round_trip_is_stable() -> None:
    topo = classify_topology(source_id="sdg-corpora/pin_macbook")
    raw = reason_for(topo)
    assert raw == json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError, match="not the run configuration"):
        configuration_from_reason("a person clicked start")
