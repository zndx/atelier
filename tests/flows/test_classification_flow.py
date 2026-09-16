"""ClassificationFlow graph, coverage pass gate, Complete capabilities."""

from __future__ import annotations

import pytest

from atelier.flows.classify import ClassificationFlow
from atelier.flows.classify_phases import (
    STEP_GRAPH,
    assert_coverage,
    coverage_complete,
    llm_capability_for_sweep,
    probe_requires_precondition,
    unclassified_targets,
)
from atelier.flows.resident import load_classification_rows
from atelier.flows.lattice import GURU_NOCAP, assert_complete_capability
from atelier.flows.platform_metaflow import GURU_NOPLATFORM, require_signals_metaflow


def test_flow_graph_matches_resident_sequence() -> None:
    graph = ClassificationFlow._graph
    assert set(graph.nodes) == {step for step, _ in STEP_GRAPH}
    for step, nxt in STEP_GRAPH:
        node = graph.nodes[step]
        outs = list(node.out_funcs or [])
        if nxt is None:
            assert outs == []
        else:
            assert nxt in outs


def test_coverage_complete_requires_every_target() -> None:
    rows = [
        {"qualified_name": "t.a", "predicted_code": "1.1"},
        {"qualified_name": "t.b", "predicted_code": "2.0"},
    ]
    assert coverage_complete(rows, ["t.a", "t.b"])
    assert not coverage_complete(rows, ["t.a", "t.b", "t.c"])
    assert unclassified_targets(rows, ["t.a", "t.b", "t.c"]) == ["t.c"]
    assert unclassified_targets(
        [{"qualified_name": "t.a", "predicted_code": ""}],
        ["t.a"],
    ) == ["t.a"]


def test_empty_predicted_code_is_not_classified() -> None:
    rows = [{"column_name": "x", "predicted_code": None}]
    assert not coverage_complete(rows, ["x"])


def test_probe_skips_expensive_precondition_when_final() -> None:
    assert not probe_requires_precondition({"final": True})
    assert probe_requires_precondition({"final": False})


def test_sweep_uses_instruct_revisit_uses_thinking() -> None:
    assert llm_capability_for_sweep(revisit=False) == "instruct"
    assert llm_capability_for_sweep(revisit=True) == "thinking"
    assert_complete_capability("instruct")
    assert_complete_capability("thinking")
    with pytest.raises(RuntimeError, match=GURU_NOCAP):
        assert_complete_capability("referee")


def test_evaluate_fails_when_a_target_is_missing() -> None:
    rows = [{"qualified_name": "t.a", "predicted_code": "1"}]
    with pytest.raises(RuntimeError, match="coverage incomplete"):
        assert_coverage(rows, ["t.a", "t.b"])
    assert_coverage(rows, ["t.a"])


def test_load_classification_rows_from_result_dir(tmp_path) -> None:
    (tmp_path / "classifications.json").write_text(
        '[{"qualified_name": "t.a", "predicted_code": "1.1"}]',
        encoding="utf-8",
    )
    rows = load_classification_rows({"result_path": str(tmp_path)})
    assert rows[0]["predicted_code"] == "1.1"


def test_start_refuses_local_datastore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATELIER_METAFLOW_MODE", "local")
    monkeypatch.setenv("METAFLOW_DEFAULT_DATASTORE", "local")
    with pytest.raises(Exception, match=GURU_NOPLATFORM):
        require_signals_metaflow()
