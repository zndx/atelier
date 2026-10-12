"""ClassificationFlow graph, coverage pass gate, Complete capabilities."""

from __future__ import annotations

import pytest

from atelier.flows.classify import ClassificationFlow
from atelier.flows.classify_phases import (
    STEP_GRAPH,
    assert_coverage,
    coverage_complete,
    llm_capability_for_sweep,
    merge_shard_labels,
    probe_requires_precondition,
    unclassified_targets,
)
from atelier.flows.resident import load_classification_rows
from atelier.flows.lattice import GURU_NOCAP, assert_complete_capability
from atelier.flows.platform_metaflow import (
    GURU_NOPLATFORM,
    argv_with_kubernetes,
    require_signals_metaflow,
)


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
    # The two System One replicas are a foreach, visible as two tasks.
    assert graph.nodes["sweep"].type == "foreach"
    assert graph.nodes["sweep_shard"].type == "linear"
    assert graph.nodes["sweep_join"].type == "join"


def test_merge_shard_labels_sorts_and_rejects_overlap() -> None:
    merged = merge_shard_labels([
        [{"qualified_name": "billing.note", "predicted_code": "SDG.B"}],
        [{"qualified_name": "billing.invoice_id", "predicted_code": "SDG.A.1"}],
    ])
    assert [row["qualified_name"] for row in merged] == [
        "billing.invoice_id", "billing.note",
    ]
    with pytest.raises(RuntimeError, match="overlapped"):
        merge_shard_labels([
            [{"qualified_name": "billing.note", "predicted_code": "SDG.B"}],
            [{"qualified_name": "billing.note", "predicted_code": "SDG.A"}],
        ])


def test_airflow_create_argv_requests_kubernetes() -> None:
    argv = argv_with_kubernetes(["run_classify.py", "airflow", "create", "dag.py"])
    assert argv == [
        "run_classify.py", "--with", "kubernetes", "airflow", "create", "dag.py",
    ]
    kept = argv_with_kubernetes(["run_classify.py", "--with", "batch", "run"])
    assert kept == ["run_classify.py", "--with", "batch", "run"]


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


def test_sweep_uses_decisions_revisit_uses_thinking() -> None:
    assert llm_capability_for_sweep(revisit=False) == "decisions"
    assert llm_capability_for_sweep(revisit=True) == "thinking"
    assert_complete_capability("instruct")
    assert_complete_capability("thinking")
    with pytest.raises(RuntimeError, match=GURU_NOCAP):
        assert_complete_capability("decisions")
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


def test_run_dst_pipeline_binds_reference_taxonomy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class _BE:
        @classmethod
        def from_cfg(cls, cfg):
            return "backend"

    def fake_pipeline(cfg, fsm, **kw):
        captured["source_id"] = kw.get("source_id")
        captured["taxonomy_id"] = kw.get("taxonomy_id")
        captured["category_set"] = kw.get("category_set")
        captured["cfg_tax"] = getattr(cfg, "classify_taxonomy_id", None)
        captured["llm"] = kw.get("llm_backend")
        captured["include_revisit"] = kw.get("include_revisit")
        return {"state": "OK"}

    monkeypatch.setattr(
        "atelier.classify.pipeline.run_classification_pipeline", fake_pipeline,
    )
    monkeypatch.setattr("atelier.classify.get_fsm", lambda dao: object())
    monkeypatch.setattr("atelier.db.dao.AtelierDao", lambda: object())
    monkeypatch.setattr(
        "atelier.flows.resident.load_category_set",
        lambda cfg, sid: {"vocab": sid},
    )
    monkeypatch.setattr(
        "atelier.classify.complete_backend.CompleteLLMBackend", _BE,
    )
    from atelier.config import AtelierConfig
    from atelier.flows.resident import run_dst_pipeline

    run_dst_pipeline(
        AtelierConfig(),
        "sdg-corpora/b24ef9f60660_macbook",
        skip_precondition=True,
        taxonomy_id="sdg-corpora/b24ef9f60660_reference",
    )
    assert captured["source_id"] == "sdg-corpora/b24ef9f60660_macbook"
    assert captured["taxonomy_id"] == "sdg-corpora/b24ef9f60660_reference"
    assert captured["cfg_tax"] == "sdg-corpora/b24ef9f60660_reference"
    assert captured["category_set"] == {"vocab": "sdg-corpora/b24ef9f60660_reference"}
    assert captured["llm"] is None
    assert captured["include_revisit"] is True

    run_dst_pipeline(
        AtelierConfig(),
        "sdg-corpora/b24ef9f60660_macbook",
        skip_precondition=True,
        taxonomy_id="sdg-corpora/b24ef9f60660_reference",
        include_revisit=False,
    )
    assert captured["include_revisit"] is False

    run_dst_pipeline(
        AtelierConfig(),
        "sdg-corpora/b24ef9f60660_macbook",
        skip_precondition=True,
        include_revisit=False,
        llm_backend="replay",
    )
    assert captured["llm"] == "replay"
    assert captured["include_revisit"] is False


def test_run_decisions_shard_assigns_three_and_two(monkeypatch: pytest.MonkeyPatch) -> None:
    from atelier.classify.llm_backend import ColumnClassification, LLMResponse
    from atelier.classify.sampler import ColumnSample, TableSample
    from atelier.flows.resident import run_decisions_shard

    tables = [
        TableSample(name="billing", columns=[
            ColumnSample(name="invoice_id", table_name="billing", values=["INV"]),
            ColumnSample(name="amount", table_name="billing", values=["10"]),
            ColumnSample(name="note", table_name="billing", values=["x"]),
        ]),
        TableSample(name="other", columns=[
            ColumnSample(name="row_id", table_name="other", values=["1"]),
            ColumnSample(name="status", table_name="other", values=["open"]),
        ]),
    ]
    monkeypatch.setattr(
        "atelier.flows.resident.load_sweep_samples", lambda cfg, sid: tables,
    )
    monkeypatch.setattr(
        "atelier.flows.resident.load_category_set", lambda cfg, sid: object(),
    )

    class _Backend:
        @classmethod
        def from_cfg(cls, cfg, *, parallel: int = 1):
            assert parallel == 1
            return cls()

        def health_check(self) -> bool:
            return True

        def bind_taxonomy(self, category_set) -> None:
            return None

        def classify_batch(self, samples, system_prompt, revisit_context=None, table_name=None):
            return LLMResponse(
                classifications=[
                    ColumnClassification(
                        column_name=sample.name,
                        category_code="SDG.A",
                        confidence=0.8,
                        evidence="",
                    )
                    for sample in samples
                ],
                input_tokens=0,
                output_tokens=0,
                model="vllm-sr/d3-flash",
            )

    monkeypatch.setattr(
        "atelier.classify.decisions_backend.DecisionsBackend", _Backend,
    )
    shard0 = run_decisions_shard(object(), "sample", 0, width=2, taxonomy_id="tax")
    shard1 = run_decisions_shard(object(), "sample", 1, width=2, taxonomy_id="tax")
    assert [row["qualified_name"] for row in shard0] == [
        "billing.invoice_id", "billing.note", "other.status",
    ]
    assert [row["qualified_name"] for row in shard1] == [
        "billing.amount", "other.row_id",
    ]


def test_start_refuses_local_datastore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATELIER_METAFLOW_MODE", "local")
    monkeypatch.setenv("METAFLOW_DEFAULT_DATASTORE", "local")
    with pytest.raises(Exception, match=GURU_NOPLATFORM):
        require_signals_metaflow()
