"""Decisions sweep: distribution mapping, genus walk, Decide request shape."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from atelier.classify.belief import FrameOfDiscernment
from atelier.classify.decisions_backend import (
    DecisionsBackend,
    REPLICA_COUNT,
    ReplayDecisionsBackend,
    choice_distribution,
    entity_state,
    select_shard_columns,
)
from atelier.classify.decisions_walk import (
    WalkError,
    classification_from_distribution,
    structural_genera,
    walk_entity,
)
from atelier.classify.mass_functions import llm_to_mass
from atelier.classify.sampler import ColumnSample
from atelier.classify.taxonomy import HierarchicalCategorySet, ReferenceCategory
from atelier.engine.forwarder import NoPeerServes
from atelier.flows.lattice import GURU_NOCAP, decide


def _cat(code: str, parent: str | None, label: str) -> ReferenceCategory:
    return ReferenceCategory(
        code=code, label=label, embedding_text=label,
        description=label, parent_code=parent,
    )


def _taxonomy() -> HierarchicalCategorySet:
    cats = [
        _cat("SDG", None, "root"),
        _cat("SDG.A", "SDG", "genus A"),
        _cat("SDG.A.1", "SDG.A", "leaf A1"),
        _cat("SDG.A.2", "SDG.A", "leaf A2"),
        _cat("SDG.B", "SDG", "genus B"),
        _cat("SDG.B.1", "SDG.B", "leaf B1"),
    ]
    return HierarchicalCategorySet(name="sdg", categories=cats)


def test_distribution_maps_top1_and_three_alternatives() -> None:
    code, confidence, alts = classification_from_distribution({
        "SDG.A": 0.5,
        "SDG.B": 0.2,
        "SDG.C": 0.15,
        "SDG.D": 0.1,
        "SDG.E": 0.05,
    })
    assert code == "SDG.A"
    assert confidence == 0.5
    assert [a["code"] for a in alts] == ["SDG.B", "SDG.C", "SDG.D"]


def test_parent_stop_wins_the_choice() -> None:
    seen: list[list[str]] = []

    def ask(criteria, instructions):
        codes = list(criteria)
        seen.append(codes)
        if codes == ["SDG.A", "SDG.B"]:
            return {"SDG.A": 0.8, "SDG.B": 0.2}
        if "SDG.A" in codes and "SDG.A.1" in codes:
            return {"SDG.A": 0.7, "SDG.A.1": 0.2, "SDG.A.2": 0.1}
        return {"SDG.B": 0.9, "SDG.B.1": 0.1}

    result = walk_entity(_taxonomy(), ask, floor=0.05)
    assert result.category_code == "SDG.A"
    assert result.confidence == 0.7
    assert "SDG.B" in seen[0]
    assert any("SDG.B.1" in codes for codes in seen)


def test_code_absent_from_the_frame_is_vacuous() -> None:
    vocab = _taxonomy()
    frame = FrameOfDiscernment(vocab)
    code, confidence, alts = classification_from_distribution({"NOT.A.CODE": 0.9, "SDG.A": 0.1})
    assert code == "NOT.A.CODE"
    mass = llm_to_mass(code, confidence, alts, frame, discount=0.15)
    assert mass.masses == frame.vacuous().masses


def test_genera_are_nodes_with_a_parent_and_children() -> None:
    genera = structural_genera(_taxonomy())
    assert genera == ["SDG.A", "SDG.B"]
    assert "SDG" not in genera


def test_opening_choice_over_255_fails_closed() -> None:
    cats = [_cat("ROOT", None, "root")]
    for i in range(256):
        code = f"ROOT.G{i}"
        cats.append(_cat(code, "ROOT", code))
        cats.append(_cat(f"{code}.L", code, "leaf"))
    vocab = HierarchicalCategorySet(name="wide", categories=cats)
    with pytest.raises(WalkError, match="255"):
        walk_entity(vocab, lambda criteria, instructions: {})


def test_empty_taxonomy_fails_closed() -> None:
    vocab = HierarchicalCategorySet(name="empty", categories=[])
    with pytest.raises(WalkError, match="empty"):
        walk_entity(vocab, lambda criteria, instructions: {})


def test_two_genera_above_the_floor_are_both_asked_and_the_budget_stops() -> None:
    calls = {"n": 0}

    def ask(criteria, instructions):
        calls["n"] += 1
        codes = list(criteria)
        if set(codes) == {"SDG.A", "SDG.B"}:
            return {"SDG.A": 0.6, "SDG.B": 0.4}
        return {codes[0]: 1.0}

    stopped = walk_entity(_taxonomy(), ask, floor=0.05, max_calls=2)
    assert calls["n"] == 2
    assert stopped.truncated is True
    assert stopped.category_code == "SDG.A"


def test_backend_maps_one_entity_and_refuses_a_revisit() -> None:
    def decide(state, questions, **kw):
        assert json.loads(json.dumps(state))
        spec = questions["choice"]
        assert spec["type"] == "choice"
        criteria = spec["criteria"]
        if set(criteria) == {"SDG.A", "SDG.B"}:
            answers = {"choice": {"SDG.A": 0.9, "SDG.B": 0.1}}
        else:
            winner = next(iter(criteria))
            answers = {"choice": {winner: 1.0}}

        return SimpleNamespace(
            model="vllm-sr/d3-flash",
            answers=answers,
            input_tokens=2,
            output_tokens=1,
        )

    backend = DecisionsBackend(decide_fn=decide, category_set=_taxonomy(), descend_floor=0.5)
    sample = ColumnSample(name="amount", column_type="int", values=["10"])
    response = backend.classify_batch([sample], "unused")
    assert response.classifications[0].category_code == "SDG.A"
    assert response.classifications[0].evidence == ""
    assert entity_state(sample).startswith("amount ")
    with pytest.raises(RuntimeError, match="thinking revisit"):
        backend.classify_batch([sample], "unused", revisit_context={"amount": {}})


def test_live_choice_object_uses_probabilities_not_the_confidence_field() -> None:
    """Engine/Decide returns {type, choice, probabilities, confidence}.

    The sweep confidence is probabilities[chosen option]. The tiny
    confidence field on the choice object is a different score.
    """
    live = {
        "choice": {
            "type": "choice",
            "choice": "SDG.B",
            "probabilities": {
                "SDG.A": 0.49720050402878213,
                "SDG.B": 0.5027994959712179,
            },
            "confidence": 2.2613432935192357e-05,
        }
    }
    assert choice_distribution(live)["SDG.B"] == pytest.approx(0.5027994959712179)
    assert "confidence" not in choice_distribution(live)
    assert "choice" not in choice_distribution(live)

    def decide(state, questions, **kw):
        return SimpleNamespace(
            model="vllm-sr/d3-flash",
            answers=live,
            input_tokens=2,
            output_tokens=1,
        )

    # Floor above both genera so the opening distribution is the label.
    backend = DecisionsBackend(
        decide_fn=decide, category_set=_taxonomy(), descend_floor=0.99,
    )
    sample = ColumnSample(name="invoice_id", column_type="string", values=["INV-1001"])
    row = backend.classify_batch([sample], "unused").classifications[0]
    assert row.category_code == "SDG.B"
    assert row.confidence == pytest.approx(0.5027994959712179)
    assert row.alternatives[0]["code"] == "SDG.A"
    assert row.alternatives[0]["confidence"] == pytest.approx(0.49720050402878213)


def test_structured_choice_without_probabilities_fail_closes() -> None:
    answers = {
        "choice": {
            "type": "choice",
            "choice": "SDG.B",
            "confidence": 2.2613432935192357e-05,
        }
    }
    with pytest.raises(RuntimeError, match="no choice distribution"):
        choice_distribution(answers)


def test_decide_request_is_one_choice_and_fail_closes_without_a_peer() -> None:
    seen: dict = {}

    class _Route:
        label = "gaius@127.0.0.1:50051"

    class _Fwd:
        def resolve(self, cap):
            assert cap == "decisions"
            return _Route()

        def decide(self, req, timeout=None):
            seen["req"] = req
            class _Resp:
                id = "d1"
                model = "vllm-sr/d3-flash"
                answers_json = json.dumps({"choice": {"SDG.A": 0.8, "SDG.B": 0.2}})
                diagnostics_json = "{}"
                input_tokens = 4
                output_tokens = 1
                latency_ms = 3.0
                fulfilled_by = "direct@d3-flash-0/vllm-sr/d3-flash"

            return _Resp()

    result = decide("amount type=int values=[10]", {
        "choice": {
            "type": "choice",
            "instructions": "Choose the genus for this entity.",
            "criteria": {"SDG.A": "genus A", "SDG.B": None},
        },
    }, forwarder=_Fwd())
    req = seen["req"]
    assert req.capability == "decisions"
    assert json.loads(req.state_json) == "amount type=int values=[10]"
    assert req.cache_salt == ""
    assert req.instructions == ""
    assert len(req.questions) == 1
    question = req.questions[0]
    assert question.type == "choice"
    assert json.loads(question.criteria_json)["SDG.B"] is None
    assert result.model == "vllm-sr/d3-flash"
    assert result.answers["choice"]["SDG.A"] == 0.8

    class _Missing:
        def resolve(self, cap):
            raise NoPeerServes(cap, ["gaius@none"])

    with pytest.raises(RuntimeError, match=GURU_NOCAP):
        decide("x", {"choice": {"type": "choice", "criteria": {"A": None}}}, forwarder=_Missing())


def test_five_entities_split_three_and_two() -> None:
    from atelier.classify.sampler import TableSample

    billing = TableSample(name="billing", columns=[
        ColumnSample(name="invoice_id", table_name="billing"),
        ColumnSample(name="amount", table_name="billing"),
        ColumnSample(name="note", table_name="billing"),
    ])
    other = TableSample(name="other", columns=[
        ColumnSample(name="row_id", table_name="other"),
        ColumnSample(name="status", table_name="other"),
    ])
    shard0 = select_shard_columns([billing, other], 0, REPLICA_COUNT)
    shard1 = select_shard_columns([billing, other], 1, REPLICA_COUNT)
    assert [col.name for _table, col in shard0] == ["invoice_id", "note", "status"]
    assert [col.name for _table, col in shard1] == ["amount", "row_id"]


def test_parallel_walks_preserve_input_order() -> None:
    def decide(state, questions, **_kwargs):
        return SimpleNamespace(
            answers={"choice": {"SDG.A": 0.9, "SDG.B": 0.1}},
            model="vllm-sr/d3-flash",
            input_tokens=1,
            output_tokens=1,
        )

    backend = DecisionsBackend(
        decide_fn=decide,
        category_set=_taxonomy(),
        descend_floor=0.99,
        parallel=2,
    )
    response = backend.classify_batch([
        ColumnSample(name="invoice_id", values=["INV"]),
        ColumnSample(name="amount", values=["10"]),
    ], "")
    assert [item.column_name for item in response.classifications] == ["invoice_id", "amount"]
    assert backend._parallel == 2


def test_from_cfg_parallel_defaults_to_one_shard() -> None:
    cfg = SimpleNamespace(
        classify_decisions_descend_floor=0.05,
        classify_decisions_max_calls=64,
    )
    assert DecisionsBackend.from_cfg(cfg)._parallel == 1
    assert DecisionsBackend.from_cfg(cfg, parallel=REPLICA_COUNT)._parallel == 2


def test_replay_returns_the_stored_code_without_decide() -> None:
    backend = ReplayDecisionsBackend([{
        "qualified_name": "billing.invoice_id",
        "column_name": "invoice_id",
        "predicted_code": "SDG.A.1",
        "confidence": 0.91,
        "alternatives": [{"code": "SDG.A", "confidence": 0.04}],
    }])
    sample = ColumnSample(name="invoice_id", table_name="billing", values=["INV-1"])
    response = backend.classify_batch([sample], "", table_name="billing")
    got = response.classifications[0]
    assert got.category_code == "SDG.A.1"
    assert got.column_name == "invoice_id"
    assert got.confidence == pytest.approx(0.91)
    with pytest.raises(RuntimeError, match=GURU_NOCAP):
        backend.classify_batch([sample], "", revisit_context={"invoice_id": {}})
    missing = ColumnSample(name="other", table_name="billing")
    with pytest.raises(RuntimeError, match="no decisions label"):
        backend.classify_batch([missing], "", table_name="billing")
