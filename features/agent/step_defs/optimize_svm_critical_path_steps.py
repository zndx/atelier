"""Step defs for optimize_svm_critical_path.feature — the SVM/maxsim critical-
path efficacy sequence on the test-gittables fixture (@slow / @tier-1; needs
the devenv stack: registry DB + ModernBERT + Qdrant)."""
from __future__ import annotations

import json
import os
from pathlib import Path

from behave import given, then, when

ENCODER = "answerdotai/ModernBERT-base"


def _heldout() -> list[dict]:
    import atelier.classify
    p = (Path(atelier.classify.__file__).parent / "fixtures" / "test-gittables"
         / "heldout_rows.jsonl")
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _cfg_for(taxonomy_id: str):
    os.environ["ATELIER_CLASSIFY_TAXONOMY_ID"] = taxonomy_id
    os.environ["ATELIER_CLASSIFY_SVM_ENABLED"] = "true"
    os.environ["ATELIER_CLASSIFY_SVM_SOURCE"] = "registered"
    from atelier.config import load_config
    return load_config()


@given('the operator has run `just optimize svm --fixture` for "{tid}"')
def step_run_fixture_optimize(context, tid):
    from atelier.optimize.svm.fixture import (
        build_and_promote_fixture_head, build_fixture_collection,
        load_fixture_category_set,
    )
    from atelier.classify.pipeline import _ensure_registered_svm_head
    from atelier.registry.nhsvm_head import get_current

    context.tid = tid
    context.cfg = _cfg_for(tid)
    context.cs = load_fixture_category_set()
    # Idempotent: only build if no current head yet (re-promote would collide
    # on the content-addressed head_sig).
    if get_current(tid, ENCODER) is None:
        build_and_promote_fixture_head(taxonomy_id=tid)
    build_fixture_collection(taxonomy_id=tid)  # ensure/upsert is idempotent
    # Install the head into ml_inference for predict_svm.
    assert _ensure_registered_svm_head(context.cfg, context.cs, taxonomy_id=tid)


@then('a current NHSVM head exists for "{tid}"')
def step_head_current(context, tid):
    from atelier.registry.nhsvm_head import get_current
    assert get_current(tid, ENCODER) is not None, f"no current head for {tid}"


@then('the maxsim collection for "{tid}" is current')
def step_collection_current(context, tid):
    from atelier.db.dao import AtelierDao
    assert AtelierDao().get_current_taxonomy_collection(tid) is not None, \
        f"no current maxsim collection for {tid}"


@when("the held-out fixture entities are classified")
def step_classify_heldout(context):
    from atelier.classify.belief import FrameOfDiscernment
    from atelier.classify.features import extract_features
    from atelier.classify.ml_inference import predict_svm
    from atelier.classify.maxsim_bridge import try_compute_maxsim_mass

    frame = FrameOfDiscernment(context.cs)
    results = []
    for row in _heldout():
        feats = extract_features(
            column_name=row["column"], column_type=row["column_type"],
            values=row["sample_values"], siblings=row["siblings_full"],
            source_table=row["table"])
        nhsvm = predict_svm(feats) or {}
        _mass, _status, attr = try_compute_maxsim_mass(
            cfg=context.cfg, column_features=feats, column_name=row["column"],
            table_name=row["table"], samples=row["sample_values"],
            neighbor_column_names=row["siblings_full"], pattern_summary=None,
            frame=frame, attribution_top_k=3)
        top3 = [t["code"] for t in (attr or {}).get("top_k", [])]
        results.append({
            "true_code": row["code"], "covered": row.get("covered", False),
            "maxsim_hit": row["code"] in top3,
            "nhsvm_mass": float(nhsvm.get(row["code"], 0.0)),
        })
    context.results = results


@then("recall@3 over the covered subset is at least {floor:f}")
def step_recall_at_3(context, floor):
    covered = [r for r in context.results if r["covered"]]
    assert covered, "no covered held-out entities"
    recall = sum(r["maxsim_hit"] for r in covered) / len(covered)
    assert recall >= floor, f"recall@3={recall:.3f} < {floor}"


@then("NHSVM contributes non-trivial mass on the maxsim-weak partition")
def step_nhsvm_nontrivial(context):
    weak = [r for r in context.results if not r["maxsim_hit"]]
    assert weak, "no maxsim-weak entities"
    nontrivial = sum(1 for r in weak if r["nhsvm_mass"] > 0.0)
    assert nontrivial >= max(1, len(weak) // 2), \
        f"NHSVM trivial on the weak tail: {nontrivial}/{len(weak)}"


@then("NHSVM mean mass on the maxsim-weak partition is at least its maxsim-strong mass")
def step_nhsvm_differential(context):
    weak = [r["nhsvm_mass"] for r in context.results if not r["maxsim_hit"]]
    strong = [r["nhsvm_mass"] for r in context.results if r["maxsim_hit"]]
    assert weak and strong, "need both partitions"
    mean_weak = sum(weak) / len(weak)
    mean_strong = sum(strong) / len(strong)
    assert mean_weak >= mean_strong, \
        f"NHSVM not carrying the tail: weak={mean_weak:.3f} < strong={mean_strong:.3f}"


@given('no promoted NHSVM head exists for "{tid}"')
def step_no_head(context, tid):
    from atelier.registry.nhsvm_head import get_current, set_status
    from atelier.classify.ml_inference import reset_svm
    cur = get_current(tid, ENCODER)
    if cur is not None:
        set_status(cur["id"], "archived")
    reset_svm()
    context.tid = tid


@when('classification is triggered with svm source "{source}"')
def step_trigger_registered(context, source):
    from atelier.classify.pipeline import _ensure_registered_svm_head
    from atelier.optimize.svm.fixture import load_fixture_category_set
    cfg = _cfg_for(context.tid)
    # source='registered' is fail-closed: the head load must report absence,
    # which the pipeline's selector converts to a loud RuntimeError.
    context.head_loaded = _ensure_registered_svm_head(
        cfg, load_fixture_category_set(), taxonomy_id=context.tid)


@then("it fails fast naming the missing head and `just optimize`")
def step_fail_fast(context):
    assert context.head_loaded is False, \
        "registered source must fail closed (no silent fallback) when no head"
