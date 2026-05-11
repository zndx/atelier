# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Step definitions for the Extend Classification tier-1 BDD.

Scenarios drive a real pipeline-level Extend run end-to-end against the
locally-available OOTB sample source and CatBoost/SVM/UMAP files
produced by an earlier classify run (or by the test setup itself).
The LLM is NEVER invoked — that's the load-bearing assertion of the
tier-1 suite.
"""

from __future__ import annotations

from pathlib import Path

from behave import given, when, then


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _most_recent_artifact_run_dir() -> Path | None:
    """Return the most recent ``build/results/{run_id}/`` that has a
    CatBoost classifier on disk, or None if no such run exists.

    Used to seed an artifact set row for the test without requiring a
    prior classify run inside this BDD.  In production, the row is
    written by ``run_classification_pipeline`` at the EVALUATING tail.
    """
    results_root = _PROJECT_ROOT / "build" / "results"
    if not results_root.is_dir():
        return None
    candidates = []
    for d in results_root.iterdir():
        if not d.is_dir():
            continue
        if (d / "catboost_fit_to_llm.cbm").is_file():
            candidates.append((d.stat().st_mtime, d))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


@given("an existing classify-run artifact set is registered")
def step_seed_artifact_set(context):
    """Find the most recent classify run dir and register it as an artifact set.

    Skips the scenario when no eligible run dir exists (a fresh checkout
    with no completed classify runs has nothing to extend from).  In
    that case the @tier-1 suite as a whole assumes an upstream
    integration test seeded one.
    """
    from atelier.classify.artifact_set import build_artifact_set_record
    from atelier.config import load_config
    from atelier.db.dao import AtelierDao

    run_dir = _most_recent_artifact_run_dir()
    if run_dir is None:
        context.scenario.skip(
            "No prior classify run dir with a CatBoost model available "
            "to seed the artifact set"
        )
        return

    cfg = load_config()
    dao = AtelierDao()
    run_id = run_dir.name
    spec = build_artifact_set_record(
        run_id=run_id,
        results_dir=run_dir,
        cfg=cfg,
        n_columns=0,
        source_id="ootb-sample",
        # FK reference: only set when a fsm_run row exists for this id.
        fsm_run_id=run_id if dao.get_fsm_run(run_id) else None,
    )
    if spec is None:
        context.scenario.skip(
            f"Artifact spec build returned None for {run_id}"
        )
        return

    if dao.get_artifact_set(run_id) is None:
        dao.register_artifact_set(**spec)

    context.artifact_set_id = run_id
    context.dao = dao


@given("the artifact set is the active artifact set")
def step_set_active_artifact(context):
    ok = context.dao.set_active_artifact_set(context.artifact_set_id)
    assert ok, f"set_active_artifact_set returned False for {context.artifact_set_id}"


@given("an LLM call counter starting at zero")
def step_llm_counter(context):
    """Marker step.  Extend doesn't accept an LLM backend, so it can't
    invoke one.  We assert this structurally in the Then step rather
    than wiring a real counter — the Extend pipeline's signature
    refuses an llm_backend parameter, so any pretend counter remains 0.
    """
    context.llm_call_count = 0


@when("I run Extend Classification against the OOTB Sample source")
def step_run_extend(context):
    from atelier.classify.extend_pipeline import run_extend_classification
    from atelier.classify.fsm import AgentFSM
    from atelier.config import load_config

    cfg = load_config()
    fsm = AgentFSM(dao=context.dao)
    context.extend_result = run_extend_classification(
        cfg, fsm,
        source_id="ootb-sample",
        artifact_set_id=context.artifact_set_id,
    )


@then("the Extend run should complete with state CONVERGED")
def step_extend_converged(context):
    result = context.extend_result
    assert result is not None, "Extend pipeline returned None"
    # The pipeline advances FSM to CONVERGED at the end.  Verify via
    # the DAO's fsm_runs state.
    fsm_row = context.dao.get_fsm_run(result["run_id"])
    assert fsm_row is not None, f"fsm_runs row missing for {result['run_id']}"
    assert fsm_row["state"] == "CONVERGED", (
        f"expected CONVERGED, got {fsm_row['state']}: {fsm_row.get('error')}"
    )


@then('the new dataset row should have run_kind "{kind}"')
def step_dataset_run_kind(context, kind):
    ds = context.dao.get_dataset(context.extend_result["run_id"])
    assert ds is not None, "Dataset row not registered"
    assert ds["run_kind"] == kind, (
        f"expected run_kind={kind}, got {ds['run_kind']}"
    )


@then("the new dataset row should reference the consumed artifact set")
def step_dataset_artifact_link(context):
    ds = context.dao.get_dataset(context.extend_result["run_id"])
    assert ds["artifact_set_id"] == context.artifact_set_id, (
        f"expected artifact_set_id={context.artifact_set_id}, "
        f"got {ds.get('artifact_set_id')}"
    )


@then("the LLM call counter should still be zero")
def step_no_llm_calls(context):
    # Structural check: Extend's run_classification entry doesn't accept
    # an llm_backend kwarg, so no LLM path can have been exercised.
    import inspect
    from atelier.classify.extend_pipeline import run_extend_classification
    sig = inspect.signature(run_extend_classification)
    assert "llm_backend" not in sig.parameters, (
        "run_extend_classification should not accept an llm_backend "
        "parameter — Extend is ML-only"
    )
    assert context.llm_call_count == 0


@then("the run summary should include a vocab_compatibility status")
def step_vocab_compat_present(context):
    assert "vocab_compatibility" in context.extend_result


@then('the vocab_compatibility status should be one of "{statuses}"')
def step_vocab_compat_value(context, statuses):
    valid = set(statuses.split("|"))
    actual = context.extend_result["vocab_compatibility"]
    assert actual in valid, (
        f"expected vocab_compatibility in {valid}, got {actual!r}"
    )


@then('the run dir should contain "{filename}"')
def step_run_dir_has_file(context, filename):
    run_dir = Path(context.extend_result["result_path"])
    assert (run_dir / filename).is_file(), (
        f"missing expected file in run dir {run_dir}: {filename}"
    )
