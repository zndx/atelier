"""Step definitions for the artifact-sets gateway BDD.

Drives the FastAPI gateway in-process via TestClient against a real DB
so URL routing, body validation, and the DAO integration are all
exercised together.  The gateway requires an artifact-set row to exist
for most scenarios — Background seeds one from the most recent classify
run dir on disk, identical to the extend_pipeline_steps Background.
"""

from __future__ import annotations

import json
from pathlib import Path

from behave import given, when, then


_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _testclient(context):
    if not hasattr(context, "_tc"):
        from fastapi.testclient import TestClient
        from atelier.gateway import app
        context._tc = TestClient(app)
    return context._tc


def _expand_placeholders(s: str, context) -> str:
    """Substitute ``{seeded_artifact_set_id}`` in feature paths."""
    if "{seeded_artifact_set_id}" in s:
        return s.replace(
            "{seeded_artifact_set_id}", context.seeded_artifact_set_id
        )
    return s


def _most_recent_classify_run() -> Path | None:
    """Find the most recent build/results/{run_id}/ that has a CatBoost model."""
    results_root = _PROJECT_ROOT / "build" / "results"
    if not results_root.is_dir():
        return None
    candidates = []
    for d in results_root.iterdir():
        if d.is_dir() and (d / "catboost_fit_to_llm.cbm").is_file():
            candidates.append((d.stat().st_mtime, d))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


@given("the FastAPI gateway is reachable")
def step_gateway_reachable(context):
    client = _testclient(context)
    r = client.get("/api/artifact-sets")
    assert r.status_code == 200, (
        f"Gateway is not serving /api/artifact-sets (got {r.status_code})"
    )


@given("an artifact set is registered for the OOTB Sample source")
def step_seed_artifact_set(context):
    """Reuse the existing helpers to seed one row from disk.

    Mirrors the extend_pipeline_steps Background: locate the most
    recent classify run dir, build the artifact spec, register if not
    already present.  The gateway's own register endpoint isn't
    tested here (it doesn't exist by design — registration is a
    pipeline-side concern, not an operator-facing one).
    """
    from atelier.classify.artifact_set import build_artifact_set_record
    from atelier.config import load_config
    from atelier.db.dao import AtelierDao

    run_dir = _most_recent_classify_run()
    if run_dir is None:
        context.scenario.skip("No classify run dir available to seed from")
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
        fsm_run_id=run_id if dao.get_fsm_run(run_id) else None,
    )
    if spec is None:
        context.scenario.skip(f"Artifact spec build returned None for {run_id}")
        return

    if dao.get_artifact_set(run_id) is None:
        dao.register_artifact_set(**spec)

    context.seeded_artifact_set_id = run_id
    context.dao = dao


def _record_response(context, r) -> None:
    """Populate the response_* attributes the shared http_steps assert on.

    Matches the format produced by features/gateway/step_defs/http_steps.py
    so we can reuse ``the response status should be N`` and other shared
    assertion steps in the artifact-sets feature without rewriting them.
    """
    context.response_status = r.status_code
    body = r.text
    context.response_body = body
    context.response_headers = dict(r.headers)
    try:
        context.response_json = r.json()
    except Exception:
        context.response_json = None


@when('I call GET "{path}"')
def step_call_get(context, path):
    expanded = _expand_placeholders(path, context)
    _record_response(context, _testclient(context).get(expanded))


@when('I call POST "{path}"')
def step_call_post_no_body(context, path):
    expanded = _expand_placeholders(path, context)
    _record_response(context, _testclient(context).post(expanded))


@when('I call POST "{path}" with body {body}')
def step_call_post_with_body(context, path, body):
    expanded = _expand_placeholders(path, context)
    payload = json.loads(body)
    _record_response(context, _testclient(context).post(expanded, json=payload))


@then("the response status should be in {a:d} or {b:d}")
def step_status_in(context, a, b):
    assert context.response_status in (a, b), (
        f"expected status {a} or {b}, got {context.response_status}: "
        f"{context.response_body[:200]}"
    )


@then('the response body should have field "{key}"')
def step_field_present(context, key):
    assert isinstance(context.response_json, dict), "Response is not a JSON object"
    assert key in context.response_json, (
        f"missing field {key!r}; got keys {list(context.response_json.keys())}"
    )


@then('the response body should have field "{key}" with value {value}')
def step_field_value(context, key, value):
    assert isinstance(context.response_json, dict), "Response is not a JSON object"
    parsed_value = json.loads(value)  # bool/number/string from JSON
    actual = context.response_json.get(key)
    assert actual == parsed_value, (
        f"field {key}: expected {parsed_value!r}, got {actual!r}"
    )


@then("the artifact_sets list should contain at least {n:d} row")
@then("the artifact_sets list should contain at least {n:d} rows")
def step_artifact_sets_count(context, n):
    assert isinstance(context.response_json, dict)
    rows = context.response_json.get("artifact_sets", [])
    assert len(rows) >= n, (
        f"expected >= {n} rows, got {len(rows)}"
    )
