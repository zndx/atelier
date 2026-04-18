"""Step defs for features/gateway/terminal_models.feature.

Exercises the ``/api/terminal/models`` endpoints in-process via the
FastAPI TestClient.  No real SDK calls — the catalog + selection +
availability logic is deterministic against a freshly loaded
AtelierConfig.
"""

from __future__ import annotations

import os

from behave import given, when, then


def _tc(context):
    if not hasattr(context, "_tc"):
        from fastapi.testclient import TestClient
        from atelier.gateway import app
        context._tc = TestClient(app)
    return context._tc


# ── Background fixtures ────────────────────────────────────────


@given("ANTHROPIC_API_KEY is set so Anthropic rows are available")
def step_anthropic_set(context):
    context.prior_anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-bdd-fixture"
    context.add_cleanup(_restore_anthropic, context)


def _restore_anthropic(context):
    prior = getattr(context, "prior_anthropic_key", None)
    if prior is None:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    else:
        os.environ["ANTHROPIC_API_KEY"] = prior


@given("the terminal model override is cleared")
def step_override_cleared(context):
    from atelier.terminal_selection import clear_active
    clear_active()


@given("Bedrock credentials are absent")
def step_no_bedrock(context):
    # Sweep the AWS env vars the credential probe reads, remembering
    # prior values for post-scenario restore.
    keys = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN", "AWS_REGION")
    context.prior_aws = {k: os.environ.pop(k, None) for k in keys}
    context.add_cleanup(_restore_aws, context)


def _restore_aws(context):
    for k, v in getattr(context, "prior_aws", {}).items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


# ── GET catalog ────────────────────────────────────────────────


@when("I GET the terminal models catalog")
def step_get_catalog(context):
    r = _tc(context).get("/api/terminal/models")
    context.tc_status = r.status_code
    context.tc_json = r.json()


@then("the catalog has {count:d} entries")
def step_catalog_count(context, count):
    assert context.tc_status == 200, context.tc_status
    assert len(context.tc_json["models"]) == count, (
        f"expected {count} models, got {len(context.tc_json['models'])}"
    )


@then('the catalog includes a "{provider}" provider row')
@then('the catalog includes an "{provider}" provider row')
def step_catalog_has_provider(context, provider):
    rows = context.tc_json["models"]
    assert any(m["provider"] == provider for m in rows), (
        f"no {provider!r} row in {[m['provider'] for m in rows]}"
    )


@then('at least one "{provider}" row is available')
def step_provider_available(context, provider):
    rows = [m for m in context.tc_json["models"]
            if m["provider"] == provider and m["available"]]
    assert rows, (
        f"no available {provider!r} rows: "
        f"{[(m['id'], m['unavailable_reason']) for m in context.tc_json['models'] if m['provider'] == provider]}"
    )


@then("the terminal catalog reports override_set {flag:w}")
def step_override_flag(context, flag):
    r = _tc(context).get("/api/terminal/models")
    expected = flag.lower() == "true"
    assert r.json()["override_set"] is expected, (
        f"override_set = {r.json()['override_set']}, expected {expected}"
    )


@then("every row has a stats object with n >= 0")
def step_stats_present(context):
    for m in context.tc_json["models"]:
        assert "stats" in m and isinstance(m["stats"], dict), m
        assert m["stats"]["n"] >= 0, m["stats"]


# ── POST active ────────────────────────────────────────────────


@when('I POST terminal model id "{entry_id}"')
def step_post_active(context, entry_id):
    r = _tc(context).post(
        "/api/terminal/models/active", json={"id": entry_id},
    )
    context.tc_status = r.status_code
    context.tc_json = r.json()


@then("the selection response is ok")
def step_selection_ok(context):
    assert context.tc_status == 200, (
        f"status={context.tc_status}, body={context.tc_json}"
    )
    assert "error" not in context.tc_json, context.tc_json


@then('the active terminal model id is "{entry_id}"')
def step_active_id(context, entry_id):
    got = (context.tc_json.get("active") or {}).get("id")
    assert got == entry_id, f"active id = {got!r}, expected {entry_id!r}"


@then("the selection response reports not available")
def step_selection_unavail(context):
    body = context.tc_json
    err = str(body.get("error", "")).lower()
    assert "not available" in err or "not configured" in err, (
        f"expected not-available error, got: {body}"
    )


@then("the selection response reports unknown id")
def step_selection_unknown(context):
    body = context.tc_json
    err = str(body.get("error", "")).lower()
    assert "unknown" in err, f"expected unknown-id error, got: {body}"


# ── DELETE override ────────────────────────────────────────────


@when("I DELETE the terminal model override")
def step_delete_override(context):
    r = _tc(context).delete("/api/terminal/models/active")
    context.tc_status = r.status_code
    context.tc_json = r.json()
