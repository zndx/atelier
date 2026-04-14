"""Reusable TestClient step definitions for tier-0 gateway tests.

Uses FastAPI's TestClient to run the gateway in-process. gRPC calls
fail gracefully (gateway returns JSON error envelopes). These steps
validate routing, error handling, and response envelope shapes.
"""

from behave import when, then


def _get_test_client(context):
    """Lazy-init a shared TestClient for the test session."""
    if not hasattr(context, "_tc"):
        from fastapi.testclient import TestClient
        from atelier.gateway import app
        context._tc = TestClient(app)
    return context._tc


@when("I GET {path} from a test client")
def step_tc_get(context, path):
    client = _get_test_client(context)
    resp = client.get(path)
    context.tc_status = resp.status_code
    try:
        context.tc_json = resp.json()
    except Exception:
        context.tc_json = None
    context.tc_body = resp.text


@then("the test client response should be valid JSON")
def step_tc_valid_json(context):
    assert context.tc_json is not None, (
        f"Response is not valid JSON (status={context.tc_status}): "
        f"{context.tc_body[:200]}"
    )


@then('the test client JSON should contain "{key}"')
def step_tc_json_contains(context, key):
    assert context.tc_json is not None, "Response is not JSON"
    assert key in context.tc_json, (
        f"Key '{key}' not in response, got keys: {list(context.tc_json.keys())}"
    )


@then('the test client JSON "{key}" should be a list')
def step_tc_json_key_is_list(context, key):
    assert context.tc_json is not None, "Response is not JSON"
    val = context.tc_json.get(key)
    assert isinstance(val, list), (
        f"'{key}' is not a list: {type(val)}"
    )
