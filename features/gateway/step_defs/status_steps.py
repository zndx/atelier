"""Step definitions for the status endpoint."""

import json

from behave import when, then


@when("I GET /api/status from a test client")
def step_get_status_test_client(context):
    from fastapi.testclient import TestClient
    from atelier.gateway import app

    client = TestClient(app)
    resp = client.get("/api/status")
    context.status_code = resp.status_code
    context.status_body = resp.json()


@when("I GET /api/status from the running gateway")
def step_get_status_live(context):
    import urllib.request

    from atelier.config import load_config

    cfg = load_config()
    url = f"http://localhost:{cfg.gateway_port}/api/status"
    resp = urllib.request.urlopen(url, timeout=10)
    context.status_code = resp.status
    context.status_body = json.loads(resp.read().decode())


@then('the status response contains "{key}"')
def step_status_contains(context, key):
    assert key in context.status_body, \
        f"Expected '{key}' in status response, got keys: {list(context.status_body.keys())}"


@then('the status response does not contain "{key}"')
def step_status_not_contains(context, key):
    flat = json.dumps(context.status_body)
    assert f'"{key}"' not in flat, \
        f"Found '{key}' in status response — potential credential leak"


@then('the config section contains "{key}"')
def step_config_contains(context, key):
    config = context.status_body.get("config", {})
    assert key in config, \
        f"Expected '{key}' in config section, got keys: {list(config.keys())}"


@then('the status response "{path}" is true')
def step_status_path_true(context, path):
    value = context.status_body
    for part in path.split("."):
        value = value[part]
    assert value is True, f"Expected {path}=True, got {value}"
