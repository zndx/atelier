"""Reusable HTTP step definitions for tier-1 gateway tests.

All tier-1 gateway scenarios share these steps for HTTP GET/POST against
the live running gateway. Uses urllib.request (stdlib, no extra deps).
"""

import json
import urllib.request
import urllib.error

from behave import when, then


def _gateway_url(context, path):
    """Build gateway URL from config."""
    from atelier.config import load_config
    if not hasattr(context, "_gateway_port"):
        cfg = load_config()
        context._gateway_port = cfg.gateway_port
    return f"http://localhost:{context._gateway_port}{path}"


# ── HTTP verbs ──────────────────────────────────────────────────────


@when('I GET "{path}" from the gateway')
def step_http_get(context, path):
    url = _gateway_url(context, path)
    try:
        resp = urllib.request.urlopen(url, timeout=15)
        context.response_status = resp.status
        body = resp.read().decode()
        context.response_body = body
        context.response_headers = dict(resp.headers)
        try:
            context.response_json = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            context.response_json = None
    except urllib.error.HTTPError as e:
        context.response_status = e.code
        body = e.read().decode()
        context.response_body = body
        context.response_headers = dict(e.headers)
        try:
            context.response_json = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            context.response_json = None


@when('I POST "{path}" to the gateway')
def step_http_post(context, path):
    url = _gateway_url(context, path)
    data = None
    headers = {}
    if context.text:
        data = context.text.encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        context.response_status = resp.status
        body = resp.read().decode()
        context.response_body = body
        context.response_headers = dict(resp.headers)
        try:
            context.response_json = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            context.response_json = None
    except urllib.error.HTTPError as e:
        context.response_status = e.code
        body = e.read().decode()
        context.response_body = body
        context.response_headers = dict(e.headers)
        try:
            context.response_json = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            context.response_json = None


# ── Response assertions ─────────────────────────────────────────────


@then("the response status should be {code:d}")
def step_check_status(context, code):
    assert context.response_status == code, (
        f"Expected status {code}, got {context.response_status}"
    )


@then('the response JSON should contain "{key}"')
def step_json_contains_key(context, key):
    assert context.response_json is not None, "Response is not JSON"
    assert key in context.response_json, (
        f"Key '{key}' not in response JSON, got keys: "
        f"{list(context.response_json.keys())}"
    )


@then('the response JSON "{key}" should be a non-empty list')
def step_json_key_nonempty_list(context, key):
    assert context.response_json is not None, "Response is not JSON"
    val = context.response_json.get(key)
    assert isinstance(val, list), f"'{key}' is not a list: {type(val)}"
    assert len(val) > 0, f"'{key}' is an empty list"


@then("the response JSON should be a non-empty list")
def step_json_is_nonempty_list(context):
    assert context.response_json is not None, "Response is not JSON"
    assert isinstance(context.response_json, list), (
        f"Response is not a list: {type(context.response_json)}"
    )
    assert len(context.response_json) > 0, "Response is an empty list"


@then('the response JSON "{key}" should have at least {n:d} items')
def step_json_key_min_items(context, key, n):
    assert context.response_json is not None, "Response is not JSON"
    val = context.response_json.get(key)
    assert isinstance(val, list), f"'{key}' is not a list: {type(val)}"
    assert len(val) >= n, (
        f"'{key}' has {len(val)} items, expected at least {n}"
    )


@then('the response JSON "{key}" should be at least {n:d}')
def step_json_key_min_value(context, key, n):
    assert context.response_json is not None, "Response is not JSON"
    val = context.response_json.get(key)
    assert val is not None, f"Key '{key}' not in response JSON"
    assert val >= n, f"'{key}' is {val}, expected at least {n}"


@then('the response JSON "{key}" should be "{value}"')
def step_json_key_equals(context, key, value):
    assert context.response_json is not None, "Response is not JSON"
    actual = context.response_json.get(key)
    assert str(actual) == value, (
        f"'{key}' is '{actual}', expected '{value}'"
    )


@then('the response JSON "{key}" should be true')
def step_json_key_true(context, key):
    assert context.response_json is not None, "Response is not JSON"
    val = context.response_json.get(key)
    assert val is True, f"'{key}' is {val}, expected True"


@then('the response should contain "{text}"')
def step_response_contains_text(context, text):
    assert text in context.response_body or text in str(context.response_headers), (
        f"Response does not contain '{text}'"
    )
