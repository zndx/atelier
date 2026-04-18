"""Step defs for features/gateway/auto_start.feature.

Monkeypatches ``atelier.gateway.fsm_start`` with a spy so the scenarios
can assert on invocation without actually running the pipeline.
Env vars are patched per-scenario and restored via context.add_cleanup.
"""

from __future__ import annotations

import os

from behave import given, when, then


def _patch_env(context, key: str, value: str) -> None:
    if not hasattr(context, "_auto_patched_env"):
        context._auto_patched_env = {}
    if key not in context._auto_patched_env:
        context._auto_patched_env[key] = os.environ.get(key)
    if value == "":
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    if not any(cb.__name__ == "_restore_auto_env"
               for cb in getattr(context, "_cleanups", [])):
        context.add_cleanup(_restore_auto_env, context)


def _restore_auto_env(context) -> None:
    for key, prior in getattr(context, "_auto_patched_env", {}).items():
        if prior is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prior


@given('ATELIER_CLASSIFY_AUTO_START is "{value}"')
def step_autostart_flag(context, value):
    _patch_env(context, "ATELIER_CLASSIFY_AUTO_START", value)


@given('ATELIER_CLASSIFY_CONNECTION is "{value}"')
def step_autostart_conn(context, value):
    _patch_env(context, "ATELIER_CLASSIFY_CONNECTION", value)


@given('ATELIER_CLASSIFY_CONNECTION is ""')
def step_autostart_conn_empty(context):
    # behave's {value} placeholder requires at least one char, so the
    # empty-override case (operator forgot to set CONNECTION) needs an
    # explicit step.
    _patch_env(context, "ATELIER_CLASSIFY_CONNECTION", "")


@given('ATELIER_CLASSIFY_DATABASE is "{value}"')
def step_autostart_db(context, value):
    _patch_env(context, "ATELIER_CLASSIFY_DATABASE", value)


@given('ATELIER_CLASSIFY_DATABASE is ""')
def step_autostart_db_empty(context):
    _patch_env(context, "ATELIER_CLASSIFY_DATABASE", "")


@when("_maybe_auto_start_classify runs")
def step_run_maybe_autostart(context):
    from atelier import gateway
    calls: list[dict] = []

    def _spy(source_id=None, **kw):
        calls.append({"source_id": source_id, **kw})
        return {"run_id": "spy", "started": True}

    prior = gateway.fsm_start
    gateway.fsm_start = _spy  # type: ignore[assignment]
    context.add_cleanup(_restore_fsm_start, gateway, prior)
    context.autostart_calls = calls
    gateway._maybe_auto_start_classify()


def _restore_fsm_start(gateway_mod, prior) -> None:
    gateway_mod.fsm_start = prior  # type: ignore[assignment]


@then("fsm_start is not called")
def step_fsm_not_called(context):
    assert context.autostart_calls == [], context.autostart_calls


@then('fsm_start is called with source_id "{source_id}"')
def step_fsm_called_with(context, source_id):
    assert len(context.autostart_calls) == 1, context.autostart_calls
    assert context.autostart_calls[0]["source_id"] == source_id, (
        context.autostart_calls
    )
