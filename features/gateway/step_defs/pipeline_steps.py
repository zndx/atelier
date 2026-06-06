"""Step definitions for pipeline integration tests via HTTP gateway."""

import json
import logging
import time
import urllib.request
import urllib.error

from behave import when, then

log = logging.getLogger(__name__)


def _gateway_url(context, path):
    from atelier.config import load_config
    if not hasattr(context, "_gateway_port"):
        cfg = load_config()
        context._gateway_port = cfg.gateway_port
    return f"http://localhost:{context._gateway_port}{path}"


# Terminal states where we stop polling
_TERMINAL_STATES = {"CONVERGED", "ERROR", "IDLE"}


@when('I poll "/api/fsm/status" until state is terminal')
def step_poll_fsm(context):
    """Poll FSM status with proof-of-progress timeout.

    As long as the state is changing (progress), keep going.
    If stuck in the same state for 60s, give up.
    """
    url = _gateway_url(context, "/api/fsm/status")
    deadline = time.monotonic() + 300  # hard cap: 5 minutes
    last_state = None
    last_change = time.monotonic()
    stale_timeout = 60  # no progress for 60s → fail

    while time.monotonic() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=10)
            body = json.loads(resp.read().decode())
            state = body.get("state", "IDLE")

            if state != last_state:
                log.info("FSM state: %s → %s", last_state, state)
                last_state = state
                last_change = time.monotonic()

            if state in _TERMINAL_STATES and state != "IDLE":
                context.fsm_final_state = state
                context.fsm_final_body = body
                return

            # IDLE after having been non-IDLE means pipeline finished
            if state == "IDLE" and last_state is not None:
                context.fsm_final_state = state
                context.fsm_final_body = body
                return

        except Exception as e:
            log.warning("FSM poll error: %s", e)

        if time.monotonic() - last_change > stale_timeout:
            raise AssertionError(
                f"FSM stuck in state '{last_state}' for {stale_timeout}s"
            )

        time.sleep(3)

    raise AssertionError(
        f"FSM did not reach terminal state within 300s. Last: {last_state}"
    )


@then('the FSM state should be "{expected}"')
def step_fsm_state_is(context, expected):
    actual = getattr(context, "fsm_final_state", None)
    body = getattr(context, "fsm_final_body", {})
    assert actual == expected, (
        f"FSM state is '{actual}', expected '{expected}'. "
        f"Progress: {body.get('progress', {})}"
    )


@then("the latest dataset should have row_count greater than {n:d}")
def step_latest_dataset_row_count(context, n):
    datasets = context.response_json.get("datasets", [])
    assert datasets, "No datasets found"
    # Sort by created_at descending if available, otherwise take last
    latest = datasets[-1]
    row_count = latest.get("row_count", 0)
    assert row_count > n, (
        f"Latest dataset row_count is {row_count}, expected > {n}"
    )


@then('the latest run should have state "{expected}"')
def step_latest_run_state(context, expected):
    runs = context.response_json.get("runs", [])
    assert runs, "No FSM runs found"
    latest = runs[-1]
    state = latest.get("state", "")
    assert state == expected, (
        f"Latest run state is '{state}', expected '{expected}'"
    )
