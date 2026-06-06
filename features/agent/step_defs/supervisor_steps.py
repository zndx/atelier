"""Step defs for features/agent/supervisor.feature.

Exercises Pillar 3 invariants (Bedrock-only overlay, validation) plus
Pillar 1 halving-retry behavior and Pillar 2 nautilus triggers entirely
in-process so the suite stays tier-0 (no SDK, no gateway).
"""

from __future__ import annotations

import os

from behave import given, then, when

from atelier.classify.bootstrap import (
    BootstrapState, FatalLLMError, _classify_batch_with_retry,
)
from atelier.classify.fsm import AgentFSM, FSMState
from atelier.classify.llm_backend import ColumnClassification, LLMResponse
from atelier.classify.sampler import ColumnSample
from atelier.overwatch.hooks import evaluate_hook
from atelier.overwatch.nautilus import (
    NautilusConfig, NautilusWatcher,
    register_state as nautilus_register_state,
    unregister_state as nautilus_unregister_state,
)
from atelier.overwatch.remediation import (
    OverlayProposal, RemediationError, validate_proposal,
)


# ── Helpers ─────────────────────────────────────────────────────


def _ok_response(samples):
    return LLMResponse(
        classifications=[
            ColumnClassification(s.name, "ICE.DESIG.NAME", 0.9, "ev")
            for s in samples
        ],
        input_tokens=100, output_tokens=100, model="test", finish_reason="stop",
    )


class _FailAboveBackend:
    def __init__(self, threshold):
        self.threshold = threshold
        self.calls = []

    def classify_batch(self, samples, _sys, revisit_context=None, table_name=None):
        n = len(samples)
        self.calls.append(n)
        if n > self.threshold:
            raise TimeoutError(f"batch {n} failed")
        return _ok_response(samples)


class _AlwaysFailBackend:
    def classify_batch(self, samples, *a, **kw):
        raise TimeoutError("always fail")


class _AuthBackend:
    def classify_batch(self, *a, **kw):
        # Name-matching works without importing a real provider SDK.
        class AuthenticationError(Exception):
            pass
        raise AuthenticationError("invalid api key")


def _columns(n):
    return [
        ColumnSample(
            name=f"col_{i}", column_type="string",
            values=["a", "b"], total_count=2, null_count=0, table_name="t",
        )
        for i in range(n)
    ]


# ── Bedrock invariant ──────────────────────────────────────────


@given('a supervisor overlay proposal touching "{key}"')
def step_invariant_key(context, key):
    context.proposal = OverlayProposal(
        run_id="test", overlay={key: "anthropic"},
        rationale="test", expected_effect="test", trigger="post_mortem",
    )


@given('a supervisor overlay proposal setting "{key}" to {value}')
def step_overlay_set(context, key, value):
    # Best-effort coerce: ints / floats / bare strings.
    try:
        coerced = int(value)
    except ValueError:
        try:
            coerced = float(value)
        except ValueError:
            coerced = value
    context.proposal = OverlayProposal(
        run_id="test", overlay={key: coerced},
        rationale="test", expected_effect="test", trigger="post_mortem",
    )


@when("I validate the proposal")
def step_validate(context):
    context.validation_error = None
    context.accepted_keys = None
    try:
        context.accepted_keys = validate_proposal(context.proposal)
    except RemediationError as exc:
        context.validation_error = exc


@then('the proposal is rejected with "{fragment}"')
def step_rejected_with(context, fragment):
    assert context.validation_error is not None, (
        "proposal was accepted; expected rejection"
    )
    assert fragment in str(context.validation_error), (
        f"rejection missing {fragment!r}: {context.validation_error}"
    )


@then("the proposal is accepted")
def step_accepted(context):
    assert context.validation_error is None, (
        f"unexpected rejection: {context.validation_error}"
    )


@then('the accepted keys include "{key}"')
def step_accepted_key(context, key):
    assert context.accepted_keys is not None
    assert key in context.accepted_keys, (
        f"{key!r} not in accepted: {context.accepted_keys}"
    )


# ── Halving retry (Pillar 1) ───────────────────────────────────


@given("a mock LLM backend that fails on batches larger than {threshold:d}")
def step_mock_fail_above(context, threshold):
    context.backend = _FailAboveBackend(threshold)


@given("a mock LLM backend that always fails")
def step_mock_all_fail(context):
    context.backend = _AlwaysFailBackend()


@given("a mock LLM backend that raises an authentication error")
def step_mock_auth_fail(context):
    context.backend = _AuthBackend()


@when("I run the LLM sweep on {n:d} columns")
def step_run_sweep(context, n):
    context.state = BootstrapState()
    context.fatal_error = None
    try:
        context.results = _classify_batch_with_retry(
            context.backend, _columns(n), "sys", context.state, min_batch=1,
        )
    except FatalLLMError as exc:
        context.fatal_error = exc
        context.results = []


@when("I run the LLM sweep on {n:d} columns with min_batch {min_batch:d}")
def step_run_sweep_min(context, n, min_batch):
    context.state = BootstrapState()
    context.fatal_error = None
    try:
        context.results = _classify_batch_with_retry(
            context.backend, _columns(n), "sys", context.state,
            min_batch=min_batch,
        )
    except FatalLLMError as exc:
        context.fatal_error = exc
        context.results = []


@then("every column receives a classification")
def step_all_labeled(context):
    assert len(context.results) == len(context.backend.calls[0:1] * 0 + [1] * 25) - (25 - 25), ""  # placeholder
    # simpler: we sent 25, expect 25 results
    assert len(context.results) == 25, (
        f"expected 25 classifications, got {len(context.results)}"
    )


@then("the batch audit records a top-level halved-on-error attempt")
def step_audit_halved(context):
    tops = [a for a in context.state.batch_audit if a.depth == 0]
    assert len(tops) == 1 and tops[0].status == "halved_on_error", (
        f"top-level audit: {[(a.depth, a.status) for a in tops]}"
    )


@then("the batch audit records two success leaves")
def step_audit_leaves(context):
    leaves = [a for a in context.state.batch_audit if a.depth == 1]
    assert len(leaves) == 2 and all(a.status == "success" for a in leaves), (
        f"leaves: {[(a.depth, a.status) for a in leaves]}"
    )


@then("the batch audit records {n:d} failed per-column attempts")
def step_audit_failed_leaves(context, n):
    failed = [a for a in context.state.batch_audit if a.status == "failed"]
    assert len(failed) == n, f"got {len(failed)} failed leaves, expected {n}"


@then("the failed_columns list has {n:d} entries")
def step_failed_list(context, n):
    assert len(context.state.failed_columns) == n, (
        f"failed_columns={context.state.failed_columns}"
    )


@then("a FatalLLMError is raised")
def step_fatal_raised(context):
    assert context.fatal_error is not None, "expected FatalLLMError"


@then("the batch audit records a single fatal entry")
def step_audit_fatal(context):
    fatal = [a for a in context.state.batch_audit if a.status == "fatal"]
    assert len(fatal) == 1, (
        f"expected 1 fatal audit entry, got {len(fatal)}: "
        f"{[(a.depth, a.status) for a in context.state.batch_audit]}"
    )


# ── Nautilus (Pillar 2) ───────────────────────────────────────


def _make_fsm_in_llm_sweep():
    fsm = AgentFSM(dao=None)
    run = fsm.start_run()
    for s in (
        FSMState.LOADING_VOCAB, FSMState.DISCOVERING,
        FSMState.SAMPLING, FSMState.LLM_SWEEP,
    ):
        fsm.advance(run.id, s)
    return fsm, run.id


@given(
    "a nautilus watcher with llm_sweep_threshold {sw:d} and stall_threshold {st:d}"
)
def step_watcher_with_thresholds(context, sw, st):
    fsm, rid = _make_fsm_in_llm_sweep()
    context.nautilus_run_id = rid
    cfg = NautilusConfig(
        enabled=True, stall_threshold_s=float(st),
        llm_sweep_threshold_s=float(sw),
    )
    context.watcher = NautilusWatcher(
        rid, fsm, cfg,
        intervene_callback=lambda rec: {"decision": "observed"},
        clock=lambda: 0.0,
    )
    context.nautilus_fsm = fsm


@given("a nautilus watcher with stall_threshold {st:d}")
def step_watcher_stall_only(context, st):
    fsm, rid = _make_fsm_in_llm_sweep()
    context.nautilus_run_id = rid
    cfg = NautilusConfig(
        enabled=True, stall_threshold_s=float(st),
        llm_sweep_threshold_s=1e9,  # effectively disabled for this scenario
    )
    context.watcher = NautilusWatcher(
        rid, fsm, cfg,
        intervene_callback=lambda rec: {"decision": "observed"},
        clock=lambda: 0.0,
    )
    context.nautilus_fsm = fsm


@given("a nautilus watcher with failed_batch_threshold {t:d}")
def step_watcher_failed_thresh(context, t):
    fsm, rid = _make_fsm_in_llm_sweep()
    context.nautilus_run_id = rid
    cfg = NautilusConfig(
        enabled=True, stall_threshold_s=1e9, llm_sweep_threshold_s=1e9,
        failed_batch_threshold=t,
    )
    context.watcher = NautilusWatcher(
        rid, fsm, cfg,
        intervene_callback=lambda rec: {"decision": "observed"},
        clock=lambda: 0.0,
    )
    context.nautilus_fsm = fsm


@given("a nautilus watcher with can_cancel {flag:w} and stall threshold {st:d}")
def step_watcher_cancel(context, flag, st):
    fsm, rid = _make_fsm_in_llm_sweep()
    context.nautilus_run_id = rid
    cfg = NautilusConfig(
        enabled=True, stall_threshold_s=float(st), llm_sweep_threshold_s=1e9,
        can_cancel=(flag.lower() == "true"),
    )
    context.watcher = NautilusWatcher(
        rid, fsm, cfg,
        intervene_callback=lambda rec: {"decision": "cancelled", "reason": "test"},
        clock=lambda: 0.0,
    )
    context.nautilus_fsm = fsm


@given("a registered BootstrapState")
def step_register_state(context):
    context.nautilus_state = BootstrapState()
    nautilus_register_state(context.nautilus_run_id, context.nautilus_state)


@when(
    "the FSM is in LLM_SWEEP for {sec:d} seconds with no audit activity"
)
def step_fsm_sweep_seconds(context, sec):
    nautilus_register_state(context.nautilus_run_id, BootstrapState())
    context.watcher.tick()  # prime heartbeat
    context.watcher._clock = lambda: float(sec)
    context.watcher.tick()


@then("ticking again in the same phase fires no additional triggers")
def step_retick_same_phase(context):
    before = len(context.watcher.interventions)
    context.watcher._clock = lambda: 600.0
    context.watcher.tick()
    after = len(context.watcher.interventions)
    assert after == before, (
        f"triggers refired in same phase: {before} -> {after}"
    )


@then('the watcher fires triggers "{expected}"')
def step_watcher_fires(context, expected):
    wanted = {t.strip() for t in expected.split(",")}
    got = {r.trigger for r in context.watcher.interventions}
    assert wanted.issubset(got), f"expected {wanted} ⊆ {got}"


@when("the FSM advances LLM_SWEEP → VALIDATING after triggers fired")
def step_fsm_validating(context):
    nautilus_register_state(context.nautilus_run_id, BootstrapState())
    context.watcher.tick()  # prime LLM_SWEEP heartbeat
    context.watcher._clock = lambda: 500.0
    context.watcher.tick()  # fires triggers
    context.nautilus_fsm.advance(context.nautilus_run_id, FSMState.VALIDATING)
    context.watcher._clock = lambda: 700.0
    context.watcher.tick()  # VALIDATING phase just entered — no trigger yet


@when("{sec:d} seconds pass in VALIDATING with no audit activity")
def step_validating_seconds(context, sec):
    context.watcher._clock = lambda: 700.0 + float(sec)
    context.watcher.tick()


@then("the watcher fires a new stall trigger in VALIDATING")
def step_validating_stall(context):
    val_stalls = [
        r for r in context.watcher.interventions
        if r.fsm_state == "VALIDATING" and r.trigger == "stall"
    ]
    assert len(val_stalls) == 1, (
        f"expected 1 VALIDATING stall, got {len(val_stalls)}"
    )


@when("the FSM is in LLM_SWEEP with {n:d} failed batch entries")
def step_seed_failed_audit(context, n):
    from atelier.classify.bootstrap import BatchAttempt
    state = BootstrapState()
    for i in range(n):
        state.batch_audit.append(BatchAttempt(
            batch_index=i, col_count=1, depth=1, status="failed",
            error_class="TimeoutError",
        ))
    nautilus_register_state(context.nautilus_run_id, state)
    context.watcher.tick()
    context.watcher._clock = lambda: 30.0  # well under stall threshold
    context.watcher.tick()


@then('the watcher fires trigger "{name}"')
def step_watcher_fires_single(context, name):
    got = {r.trigger for r in context.watcher.interventions}
    assert name in got, f"expected {name} in {got}"


@when("the watcher ticks past a stall threshold with a cancel decision")
def step_watcher_cancel_tick(context):
    context.watcher.tick()  # prime heartbeat
    context.watcher._clock = lambda: 500.0
    context.watcher.tick()


@then("the BootstrapState cancelled flag is set")
def step_cancelled_set(context):
    assert context.nautilus_state.cancelled is True


@then("the cancellation reason matches the decision")
def step_cancel_reason(context):
    assert "test" in context.nautilus_state.cancellation_reason


@then("the BootstrapState cancelled flag is false")
def step_cancelled_false(context):
    assert context.nautilus_state.cancelled is False


# ── Controlled CLIs ────────────────────────────────────────────


@given('overwatch autonomy is "{tier}"')
def step_autonomy_tier(context, tier):
    # The CLI reads load_config(), so pin the env var rather than
    # mutating config_overlay.  load_config() is called fresh on every
    # CLI invocation so the override is picked up.
    context.prior_autonomy_env = os.environ.get("ATELIER_OVERWATCH_AUTONOMY")
    os.environ["ATELIER_OVERWATCH_AUTONOMY"] = tier
    context.add_cleanup(_restore_autonomy, context)


def _restore_autonomy(context):
    if context.prior_autonomy_env is None:
        os.environ.pop("ATELIER_OVERWATCH_AUTONOMY", None)
    else:
        os.environ["ATELIER_OVERWATCH_AUTONOMY"] = context.prior_autonomy_env


@when('I invoke apply_and_rerun for run "{run_id}"')
def step_invoke_apply(context, run_id):
    from atelier.overwatch.apply_and_rerun import main
    context.cli_rc = main([run_id, "--dry-run"])


@when('I invoke kill_run for run "{run_id}" with reason "{reason}"')
def step_invoke_kill(context, run_id, reason):
    from atelier.overwatch.kill_run import main
    context.cli_rc = main([run_id, "--reason", reason])


@when('I invoke write_proposal with an overlay touching "{key}"')
def step_invoke_write(context, key):
    import json as _json
    from atelier.overwatch.write_proposal import main
    payload = {
        "overlay": {key: "anthropic"},
        "rationale": "test",
        "expected_effect": "test",
        "trigger": "post_mortem",
    }
    context.cli_rc = main(["test_run", "--json", _json.dumps(payload), "--dry-run"])


@then("the CLI exits with code {rc:d}")
def step_cli_rc(context, rc):
    assert context.cli_rc == rc, f"expected rc={rc}, got {context.cli_rc}"


# ── Hook sandbox ───────────────────────────────────────────────


@when('I evaluate the Bash hook with command "{cmd}"')
def step_eval_bash(context, cmd):
    context.hook_decision = evaluate_hook("Bash", {"command": cmd})


@when('I evaluate the Read hook on path "{path}"')
def step_eval_read(context, path):
    context.hook_decision = evaluate_hook("Read", {"file_path": path})


@when("I evaluate the Write hook with an empty input")
def step_eval_write(context):
    context.hook_decision = evaluate_hook("Write", {})


@then('the hook decision is "{decision}"')
def step_hook_decision(context, decision):
    assert context.hook_decision["decision"] == decision, (
        f"expected {decision}, got {context.hook_decision}"
    )
