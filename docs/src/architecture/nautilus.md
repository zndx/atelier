# Nautilus — Mid-Run Pipeline Watcher

Nautilus is the in-process, mid-run watcher for a classification run. A
daemon thread polls the [FSM](./classification.md) and
`BootstrapState.batch_audit`, decides when a run is going sideways, and
hands a structured `InterventionRecord` to a callback. The callback —
not nautilus — decides what to do (record, cancel, escalate).

The thread itself is observation + decision framing. It owns no
LLM-calling code, holds no agent context, and never kills a process.
That separation keeps nautilus testable without tool-using agents in
the loop and lets the same trigger logic serve both the gateway's
auto-cancel hook and the supervisor [Overwatch](./agents.md) post-mortem.

## Why it exists

UAT surfaced a class of failure where the pipeline *stopped making
progress* without erroring — typically a frozen LLM sweep on a
problem batch with no heartbeat advance for 20+ minutes. The FSM still
read `LLM_SWEEP`; nothing was wrong from the FSM's point of view. The
operator either waited or killed the process by hand.

Nautilus closes that gap. It pairs with two other layers of
self-remediation in the pipeline:

| Pillar | Where it lives | What it does |
|---|---|---|
| **1 — Halving retry** | [`classify/bootstrap.py`](https://github.com/zndx/atelier/blob/trunk/src/atelier/classify/bootstrap.py) | Per-batch: on LLM failure, halve `columns_per_call` and retry until single-column or success. Preserves 100% column coverage. |
| **2 — Nautilus** (this doc) | [`overwatch/nautilus.py`](https://github.com/zndx/atelier/blob/trunk/src/atelier/overwatch/nautilus.py) | Per-run: observe FSM + batch_audit, fire an intervention when the run stalls, sweeps too long, or accumulates failures. |
| **3 — Supervisor Overwatch** | [`overwatch/agent.py`](https://github.com/zndx/atelier/blob/trunk/src/atelier/overwatch/agent.py), [`apply_and_rerun.py`](https://github.com/zndx/atelier/blob/trunk/src/atelier/overwatch/apply_and_rerun.py) | Post-run: read the latest intervention record, propose a config overlay, optionally rerun. Multi-attempt session in `overwatch/session.py`. |

## Triggers

Each trigger fires at most once per FSM phase. Phase change resets the
phase-scoped triggers (`stall`, `slow_llm_sweep`) so a long run can
record one intervention per phase rather than storming every poll.

| Trigger constant | Fires when | Default threshold |
|---|---|---|
| `TRIGGER_STALL` (`"stall"`) | No new `batch_audit` activity for `stall_threshold_s` while FSM is in a non-terminal state. | 120 s |
| `TRIGGER_SLOW_SWEEP` (`"slow_llm_sweep"`) | `fsm.state == LLM_SWEEP` for more than `llm_sweep_threshold_s`, regardless of batch progress. | 300 s |
| `TRIGGER_FAILED_BATCHES` (`"failed_batches"`) | Count of `batch_audit` entries with status `failed` or `fatal` exceeds `failed_batch_threshold`. | 10 |
| `TRIGGER_FSM_ERROR` (`"fsm_error"`) | Pipeline transitioned to `ERROR`. Unconditional; bypasses other evaluation. | — |

`evaluate_triggers()` is a pure function of heartbeat + config — tests
exercise it directly with a seeded `_Heartbeat` and synthetic clock,
no threads required.

## How it observes

- **State registry** (module-level `_state_registry`): the pipeline
  calls `register_state(run_id, state)` early in `run_classification_pipeline`
  and `unregister_state(run_id)` in the finally block. The registry
  is lock-guarded so nautilus never observes a partially-destructed
  state during teardown.
- **FSM polling** (`tick`): `fsm.get_status(run_id)` each poll
  (default every 10 s). Phase change resets phase-scoped triggers and
  the phase-entry clock.
- **batch_audit tail**: nautilus counts entries and failed entries.
  The pipeline appends to `state.batch_audit` between LLM calls so
  the audit length acts as the heartbeat — its non-advance is the
  stall signal.

## Dispatch and cooperative cancel

When a trigger fires, `_dispatch` builds an `InterventionRecord`,
appends it to `watcher.interventions`, and invokes
`intervene_callback(rec)` if one was supplied. The callback returns a
dict with a `decision` field (`"observed"` | `"intervened"` |
`"cancelled"`) and an optional `reason`.

If `decision == "cancelled"` *and* `cfg.can_cancel` is true, nautilus
flips `state.cancelled = True` on the registered `BootstrapState`. The
pipeline checks this flag between LLM batches in
[`bootstrap.py`](https://github.com/zndx/atelier/blob/trunk/src/atelier/classify/bootstrap.py)
and exits cleanly. **There is no SIGKILL path.** An in-flight LLM
call finishes before the run terminates. This is what "cooperative
cancel" means.

`can_cancel` is gated by `overwatch.autonomy`:

| Autonomy mode | `can_cancel` | What nautilus does on stall |
|---|---|---|
| `monitor` | false | Record only. |
| `propose` | false | Record only. The supervisor reads the record post-run. |
| `autonomous` | true | Flip `state.cancelled` so the pipeline exits. |

The gateway's default callback ([`gateway.py:2738`](https://github.com/zndx/atelier/blob/trunk/src/atelier/gateway.py))
always returns `{"decision": "cancelled"}` — so the autonomy gate is
the *only* thing keeping `propose` / `monitor` runs from auto-cancelling.

## Deployment gates — Bedrock-only and direct-Anthropic

Nautilus runs without the direct Anthropic API. It makes no LLM calls
of its own — it observes, decides, and hands a record to a callback.
The Anthropic gate (`cfg.has_overwatch`) only applies to **Pillar 3**
— the post-run supervisor agent that *consumes* nautilus's records
and proposes config overlays. Pillar 2 (this watcher) is upstream of
that gate.

Three independent gates drive what nautilus actually does:

| Gate | Source | Affects |
|---|---|---|
| `overwatch.nautilus.enabled` | HOCON / env (default `true`) | Whether the watcher attaches at all |
| `overwatch.autonomy == "autonomous"` | HOCON / env (default `propose`) | Whether nautilus can flip `state.cancelled` itself; whether `kill_run` CLI is permitted |
| `cfg.has_overwatch` (= `overwatch.enabled AND has_anthropic`) | derived from `ANTHROPIC_API_KEY` | Whether Pillar 3 supervisor agent runs post-run |

Capability matrix on a Bedrock-only deployment (no
`ANTHROPIC_API_KEY` — typical for CAI on Bedrock or air-gapped
environments):

| Capability | Bedrock + `propose` (default) | Bedrock + `autonomous` |
|---|---|---|
| Watcher thread starts and polls | ✅ | ✅ |
| Trigger detection (stall / slow-sweep / failed-batches / fsm_error) | ✅ | ✅ |
| `InterventionRecord`s queryable via `/api/overwatch/nautilus*` | ✅ | ✅ |
| Operator UI Stop (`POST /api/fsm/cancel`) | ✅ — never autonomy-gated | ✅ |
| Auto-cancel on stall (nautilus → `state.cancelled`) | ❌ recorded only — `can_cancel=False` | ✅ |
| `kill_run` CLI | ❌ rejected (autonomy gate) | ✅ |
| Post-run supervisor agent (proposes overlay, optional rerun) | ❌ requires direct Anthropic API | ❌ requires direct Anthropic API |

To unlock auto-cancel without adding an Anthropic key, set
`ATELIER_OVERWATCH_AUTONOMY=autonomous`. Autonomy is independent of
`has_overwatch`. The trade-off: nautilus will cancel runs based on
threshold rules alone, with no AI judgement layer behind the
decision.

## Config

```hocon
overwatch {
  autonomy = "propose"  # monitor | propose | autonomous

  nautilus {
    enabled = true
    poll_interval_s = 10.0
    stall_threshold_s = 120.0
    llm_sweep_threshold_s = 300.0
    failed_batch_threshold = 10
  }
}
```

Environment overrides: `ATELIER_OVERWATCH_NAUTILUS_ENABLED`,
`ATELIER_OVERWATCH_NAUTILUS_POLL_INTERVAL_S`,
`ATELIER_OVERWATCH_NAUTILUS_STALL_THRESHOLD_S`,
`ATELIER_OVERWATCH_NAUTILUS_LLM_SWEEP_THRESHOLD_S`,
`ATELIER_OVERWATCH_NAUTILUS_FAILED_BATCH_THRESHOLD`.

`nautilus_config_from_cfg(cfg)` reads these and fills `can_cancel`
from `overwatch.autonomy == "autonomous"`.

## HTTP surface

Both routes are read-only. Cancellation goes through the operator
"Stop run" UI control or the `kill_run` CLI; nautilus does not expose
a cancel endpoint of its own.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/overwatch/nautilus/{run_id}` | Watcher snapshot for a specific run: heartbeat, intervention list, cancelled flag. |
| `GET` | `/api/overwatch/nautilus` | All active watchers (typically one — runs are single-flight). |

The watcher object is held in a module-level `_active_watchers` map
so the gateway can answer status queries without plumbing the
reference through pipeline internals. Intervention history survives
watcher stop and remains queryable until the gateway restarts.

## Operator CLI: cooperative kill

```bash
uv run python -m atelier.overwatch.kill_run <run_id> \
    --reason "stuck on partner-data sweep" \
    [--session <supervisor-session-id>]
```

`kill_run` looks the run up in the nautilus registry, sets
`state.cancelled = True`, and stops the watcher. **Gated to
`autonomous` mode** — in `propose` / `monitor` an operator must use
the UI Stop control, since the supervisor (which calls this CLI in
`autonomous` mode) isn't yet authorized to cancel on its own. With
`--session`, the cancel is appended to the supervisor session's
intervention log via [`overwatch.session.record_intervention`](https://github.com/zndx/atelier/blob/trunk/src/atelier/overwatch/session.py).

## Lifecycle

1. Operator hits `POST /api/fsm/start`. Gateway spawns the pipeline thread.
2. Gateway polls `fsm.get_status()` for up to ~1 s waiting for the
   pipeline to claim a `run_id`, then constructs a `NautilusWatcher`,
   registers it in `_active_watchers`, and starts the daemon thread.
3. Pipeline calls `register_state(run_id, state)` early in
   `run_classification_pipeline`. From here nautilus can observe.
4. On each `poll_interval_s` tick: read FSM, refresh heartbeat,
   evaluate triggers, dispatch records, repeat.
5. On terminal state (`IDLE` / `CONVERGED` / `ERROR`) the watcher's
   `tick` returns `True` and the thread exits. Pipeline's `finally`
   calls `unregister_state(run_id)` and the gateway calls
   `clear_active_watcher(run_id)`.

## Testing

The watcher is split deliberately to keep tests synchronous:

- **Trigger logic** — drive `evaluate_triggers(state_name=..., now=...,
  failed_count=...)` against a watcher with a hand-seeded `_Heartbeat`
  and a fake clock. No threads, no FSM, no `BootstrapState`.
- **Dispatch / cancel** — instantiate a `NautilusWatcher` with a fake
  `intervene_callback` and a stub FSM; assert that the callback
  return value drives `state.cancelled` correctly under each
  `can_cancel` value.
- **End-to-end** — BDD scenarios under `features/agent/` cover the
  registry round-trip and the gateway routes; the slow-path watch is
  not exercised in CI (it would require a real long-running sweep).
