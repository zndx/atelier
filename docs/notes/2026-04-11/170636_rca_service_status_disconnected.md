<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# RCA — "Service Status Disconnected" on CAI deploy

## TL;DR

Three distinct bugs were compounding to produce the user experience the
operator reported:

1. **Landing card is binary AND all-or-nothing.** `/api/status.connected`
   was computed as `grpc.ok AND postgres.ok AND qdrant.ok`. Any transient
   probe failure on PGlite or Qdrant flipped the whole app to
   "Disconnected" even though gRPC (the only service the frontend
   actually needs) was fine.
2. **Web-terminal exception handler hid the real error.** When the
   `claude` CLI subprocess exited non-zero, we rendered
   `Error: Command failed with exit code 1 / Check stderr output for details`.
   That stderr text is a hardcoded *placeholder* inside `claude-agent-sdk`
   (`subprocess_cli.py:616`) — the child's stderr is inherited, not piped,
   so the SDK has nothing to show. Unless the caller passes a
   `stderr=callback`, real CLI errors never reach the operator.
3. **Multi-line paste into the web terminal** was previously dispatching
   on every `\n`, sending fragments as separate prompts. Fixed in
   `8eb4636` but not yet redeployed at the time this RCA was written.

## Evidence

### Bug #1 — binary `connected`

`src/atelier/gateway.py` (before fix, `/api/status`):

```python
connected = all(
    checks.get(svc, {}).get("ok", False)
    for svc in ("grpc", "postgres", "qdrant")
)
return {**checks, "connected": connected}
```

`ui/src/pages/Landing.tsx:86`:

```tsx
value={status?.connected ? "Connected" : "Disconnected"}
```

Flakiness from either backend → red card on the Landing page.

Known flake sources we already know about from earlier debugging:

- **PGlite** (`pglite-socket@0.0.13`) wedges on rapid connect/disconnect
  cycles; we mitigated with `_reset_status_engine()` on failure, but
  the first probe after a wedge still fails once.
- **Qdrant** HTTP `/healthz` has a 3s timeout in
  `gateway.py:279`; Qdrant cold-start on CAI easily exceeds that on a
  loaded workspace.
- **Model discovery** side-effect: we run `check_model_upgrade(cfg)`
  inside `/api/status`. It calls `anthropic.Anthropic.models.list()`,
  which is an outbound HTTPS call. If that stalls, the whole status
  response stalls with it. (Cached for 1h, so only affects the first
  probe per hour.)

### Bug #2 — hidden CLI stderr

`claude_agent_sdk/_internal/transport/subprocess_cli.py:611-618`:

```python
if returncode is not None and returncode != 0:
    self._exit_error = ProcessError(
        f"Command failed with exit code {returncode}",
        exit_code=returncode,
        stderr="Check stderr output for details",  # ← placeholder
    )
    raise self._exit_error
```

The SDK only captures stderr when the caller passes a `stderr=` callback
in `ClaudeAgentOptions` (see `types.py:1194`). Our
`src/atelier/terminal.py:_query_sdk` and
`src/atelier/agents/client.py:_run_smoke_test_async` both omitted it.
When the CLI failed (auth, budget, bedrock transient, etc.) the
operator got a useless placeholder and had to go dig in gateway logs.

### Bug #3 — multi-line paste

`src/atelier/terminal.py:feed()` originally iterated character-by-
character and dispatched on every `\r`/`\n`. Pasted multi-line
content was treated as N separate prompts. Fixed in `8eb4636` with a
paste fast-path that preserves internal newlines and submits on the
trailing `\n`.

## Fixes applied in this session

| Bug | File | Change |
|---|---|---|
| #1 | `src/atelier/gateway.py` | `connected` now tracks gRPC only; new `degraded` flag when postgres/qdrant are down but gRPC is up |
| #1 | `ui/src/pages/Landing.tsx` | Tri-state card: Connected / **Degraded** / Disconnected, with a per-service detail line when degraded |
| #2 | `src/atelier/terminal.py` | `stderr=_capture_stderr` callback + tail-20 surfacing on exception, showing real CLI output |
| #2 | `src/atelier/agents/client.py` | Same stderr callback in smoke test; error envelope now includes real tail |
| #3 | (already in `8eb4636`) | Paste fast-path — pending redeploy |

## Remediation / next steps for the operator

1. **Redeploy from `trunk`.** The commit landing these fixes is
   downstream of `8eb4636`; the paste fix won't take effect until
   that redeploy anyway.
2. Once redeployed, the web terminal will show real stderr on CLI
   failures. The next "exit code 1" should come with actionable
   output — likely budget exceeded, bedrock quota, or auth.
3. The Landing card will show **Degraded** instead of **Disconnected**
   when PGlite/Qdrant flap. The detail line under the card tells
   you which one.
4. If the operator still sees "Disconnected" (red), that means
   **gRPC itself is down** — check `bin/start-app.sh` orchestration
   and `devenv` process state. gRPC going down on CAI is usually
   a downstream symptom of either proto-stub mismatch (stale `just
   proto`) or server crash on startup; check the gateway log for
   the Python traceback.

## Likely source of the "exit code 1"

Given the user saw a successful 42-second query followed immediately
by an exit-1, the strongest candidates are, in order:

1. **`max_budget_usd=0.25` tripped**. One large tool-using turn can
   burn through $0.25 on Opus 4.6. Budget is per-query, not cumulative,
   but a follow-up prompt that does aggressive searches can hit it
   fast.
2. **Bedrock transient** — throttling, 5xx, sub-model cross-region
   fallback. We pin sub-models in `_build_sdk_env()` but a single
   `InvokeModel` failure bubbles up as CLI exit-1.
3. **Tool search / experimental beta mis-dispatch**. We default to
   `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1` on the Bedrock path,
   but only when the HOCON value is `null`. If `.env` or the AMP
   env injects a different value, tool-search can silently dispatch
   to a direct-API model that doesn't exist on Bedrock → immediate
   exit-1.

All three will now show the underlying error message in the web
terminal once the stderr callback is live.

## Reference

- `claude_agent_sdk/_internal/transport/subprocess_cli.py:611-618`
  (placeholder ProcessError)
- `claude_agent_sdk/types.py:1194` (stderr callback option)
- `src/atelier/gateway.py:/api/status` (probe aggregation)
- `src/atelier/terminal.py:_query_sdk` (web terminal SDK dispatch)
- `src/atelier/agents/client.py:_run_smoke_test_async` (smoke test)
- Prior terminal fix commit: `8eb4636`
