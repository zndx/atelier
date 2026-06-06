# Agentic Classification: Claude SDK-Driven Convergence Loop

## Summary

Implemented the agent-driven convergence loop that replaces the programmatic
bootstrap loop with a Claude agent using the Messages API with tool_use.
The agent reasons about DST conflict and decides which columns to revisit.

## Design Decision: Messages API with tool_use

Used `client.messages.create()` with `tools=` rather than `claude_agent_sdk.query()`:
- In-process Python tool dispatch (not subprocess/CLI)
- JSON schemas with guaranteed shape
- Token counting per turn for cost control
- Testable with mock client (pre-built tool_use blocks)
- Works with both Anthropic and Bedrock clients

## New Files

| File | Purpose |
|------|---------|
| `src/atelier/classify/agent_loop.py` | Agent loop core: 5 tool definitions, handlers, dispatch, conversation loop |
| `features/agent/agent_loop.feature` | 6 BDD scenarios (5 tier-0, 1 @slow) |
| `features/agent/step_defs/agent_loop_steps.py` | Step definitions with mock Anthropic client |

## Modified Files

| File | Change |
|------|--------|
| `config/base.conf` | Added `classify.agent { enabled, max_turns, model }` block |
| `src/atelier/config.py` | Added 3 HOCON mappings + 3 dataclass fields |
| `src/atelier/classify/bootstrap.py` | Added `agent_reasoning`, `agent_turns`, `agent_converged_reason` to BootstrapState |
| `src/atelier/classify/pipeline.py` | Conditional branch: agent loop vs programmatic loop |
| `src/atelier/gateway.py` | WebSocket orchestration broadcast (client tracking + broadcast_orchestration_event) |
| `features/steps/__init__.py` | Re-export agent_loop_steps |

## Agent Tools (5)

| Tool | Handler | Wraps |
|------|---------|-------|
| `get_conflict_report` | Format high-K columns with disagreements | bootstrap._identify_disagreements |
| `revisit_columns` | LLM revisit + ML revalidation | bootstrap._llm_revisit + _run_ml_validation |
| `check_convergence` | Coverage, mean K, trend, history | bootstrap metrics functions |
| `get_column_detail` | Full evidence breakdown for one column | pipeline._classify_column |
| `declare_converged` | Record reason, signal loop exit | State update |

## Configuration

```hocon
classify.agent {
    enabled = false   # ${?ATELIER_CLASSIFY_AGENT_ENABLED}
    max_turns = 10    # ${?ATELIER_CLASSIFY_AGENT_MAX_TURNS}
    model = null      # ${?ATELIER_CLASSIFY_AGENT_MODEL} (falls back to agents.model)
}
```

## BDD Results

- 87 tier-0 scenarios passing (was 82, +5 new agent loop)
- 6 total new scenarios (5 fast + 1 @slow)
- 0 regressions
- Mock Anthropic client exercises full tool dispatch without API calls

## Key Design Principles

1. **Additive** — `classify_agent_enabled=false` by default, programmatic loop unchanged
2. **Same code paths** — agent tools call the same functions as the programmatic loop
3. **Observable** — reasoning captured in `state.agent_reasoning`, streamed via WebSocket
4. **Cost bounded** — `max_turns=10`, token tracking, early termination
5. **Testable without API** — mock client with scripted tool_use blocks
