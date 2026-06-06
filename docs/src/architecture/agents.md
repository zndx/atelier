# Keystone Agents

Atelier uses the Claude Agent SDK to drive classification convergence.
Rather than a fixed programmatic loop, an LLM agent reasons about which
columns to revisit based on DST conflict metrics, evidence breakdowns,
and convergence trends.

## Agent Convergence Loop

The agent loop (`src/atelier/classify/agent_loop.py`) wraps the bootstrap
pipeline functions as six Claude tools. Claude receives an initial state
summary and iteratively calls tools until it determines the classification
has converged.

### Flow

```
1. Initial state → agent sees mean gap, mean belief, coverage, K (diagnostic)
2. Agent calls get_conflict_report → identifies uncertain columns (high gap or low belief)
3. Agent calls get_column_detail → inspects per-source evidence breakdown
4. Agent calls revisit_columns → re-classifies with enriched context
5. Agent calls check_convergence → verifies gap trend + belief floor
6. Repeat 2-5 until satisfied
7. Agent calls declare_converged with reason
```

The conversation loop runs up to `classify_agent_max_turns` (default 10)
Messages API round-trips. Each tool call returns structured JSON that the
agent uses to plan its next action.

### Five Tools

| Tool | Input | Returns | Purpose |
|------|-------|---------|---------|
| `get_conflict_report` | `k_threshold` (float) | Flagged columns with K, belief, plausibility, gap, settled flag | Identify uncertain or conflicting columns |
| `revisit_columns` | `column_names` (list) | Updated labels + new belief intervals | Re-classify with enriched LLM context (ML prediction + belief interval) |
| `check_convergence` | — | mean_gap, mean_bel, frac_unclear, coverage, K (diagnostic), iteration history | Assess convergence via belief-gap criteria |
| `get_column_detail` | `column_name` (string) | Per-source evidence breakdown, sample values, belief interval | Deep-dive into a specific column |
| `declare_converged` | `reason` (string) | Confirmation | Exit loop with stated rationale |

> **Historical note (2026-05-04 refactor).** Earlier revisions of
> the agent loop included a sixth `retrain_svm` tool that retrained
> the SVM on accumulated LLM labels and hot-swapped the result.
> That tool was removed alongside the M9 in-loop SVM-on-LLM-labels
> retrain machinery (commits 8627c2c, 5199379, cc59d01) for the
> source-independence reasons documented in `ontology_alignment.py`. The
> SVM is now trained once on synth and translated into the user
> vocabulary at inference time; there is no per-run SVM retraining
> for the agent to drive.

### Agent System Prompt

The system prompt guides the agent's strategy:

1. Examine the conflict report to understand where sources disagree
2. Inspect individual columns for uncertain cases (high gap or low belief)
3. Revisit uncertain columns to resolve ambiguity
4. Check convergence metrics (mean gap, mean belief, coverage) to decide
   whether to continue — K is available as a diagnostic but does not gate
5. Declare convergence when satisfied (or when diminishing returns)

### State Tracking

The agent loop tracks:

- `state.agent_reasoning` — text blocks from each agent turn
- `state.agent_converged_reason` — the reason given at convergence
- `state.agent_turns` — number of conversation turns
- `state.tokens_input` / `state.tokens_output` — token consumption

Each `revisit_columns` call increments `state.iteration` and triggers
full ML revalidation on all columns, not just the revisited ones. This
ensures that improved LLM labels propagate through the DST fusion.

## LLM Backend Matrix

The agent loop and LLM sweep share the same backend infrastructure.
No global provider switch — credentials determine what's available.

| Backend | Class | Config | Use Case |
|---------|-------|--------|----------|
| Anthropic | `AnthropicBackend` | `ANTHROPIC_API_KEY` | Agent loop + LLM sweep |
| Bedrock | `BedrockBackend` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_REGION` | Production default on CAI |
| Cerebras | `CerebrasBackend` | `CEREBRAS_API_KEY` | Fast inference via GLM-4.7 |
| OpenAI-compatible | `OpenAICompatibleBackend` | `ATELIER_LLM_BASE_URL` + `ATELIER_LLM_MODEL` | vLLM, any compatible endpoint |

The agent client is built via `_build_client(cfg)` which prefers Anthropic
when `ANTHROPIC_API_KEY` is set, falling back to Bedrock when AWS credentials
are available. The agent model resolves as:
`classify_agent_model` → `agent_model` → `"claude-sonnet-4-5-20250929"`.

## Configuration

All agent and bootstrap settings live in HOCON (`config/base.conf`):

```hocon
classify {
    llm {
        backend = "openai_compatible"
        model = "glm-4.7"
        base_url = null
        columns_per_call = 50
        discount = 0.10
    }
    bootstrap {
        max_iterations = 5
        k_threshold = 0.2
        coverage_target = 0.95
        max_total_llm_calls = 5000
        # Historical: these knobs gated the excised M9 in-loop SVM
        # retrain.  Retained here only as illustration of the legacy
        # config surface; the keys are no longer read by the pipeline.
        # incremental_svm_retrain = true
        # incremental_svm_min_labels = 20
    }
}

agent {
    model = "claude-sonnet-4-5-20250929"
    model = ${?ATELIER_AGENT_MODEL}
}

classify {
    agent_model = null
    agent_model = ${?ATELIER_CLASSIFY_AGENT_MODEL}
    agent_max_turns = 10
}
```

When `classify.agent_model` is set, it overrides `agent.model` for the
classification convergence loop specifically.

## Agent vs Programmatic Loop

The bootstrap pipeline (`bootstrap.py`) contains the programmatic
convergence loop as well: sweep → validate → revisit uncertain → repeat.
The agent loop is an alternative that delegates the revisit strategy to
Claude. Both paths share the same underlying functions (`_llm_sweep`,
`_run_ml_validation`, etc.) and produce identical DST evidence.

The agent approach is preferred when:
- The corpus has wide-belief-gap columns where independent evidence
  sources disagree in non-obvious ways
- You want reasoning traces explaining why convergence was declared
- The LLM backend supports tool_use (Anthropic, Bedrock with Claude)

The programmatic approach is used when:
- The LLM backend doesn't support tool_use (vLLM, Cerebras)
- Deterministic behavior is required
- Cost must be minimized (fewer API calls)

## WebSocket Orchestration

The gateway exposes `/ws/orchestration` for live agent event streaming.
Events include `agent_spawned`, `agent_reasoning`, `agent_tool_call`,
and `agent_completed`. The React frontend's Agent Canvas page consumes
these events to render the agent's decision process in real time.
