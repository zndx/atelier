# Agent Loop Post-Validation Fixes

## Session Summary

Continuation session that completed live validation of the agentic classification
pipeline and committed post-validation fixes discovered during live runs.

## Changes Committed

`3ecf162` fix: agent loop post-validation fixes from live Cerebras+Sonnet run

### Three targeted fixes (10 lines across 3 files):

1. **FSM transition** (`fsm.py:44`): Added `CLASSIFYING` to `LLM_SWEEP` allowed
   transitions. The agent loop enters at `LLM_SWEEP` and after convergence, pipeline
   advances to `CLASSIFYING` for the final fusion pass. Without this, the first live
   run hit `ValueError: Invalid transition: LLM_SWEEP -> CLASSIFYING`.

2. **Result propagation** (`pipeline.py:397-399`): Added `agent_turns`,
   `agent_converged_reason`, and `agent_reasoning` to the pipeline result summary
   dict. These BootstrapState fields were populated correctly by the agent loop but
   never surfaced to the caller (showed as N/A in results).

3. **Turn limit reason** (`agent_loop.py:603-606`): When the agent exhausts
   `max_turns` without explicitly calling `declare_converged`, now records
   `"Turn limit reached (N turns) without explicit convergence"` in
   `state.agent_converged_reason`. Previously this was left as `None`.

## Live Validation Results

### Run 2 (this session)
- **Config**: Cerebras GLM-4.7 (batch sweep) + Claude Sonnet (agent reasoning)
- **Result**: accuracy=0.82, mean_K=0.6866, 14 LLM calls
- **Agent**: 10 turns (hit limit), 135K input / 32K output tokens
- **Reasoning**: 10 entries captured; agent correctly identified phone_pattern
  regex false positives as root cause of persistent high K
- **Iterations**: 3 bootstrap iterations with K decreasing 0.6842 -> 0.6861 -> 0.6866

### Key Insight from Agent
The agent identified that `phone_pattern` regex incorrectly matches ISO dates
and 10-digit account numbers, creating false positive evidence that overrides
correct LLM classifications. This is a real pattern detector bug worth fixing
in a future session.

## Regression
87 tier-0 scenarios passed, 0 failed (39 skipped @slow/@tier-1)
