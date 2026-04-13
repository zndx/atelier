# ANTHROPIC_SUBAGENT_MODEL: Configurable Classification Model

## Summary

Replaced the hardcoded 3-tier credential-based auto-default logic in
`create_backend_from_cfg()` with a clean, configurable approach using a new
`ANTHROPIC_SUBAGENT_MODEL` HOCON config. Backend type (direct API vs Bedrock)
is inferred from the model identifier format. Fail-fast when no config is set.

## Problem

The previous implementation hardcoded Haiku 4.5 as the classification model
and used a cascade of credential checks (ANTHROPIC_API_KEY → Bedrock creds)
to guess the backend. Two issues:

1. Wrong model class — classification is subagent-level work, should be sonnet
2. No configuration knob — operators couldn't override the model without
   switching to the completely different `ATELIER_LLM_*` backend path

## Resolution Order (new)

```
1. Explicit classify LLM (ATELIER_LLM_API_KEY / ATELIER_LLM_BASE_URL)
   → configured backend (unchanged)

2. ANTHROPIC_SUBAGENT_MODEL is set
   → is_bedrock_model(model) → BedrockStructuredBackend
   → else → AnthropicStructuredBackend

3. Neither → ValueError (fail fast)
```

## Changes

### `src/atelier/config.py`
- Added `is_bedrock_model()` shared utility (reuses pattern from agents/client.py)
- Added `classify_subagent_model` field + HOCON mapping + ENV special case
- Simplified `has_classify_llm` — no more bare credential probes

### `src/atelier/agents/client.py`
- Replaced inline Bedrock detection with `is_bedrock_model()` import

### `config/base.conf`
- Added `classify.subagent_model` with `${?ANTHROPIC_SUBAGENT_MODEL}`

### `src/atelier/classify/llm_backend.py`
- Removed `ANTHROPIC_STRUCTURED_MODEL` and `BEDROCK_STRUCTURED_MODEL` constants
- Rewrote `create_backend_from_cfg()` with fail-fast semantics
- Added `_build_anthropic_backend()` and `_build_bedrock_backend()` helpers

### `src/atelier/gateway.py`
- Updated error message to mention `ANTHROPIC_SUBAGENT_MODEL`

### `src/atelier/classify/pipeline.py`
- Updated module and function docstrings

### `features/agent/backend.feature`
- Replaced 3 credential-based auto-default scenarios with:
  - Subagent model (Bedrock format) → BedrockStructuredBackend
  - Subagent model (plain Anthropic) → AnthropicStructuredBackend
  - No config → ValueError

### `features/agent/step_defs/backend_steps.py`
- Added parameterized Given steps for subagent model scenarios
- Added "attempt to call create_backend_from_cfg" When step

## Verification

- BDD tier-0 fast: 17 features passed, 63 scenarios passed, 0 failed
