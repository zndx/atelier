<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# BedrockStructuredBackend: Complete LLM Auto-Default

## Summary

Added `BedrockStructuredBackend` to complete the LLM auto-default hierarchy.
In production CAI where only Bedrock credentials exist (no ANTHROPIC_API_KEY),
the classification pipeline now auto-defaults to Haiku 4.5 on Bedrock with
structured output via `invoke_model`.

## Problem

The `AnthropicStructuredBackend` (added in prior session) only worked when
`ANTHROPIC_API_KEY` was present. Production CAI uses Bedrock with AWS
credentials — the pipeline couldn't auto-default to an LLM, defeating
the consolidation.

## Resolution Order

```
1. Explicit classify LLM (ATELIER_LLM_API_KEY / ATELIER_LLM_BASE_URL)
   → configured backend (openai_compatible, cerebras, bedrock, etc.)

2. ANTHROPIC_API_KEY
   → AnthropicStructuredBackend + Haiku 4.5 (dev default)

3. AWS Bedrock credentials (has_bedrock)
   → BedrockStructuredBackend + Haiku 4.5 on Bedrock (production default)

4. Nothing → ValueError
```

## Key Design Decision

The `AnthropicBedrock` SDK wrapper and the Bedrock Converse API do NOT
support `output_config` (structured JSON Schema output). The only path
that supports it is `boto3.client("bedrock-runtime").invoke_model()` with
the raw Anthropic Messages API format (`anthropic_version: "bedrock-2023-05-31"`).

This is why we have both:
- `BedrockBackend` — Converse API, for explicit `backend=bedrock` config
- `BedrockStructuredBackend` — invoke_model, for auto-default with structured output

## Changes

### `src/atelier/classify/llm_backend.py`
- Added `BEDROCK_STRUCTURED_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"`
- Added `BedrockStructuredBackend` class (invoke_model + output_config + cache_control)
- Extracted `_parse_structured_response()` as shared module-level function
  (used by both `AnthropicStructuredBackend` and `BedrockStructuredBackend`)
- Updated `create_backend()` to recognize `"bedrock_structured"`
- Updated `create_backend_from_cfg()` with Bedrock as tier 3
  - Uses `cfg.agent_default_haiku_model` when set (production has
    `ANTHROPIC_DEFAULT_HAIKU_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0`)
  - Falls back to `BEDROCK_STRUCTURED_MODEL` constant

### `src/atelier/config.py`
- Updated `has_classify_llm` to include `self.has_bedrock`

### `src/atelier/gateway.py`
- Updated error message to mention all three credential paths

### `src/atelier/classify/pipeline.py`
- Updated module and function docstrings to mention Bedrock

### `features/agent/backend.feature`
- Added 5 new scenarios: factory for bedrock_structured, factory for
  anthropic_structured, auto-default Bedrock only, auto-default
  Anthropic only, auto-default prefers Anthropic over Bedrock

### `features/agent/step_defs/backend_steps.py`
- Added step definitions for all new scenarios

## Verification

- UI build: clean (`pnpm build`)
- BDD tier-0: 17 features passed, 63 scenarios passed (up from 58), 0 failed
- 7 auto-default assertion scenarios all pass:
  1. Bedrock only → BedrockStructuredBackend
  2. Anthropic only → AnthropicStructuredBackend
  3. Both → prefers Anthropic (direct API)
  4. Explicit LLM → uses configured backend
  5. No creds → ValueError
  6. Bedrock uses agent_default_haiku_model when set
  7. Bedrock falls back to BEDROCK_STRUCTURED_MODEL constant
