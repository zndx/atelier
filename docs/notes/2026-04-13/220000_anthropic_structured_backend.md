# AnthropicStructuredBackend: Default LLM for Classification

## Summary

Added `AnthropicStructuredBackend` as the default classification LLM backend
when `ANTHROPIC_API_KEY` is available (which is always true in Claude Code
and CAI environments).  The pipeline no longer falls back to ML-only/mock
mode — it always has an LLM available.

## Motivation

`ANTHROPIC_API_KEY` is always present because Claude Code itself requires it.
Falling back to ML-only classification when no *explicit* classify LLM was
configured was dishonest — it misrepresented the platform's capabilities.
The Anthropic Python SDK's structured output feature (`output_config` with
JSON Schema) provides guaranteed-valid responses via constrained decoding,
eliminating all regex/JSON parsing fallbacks.

## Changes

### `src/atelier/classify/llm_backend.py`
- Added `ANTHROPIC_STRUCTURED_MODEL = "claude-haiku-4-5-20251001"` — Haiku 4.5
  is ideal for classification ($1/$5 per M tokens, fast, sufficient accuracy)
- Added `CLASSIFICATION_OUTPUT_SCHEMA` — JSON Schema for structured output
  with `additionalProperties: false` on all objects
- Added `AnthropicStructuredBackend` class:
  - Uses `output_config={"format": {"type": "json_schema", "schema": ...}}`
    for guaranteed valid JSON (no regex parsing needed)
  - System prompt sent with `cache_control: {"type": "ephemeral"}` for prompt
    caching (90% input token discount on subsequent calls)
  - `_parse_structured()` is a simple `json.loads()` — schema guarantees validity
  - Health check uses a minimal structured schema
- Updated `create_backend()` to recognize `"anthropic_structured"` type
- Updated `create_backend_from_cfg()` with three-tier resolution:
  1. Explicit classify LLM (ATELIER_LLM_API_KEY / ATELIER_LLM_BASE_URL)
  2. ANTHROPIC_API_KEY → auto-default to structured + Haiku 4.5
  3. ValueError if no credentials at all

### `src/atelier/config.py`
- Updated `has_classify_llm` property to also check `anthropic_api_key`
  (since it can serve as the classify LLM backend via auto-default)

### `src/atelier/gateway.py`
- Removed `use_mock = not cfg.has_classify_llm` auto-fallback
- Gateway now checks `has_classify_llm` and returns an error if no
  credentials are available (should never happen with Claude Code)
- Pipeline always runs with real LLM (or explicit `use_mock` for testing)

### `src/atelier/classify/pipeline.py`
- Updated module and function docstrings to reflect the new default behavior
- Removed references to "no LLM" fallback in comments

## Backend Resolution Order

```
ATELIER_LLM_API_KEY set?
  → Yes: Use explicit backend (openai_compatible, cerebras, bedrock, etc.)
  → No: ANTHROPIC_API_KEY set?
    → Yes: AnthropicStructuredBackend + Haiku 4.5 (auto-default)
    → No: ValueError (should never happen in production)
```

## Verification

- UI build: clean (`pnpm build`)
- BDD tier-0: 17 features passed, 0 failed, 58 scenarios passed
- Config assertions: `has_classify_llm` returns True when ANTHROPIC_API_KEY set
- Backend factory: auto-creates AnthropicStructuredBackend from ANTHROPIC_API_KEY
- Structured parse: JSON Schema output parses correctly
