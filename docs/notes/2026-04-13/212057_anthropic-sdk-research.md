# Anthropic SDK Research: Classification Backend Options

Research date: 2026-04-13

## Summary

Three distinct Anthropic SDK products exist. The `anthropic` Python package (Messages
API with structured output) is the correct choice for classification backends. The
Claude Agent SDK (`claude-agent-sdk`) is designed for agentic workflows with tool use
and is overkill for batch classification. Both share the same `ANTHROPIC_API_KEY`.

## 1. Three SDKs Clarified

### anthropic (Python SDK / Client SDK)
- **Package**: `anthropic` (PyPI), already in pyproject.toml at `>=0.84.0`
- **Purpose**: Direct API access -- messages, structured output, batch processing
- **Key method**: `client.messages.create()` and `client.messages.parse()`
- **Structured output**: Native JSON Schema via `output_config.format` (GA, not beta)
- **Best for**: Classification pipeline -- deterministic, schema-validated JSON

### claude-agent-sdk (Agent SDK, formerly Claude Code SDK)
- **Package**: `claude-agent-sdk` (PyPI), already in pyproject.toml at `>=0.1.0`
- **Purpose**: Agentic workflows with built-in tools (Read, Write, Bash, Grep, etc.)
- **Key method**: `query()` async generator with `ClaudeAgentOptions`
- **Structured output**: Via `output_format` option on `query()`
- **Best for**: Orchestration canvas agents, not batch classification
- **Note**: Renamed from claude-code-sdk; migration guide available

### anthropic-sdk-typescript (@anthropic-ai/sdk)
- TypeScript equivalent of the Python SDK; not relevant here

## 2. Structured Output -- The Key Feature

The `anthropic` SDK now has GA structured output (no beta headers needed):

```python
from pydantic import BaseModel
from anthropic import Anthropic

class ColumnClassification(BaseModel):
    column_name: str
    category_code: str | None
    confidence: float
    evidence: str
    alternatives: list[dict]

client = Anthropic()
response = client.messages.parse(
    model="claude-haiku-4-5",
    max_tokens=4096,
    output_format=ColumnClassification,  # Pydantic model
    messages=[{"role": "user", "content": prompt}],
)
# response.parsed_output is a validated ColumnClassification
```

Or for batch arrays, use `output_config` with JSON Schema directly:

```python
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=4096,
    system=system_prompt,
    messages=[{"role": "user", "content": user_prompt}],
    output_config={
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "column_name": {"type": "string"},
                        "category_code": {"type": ["string", "null"]},
                        "confidence": {"type": "number"},
                        "evidence": {"type": "string"},
                        "alternatives": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "code": {"type": "string"},
                                    "confidence": {"type": "number"}
                                },
                                "required": ["code", "confidence"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["column_name", "category_code", "confidence", "evidence", "alternatives"],
                    "additionalProperties": False
                }
            }
        }
    },
)
```

This eliminates all the regex/JSON parsing fallbacks in `_parse_classifications()`.

## 3. Batch API for High Throughput

The Messages Batch API (`client.messages.batches`) supports:
- Up to 10,000 requests per batch
- 50% cost discount vs synchronous API
- Processing within 24 hours
- Compatible with structured outputs
- Extended output (300k tokens) for Opus 4.6 / Sonnet 4.6

For classification: submit all column batches as a single batch job, poll for
completion, collect results. Ideal for large-scale runs (thousands of columns).

## 4. Pricing (April 2026)

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Best for |
|-------|----------------------|------------------------|----------|
| Claude Haiku 4.5 | $1.00 | $5.00 | Classification (fast, cheap) |
| Claude Sonnet 4.6 | $3.00 | $15.00 | Complex classification |
| Claude Opus 4.6 | $5.00 | $25.00 | Overkill for classification |

**Batch API**: 50% off all prices above.
**Prompt caching**: Cache hits at 0.1x input price (90% discount).

### Cost estimate for classification pipeline

A typical batch of 50 columns:
- System prompt with taxonomy table: ~2,000 tokens
- User prompt (50 columns with metadata): ~3,000 tokens
- Response (50 classifications): ~2,500 tokens

Per batch with Haiku 4.5:
- Input: 5,000 tokens * $1.00/1M = $0.005
- Output: 2,500 tokens * $5.00/1M = $0.0125
- Total: ~$0.0175 per batch of 50 columns

For 1,000 columns (20 batches):
- Sync API: ~$0.35
- Batch API: ~$0.175
- With prompt caching: ~$0.12

With prompt caching, the system prompt (taxonomy table) is cached across batches,
reducing input costs by ~90% on subsequent calls.

## 5. CAI Credential Reuse

The `ANTHROPIC_API_KEY` set for Claude Code / Agent SDK is the same key used by the
`anthropic` Python SDK. The existing code already handles this:

- `cfg.anthropic_api_key` -- loaded from HOCON `agents.api_key` <- `${?ANTHROPIC_API_KEY}`
- `cfg.classify_llm_api_key` -- loaded from HOCON `classify.llm.api_key`
- Both can use the same key; the classification pipeline just needs
  `classify_llm_api_key` set to the same value

On Bedrock: `AnthropicBedrock()` client works identically for structured output.

## 6. Current Architecture vs Recommended Changes

### Current (llm_backend.py)
- `AnthropicBackend.classify_batch()`: Uses `client.messages.create()` with free-text
  response, then regex-parses JSON from the text
- No structured output enforcement -- relies on prompt engineering + fallback parsing
- Separate backends: Anthropic, OpenAI, Cerebras, Bedrock

### Recommended: `AnthropicStructuredBackend`
- Use `output_config` with JSON Schema for guaranteed valid output
- Eliminate `_parse_classifications()` regex fallbacks entirely
- Use `client.messages.parse()` with Pydantic models for type safety
- Add prompt caching for the system prompt (taxonomy table is static per run)
- Optional: Use Batch API for large-scale runs (>500 columns)

### Implementation plan
1. Add `AnthropicStructuredBackend(LLMBackend)` to `llm_backend.py`
2. Use `output_config` with the classification JSON Schema
3. Add prompt caching: wrap system prompt in `{"type": "text", "text": ..., "cache_control": {"type": "ephemeral"}}`
4. Register as `backend="anthropic_structured"` in factory
5. Default to Haiku 4.5 model for cost efficiency
6. Optional: Add `AnthropicBatchBackend` for Batch API support

## 7. Agent SDK -- When to Use It

The Agent SDK (`claude-agent-sdk`) is correct for the orchestration canvas agents
(the XYFlow topology), NOT for classification. The existing `agents/client.py` properly
uses it for:
- Credential validation (`validate_credentials`)
- Smoke tests (`run_smoke_test`)
- Multi-turn agentic workflows with tools

Classification is a single-turn structured extraction task -- the Messages API is the
right tool.
