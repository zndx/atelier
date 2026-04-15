# Bedrock CAI Runtime Fixes

**Date:** 2026-04-15

## Root Causes

Three runtime bugs that reproduce on every CAI redeployment:

1. **Bedrock `invoke_model` doesn't support `output_config`**: The
   `BedrockStructuredBackend` was using `output_config`/`json_schema`
   which is a direct-API-only feature. Bedrock silently ignored it,
   producing unstructured text that failed to parse.

2. **Cross-region inference profile ARNs need region-aware clients**:
   ARNs like `arn:aws:bedrock:us-west-2:...` encode the target region.
   Without extracting it, boto3 connects to the default `AWS_REGION`
   and gets `ResourceNotFoundException`.

3. **Silent 0-label convergence masks LLM failures**: When all LLM
   calls failed (wrong region, bad credentials), `_llm_sweep()` silently
   continued with zero labels and reported `CONVERGED` — completely
   masking the configuration error.

## Changes

### `src/atelier/config.py`
- Added `region_from_arn(model_id)`: extracts AWS region from a Bedrock
  ARN (4th colon-delimited field). Returns None for non-ARN model IDs.

### `src/atelier/agents/client.py`
- `_build_bedrock_client()`: uses `region_from_arn(cfg.agent_model)` with
  fallback to `cfg.aws_region`
- `_build_sdk_env()`: sets `AWS_REGION` and `AWS_DEFAULT_REGION` from
  ARN-derived region, ensuring the Claude CLI subprocess connects to
  the correct endpoint

### `src/atelier/classify/llm_backend.py`
- **`BedrockStructuredBackend`**: Replaced `output_config`/`json_schema`
  with forced tool-use (`tools` + `tool_choice`). The `classify_columns`
  tool's `input_schema` is the same JSON schema. Response parsing extracts
  the `tool_use` content block.
- Extended thinking compatibility: `tool_choice` = `"auto"` when thinking
  is enabled (Anthropic constraint), with text-block fallback parser.
- Health check migrated to tool-use pattern.
- Logging: reports item counts instead of text char counts.
- `BedrockBackend._get_client()`: ARN-aware region.
- `_build_bedrock_backend()`: ARN-aware region, logged.

### `src/atelier/classify/bootstrap.py`
- `_llm_sweep()`: tracks `batches_attempted` vs `batches_failed`.
  Raises `RuntimeError` with actionable message when ALL batches fail.

## Verification

- Unit tests: `region_from_arn`, `BedrockStructuredBackend` instantiation,
  fail-fast trigger all pass
- 98 tier-0 BDD scenarios pass (0 failed)
