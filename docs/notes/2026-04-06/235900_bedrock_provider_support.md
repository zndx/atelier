# Bedrock Provider Support

## What was done

Added AWS Bedrock as a supported provider alongside direct Anthropic API access.

### Config changes
- `config/base.conf` — added `provider`, `aws_access_key_id`, `aws_secret_access_key`, `aws_region`, `aws_session_token` to agents block
- `src/atelier/config.py` — added fields to `AtelierConfig`, HOCON mappings, `_FIELD_TO_ENV` entries (standard AWS var names), `is_bedrock` property with ARN auto-detection

### Client changes
- `src/atelier/agents/client.py` — added `_build_client()` (returns `Anthropic` or `AnthropicBedrock`), `_build_sdk_env()` (builds env dict for Claude Agent SDK), updated `validate_api_key()` and `run_smoke_test()` to be provider-aware

### Other
- `.env.example` — added Bedrock credential template
- `features/agent/agent_smoke.feature` — 2 new tier-0 scenarios for provider detection
- `features/agent/step_defs/agent_steps.py` — config-override step definitions

## BDD results
- 29 scenarios passed, 0 failed (82 steps green)

## Provider selection logic
1. Explicit: `ATELIER_AGENT_PROVIDER=bedrock`
2. Auto-detect: if `ATELIER_AGENT_MODEL` starts with `arn:aws:bedrock:`
3. Default: `anthropic` (direct API)

## Materialization
AWS credentials materialize with standard env var names:
```
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=wJal...
AWS_REGION=us-east-1
ATELIER_AGENTS_PROVIDER=bedrock
ATELIER_AGENTS_MODEL=arn:aws:bedrock:...
```
