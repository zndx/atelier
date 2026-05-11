<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# SDK Smoke Test — Full Stack Wiring

## What was done

Completed the Claude Agent SDK smoke test integration (plan steps 2-7):

### New files
- `src/atelier/agents/__init__.py` — module init, re-exports `validate_api_key`, `run_smoke_test`
- `src/atelier/agents/client.py` — SDK client wrapper:
  - `validate_api_key(cfg)` — Anthropic API key validation via minimal `messages.create()`
  - `run_smoke_test(cfg)` — Claude Agent SDK `query()` with `max_turns=1`, proves full pipeline
- `features/agent/agent_smoke.feature` — 5 BDD scenarios (3 tier-0, 2 tier-1)
- `features/agent/step_defs/agent_steps.py` — step definitions with keystone agent seeding

### Modified files
- `src/atelier/gateway.py` — added `POST /api/agents/validate-key` and `POST /api/agents/smoke-test`
- `src/atelier/db/dao.py` — added `list_agents()`, `get_agent()`, `upsert_agent()`
- `src/atelier/service.py` — wired `ListAgents`, `GetAgent` RPCs to DAO
- `bin/start-app.sh` — seeds 3 keystone agents on startup
- `features/steps/__init__.py` — re-exports agent step definitions

### Keystone agents seeded
1. **classifier** — Zero-shot column classification via LLM reasoning
2. **evidence-fuser** — Dempster-Shafer fusion of LLM + embedding + SVM evidence
3. **viz-director** — Embedding projection curation and interactive exploration

## BDD results
- 27 scenarios passed, 0 failed (78 steps)
- All tier-0 agent scenarios pass including import checks and agent seeding
- Tier-1 scenarios (API key validation, SDK smoke test) require `ANTHROPIC_API_KEY` in `.env`

## Next steps
- Set `ANTHROPIC_API_KEY` in `.env` and run tier-1 BDD to validate key
- `devenv up` + `curl -X POST localhost:8090/api/agents/validate-key` for REST endpoint test
- SDK smoke test requires Claude Code CLI as execution backend
