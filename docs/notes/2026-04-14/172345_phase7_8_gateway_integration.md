<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Phases 7-8 Complete: Gateway Integration Testing + User Journey Validation

## Summary

Established integration test coverage at the HTTP gateway level and the
experimentation user journey. BDD scenario count: 73 → 82 (+9 new), 0 failures.

## Phase 7.1: Reusable HTTP Step Library

### New Files
- `features/gateway/step_defs/http_steps.py` — Generic GET/POST steps for tier-1 tests
- Reusable assertions: status code, JSON key presence, list length, min value

### Modified Files
- `features/environment.py` — Added `_check_gateway()` to `_ensure_stack_healthy()`
  (verifies `/api/health` in addition to PostgreSQL + Qdrant)

## Phase 7.2: Gateway API Endpoint Validation

### New Files
- `features/gateway/api_endpoints.feature` — 8 tier-1 scenarios
- `features/gateway/step_defs/endpoint_steps.py` — Domain assertions (agents, skills, FSM states)

### Scenarios
| Scenario | Assertions |
|----------|------------|
| Agents list | 5+ agents, each with id/name/role, roles include sampler+classifier |
| Skills | Non-empty list, each with id/title/content |
| Data sources | Non-empty, source with "sample" display_name |
| Vocabulary stats (sample) | terms >= 300 |
| FSM status | Valid FSM state |
| Health endpoint | Contains "status" |
| Vocabulary stats (universal) | terms >= 25 |
| Status endpoint | Contains grpc/postgres/qdrant/connected |

## Phase 7.3: Pipeline Integration Test

### New Files
- `features/gateway/pipeline_integration.feature` — 2 tier-1 @slow scenarios
- `features/gateway/step_defs/pipeline_steps.py` — FSM polling with proof-of-progress timeout

### Scenarios
- OOTB sample classification → CONVERGED → dataset with 100+ rows
- Pipeline run appears in FSM history with CONVERGED state

## Phase 7.4: SPA Route Validation

### New Files
- `features/gateway/spa_routes.feature` — 4 tier-1 scenarios (Scenario Outline)
- Tests /, /agents, /workflows, /embeddings all return 200 with text/html

## Phase 7.5: User Journey — Experimentation Phase Transition

### New Files
- `src/atelier/classify/fixtures/mock_user_taxonomy.json` — 24 entries (2 internal + 22 leaves)
  - Fictional domain categories under ICE.SENSITIVE.PID.DOMAIN and ICE.NONSENSITIVE.DOMAIN
  - Safe for git — entirely fictional codes/labels
- `features/agent/experimentation.feature` — 3 tier-0 scenarios (2 fast, 1 @slow)
- `features/agent/step_defs/experimentation_steps.py`

### Taxonomy Source Override
- **CI/default**: Mock taxonomy (16 universal + 20 domain = 35 composed leaves)
- **Developer workstation**: Real meta-tagging (146/296 codes mapped into 316-leaf vocabulary)
- **Override env var**: `ATELIER_REAL_DATA_DIR=~/local/tmp/meta-tagging`

### Scenarios
| Scenario | Tier | Key Assertions |
|----------|------|----------------|
| Custom vocabulary composes with universal base | tier-0 | More leaves, ICE root reachable |
| Synth generators cover custom categories | tier-0 | >= 80% generator coverage |
| Pipeline with custom taxonomy (mock LLM) | tier-0 @slow | CONVERGED, codes in vocab, belief paths to root |

## Phase 8.1: TestClient Tier-0 Coverage

### New Files
- `features/gateway/api_testclient.feature` — 7 tier-0 scenarios
- `features/gateway/step_defs/testclient_steps.py` — TestClient GET + JSON assertions

### Scenarios
Validate that all gateway endpoints return valid JSON (even when gRPC is down):
skills (with key checks), vocabulary/stats, FSM status, agents, data-sources, health, datasets.

## Final Counts

- Tier-0 scenarios: 82 (was 73, +7 TestClient + 2 experimentation)
- Tier-1 scenarios: 14 new (8 API endpoints + 2 pipeline + 4 SPA routes)
- Total new scenarios: 23 (9 tier-0 + 14 tier-1)
- Gateway endpoint coverage: 8/23 endpoints tested (was 1/23)
- User journey: orientation → understanding → experimentation (was: none)

## Files Modified/Created

| File | Change |
|------|--------|
| `features/environment.py` | Added gateway health check |
| `features/steps/__init__.py` | Added 5 new step module re-exports |
| `features/gateway/step_defs/http_steps.py` | NEW |
| `features/gateway/step_defs/endpoint_steps.py` | NEW |
| `features/gateway/step_defs/pipeline_steps.py` | NEW |
| `features/gateway/step_defs/testclient_steps.py` | NEW |
| `features/gateway/api_endpoints.feature` | NEW |
| `features/gateway/pipeline_integration.feature` | NEW |
| `features/gateway/spa_routes.feature` | NEW |
| `features/gateway/api_testclient.feature` | NEW |
| `features/agent/experimentation.feature` | NEW |
| `features/agent/step_defs/experimentation_steps.py` | NEW |
| `src/atelier/classify/fixtures/mock_user_taxonomy.json` | NEW |
