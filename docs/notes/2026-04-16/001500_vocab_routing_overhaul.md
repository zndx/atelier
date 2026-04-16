# Vocabulary Routing & LLM Robustness Overhaul

## Summary

Addressed the ~50% accuracy regression on CAI when classifying with a
customer's 290-code domain vocabulary. Three changes:

### 1. Vocabulary routing fix

**Root cause**: `compose_vocabularies()` conflated mapping knowledge
(is_a relationships for future portable model) with classification scope
(DST frame construction). The LLM was being asked to classify into a
composed+scoped hybrid vocabulary.

**Fix**: For in-situ Hive classification, the domain vocabulary IS the
classification target. `_load_vocabulary()` now returns domain annotations
directly — no composition, no scoping. Added `vocab_uri` field to
DataSource to decouple annotations table location from data tables.

**Removed**: `with_tagging_scope()` from `taxonomy.py` (was a band-aid).

**Kept**: `_coerce_to_singleton()` (useful for any vocab) and
`build_system_prompt(category_set=)` (correct for domain vocab).

### 2. LLM truncation handling

**Root cause**: With 290 categories, the system prompt is ~48KB. Combined
with 50-column batches, responses could be truncated. `LLMResponse.truncated`
existed but was never checked by callers.

**Fix**: 
- `_classify_batch_with_retry()`: Detects truncation, halves batch, retries
  recursively until all columns classified
- `_estimate_safe_batch_size()`: Preemptively reduces batch for large vocabs
  (290 cats → 41 columns/call instead of 50)
- `LLMResponse.truncated` now covers both `"length"` (OpenAI) and
  `"max_tokens"` (Bedrock)
- `BootstrapState` tracks `truncation_count` and `effective_batch_size`
- Agent's `check_convergence` tool exposes truncation metrics

### 3. BDD scenarios

8 new tier-0 scenarios (no @slow):
- `vocabulary_routing.feature` (5): OOTB ICE, domain-direct, requires-uri,
  hierarchical belief-path, adaptive batch sizing
- `llm_robustness.feature` (3): truncation retry, metrics tracking,
  finish_reason coverage

**Total: 149 scenarios across 34 features** (was 141/32).

## Files changed

- `src/atelier/classify/pipeline.py` — `_load_vocabulary()` rewritten, `vocab_uri` param
- `src/atelier/classify/taxonomy.py` — `with_tagging_scope()` deleted
- `src/atelier/classify/bootstrap.py` — truncation retry, adaptive batch, metrics
- `src/atelier/classify/llm_backend.py` — `truncated` covers Bedrock `max_tokens`
- `src/atelier/classify/agent_loop.py` — truncation metrics in `check_convergence`
- `src/atelier/db/model.py` — `vocab_uri` field on DataSource
- `src/atelier/db/dao.py` — `vocab_uri` in create + dict helper
- `src/atelier/gateway.py` — resolve `vocab_uri` from source record
- `db/migrations/20260416100000_vocab_uri.sql` — new column
- `features/agent/vocabulary_routing.feature` — 5 new scenarios
- `features/agent/llm_robustness.feature` — 3 new scenarios
- `features/agent/step_defs/vocab_routing_steps.py` — new
- `features/agent/step_defs/llm_robustness_steps.py` — new
- `docs/src/architecture/data-sources.md` — updated routing + robustness
- `docs/src/scenarios/overview.md` — count update (149/34)
