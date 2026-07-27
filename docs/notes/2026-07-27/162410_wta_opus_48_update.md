# Web Terminal Agent → Opus 4.8 (+ active-derivation bug fix)

**Date:** 2026-07-27
**Note:** "Opus 5" doesn't exist — the Claude 5 family is Fable 5 / Sonnet 5;
the Opus line's latest is **`claude-opus-4-8`** (same $5/$25, 1M ctx, no new
breaking changes vs 4.7). Per CLAUDE.md convention, `agents.model` tracks
the latest direct-API Opus → 4.7 → 4.8.

## Changed

- `config/base.conf` — `agents.model = "claude-opus-4-8"` (+ resolver comment).
- `src/atelier/config.py` — dataclass default + docstring.
- `src/atelier/agents/client.py` — fallback model.
- `src/atelier/enrichment/{model_resolver,backend_client}.py` — apex mapping.
- `src/atelier/terminal_catalog.py` — **added `anthropic-opus-4-8`** as the
  latest entry; kept `anthropic-opus-4-7` (active API-side, pinnable, notes
  demoted). Bedrock entry untouched (`@attr:agent_model`).
- CLAUDE.md Model Defaults, docs (agents.md, secrets.md, grpc.md), test
  assertions in `tests/enrichment/test_classify_backed_generator.py`.

## Bug found + fixed while verifying

`derive_from_agent_model` returned the FIRST ref-match from the full
catalog — and the Bedrock entry's `@attr:agent_model` ref equals
`cfg.agent_model` **by construction**, so on direct-API deploys (no AWS
creds) the UNAVAILABLE Bedrock entry claimed `active`. Fix: derivation
skips unavailable entries. CAI/Bedrock semantics preserved (their entry is
available + ref-matching). New `tests/test_terminal_catalog.py` (3 tests)
pins both deploy shapes + catalog ordering. Suite: **158 passed**.

## Live verification

Gateway restarted via `devenv processes restart gateway` (NB: one restart
silently no-oped — verify by pid, not exit code). `/api/terminal/models`:
Opus 4.8 available + active, 4.7 available/pinnable, Bedrock unavailable
(no local creds), `override_set: false`.

## Blocker for actual use

`ANTHROPIC_API_KEY` still returns **401 invalid** (model_discovery cached
error) — the terminal agent will fail on first message regardless of model
until the key is refreshed. Config side is ready.
