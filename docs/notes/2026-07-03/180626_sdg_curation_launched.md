# SDG agent-mediated curation: stack landed, full run in flight

**Date:** 2026-07-03 (tasks 8 + 9 kickoff per RH; MiNiFi deploy side moved
to the Ægir session)

## Task 8 — the `just agent` SDG stack (LANDED, run in progress)

Three scripts + tests (5 hermetic tests green; total suite ~150):

- **`scripts/build_agent_mediated_sdg.py`** — single-ingress working-set
  builder. Reads EXACTLY two files (blind `corpus_columns.parquet` +
  `annotations.parquet`), sha256s them into the metadata, runs the pin
  guard, REFUSES legacy layouts where the key sits in the release dir
  (`--allow-legacy-layout` for dev fixtures). Carries per-column
  register/provenance and per-code hints (the `domain_hypernym`
  example_values — designed-for-purpose name→code metadata).
- **`scripts/run_curate_local.py`** — the curation loop on the `referee`
  capability (Nemotron via our engine). Two stages per table: shortlist
  (whole table vs the 944-code index w/ hints) → decide (per column, full
  candidate metadata + sibling context), both schema-enforced. Harness
  owns all determinism: code-exists validation with one feedback retry,
  identifier-name candidate augmentation (`_id/_ref/_key` → identifier
  codes appended), parent codes prompted as first-class (matches Ægir's
  calibrated-coarseness scoring). Resume-safe per table; table-level
  parallelism (4 workers ≈ referee's seq slots). Audit trail retains
  reasoning heads, candidates, latencies, provenance rung per decision.
- **`scripts/audit_blind_integrity.py`** — post-hoc mechanical audit:
  ingress hashes unchanged + forbidden-marker scan (reference.parquet,
  semantic_col/table, naming_map, bfo_anchor, template_id, slot_ref)
  across working set / decisions / audit trail.

**Live smoke findings (fixed in-loop):**
1. Constrained decoding flooded whitespace inside the unbounded `rationale`
   string until max_tokens truncated the JSON mid-structure. Fix:
   grammar-BOUNDED strings (maxLength on every string, bounds on numbers).
   Posted to Ægir (§12) — applies to any guided output on this stack.
2. Bare `code: label` shortlists mis-binned domain columns
   (`indicator_id → ANOMALY` at 0.92); adding the vocabulary's own
   column-name hints fixed the observed cases (`policy_id → POLICY` 0.96,
   `program_id → PROGRAM_TYPE` 1.0).
3. Identifier/FK columns still wobbled → deterministic candidate
   augmentation (skill principle 3: harness > model).

**Full run**: 1,074 tables / 2,033 columns, workers=4, ~3.5h ETA,
background task; GPUs 0–3 leased via the shared dir (noted to Ægir §12).
On completion: blind-integrity audit, then the reference seeds
`just optimize maxsim` + `svm` (Arm T stage 3).

**Referee-quality discipline note:** decisions can later be scored against
the preview key as a DIAGNOSTIC (Ægir invited preview scoring), but
iterating referee prompts against that score would Goodhart the referee —
one read, recorded, structural changes only, declared. Same posture as the
name_match freeze.

## Task 9 — MiNiFi status

- Increment ① (event stream) landed earlier (see 173513 design note).
- Increment ② (build from fork): `py_bootstrap.sh --noninteractive` asserts
  in cmake-cache creation on this nix-hybrid host — deferred to the Ægir
  session's MiNiFi effort (they own the deploy side now; suggested the Zarf
  package ship a prebuilt host binary).

## Ops state

- Atelier engine up (:50251), referee resident on GPUs 0–3 (lease held).
- Curation run: background task `b34ek8hqe`; artifacts under
  `build/data/agent_mediated/sdg/`; resume-safe (`review_state.json`).
