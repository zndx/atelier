# SDG depth gap map — what "1.0" doesn't say, and what `just optimize` must cover

**Date:** 2026-07-03
**Trigger:** RH: the golden 1.0 risks being oversold; the full breadth of the
platform (`just optimize` et al.) has not been updated to evaluate the depth
of sdg-corpora. Correct — this note maps the gap channel by channel.

## Calibration of yesterday's number

The golden smoke's 1.0 is `predictions == answer key` by construction. It
verified: pin guard → blind v2 loader → predictions schema → (table_id,
column_id) join → Ægir's scorer → rung slicing. It verified **zero**
classification capability. No pipeline ran; no evidence source fired.

## Evidence-stack readiness vs SDG-944 (verified in code)

| Channel | State | Evidence |
|---|---|---|
| `name_match` | READY (and FROZEN per pre-registration rule) — driven by the loaded category_set's labels/abbrevs | vocabulary injection is enough |
| `pattern` | **DEAD for SDG** — `DEFAULT_PATTERN_MAP` (`mass_functions.py:666`) maps to legacy `ICE.SENSITIVE.*` codes; lookups miss against SDG codes, so the channel contributes nothing | needs an SDG remap (or run with it measured-dead) |
| `maxsim` | **MISSING** — per-taxonomy Qdrant collection resolved from DB (`taxonomy_collections`, `status='current'` per taxonomy_id; `maxsim_bridge._resolve_qdrant_collection`); none exists for an SDG taxonomy → runs "degraded_no_collection" | `just optimize maxsim` (semantic_optimize.py) against the SDG vocab under a new taxonomy_id |
| `llm` | READY — prompts are vocabulary-driven | none |
| `catboost` | READY — `fit_to_llm=true` trains in-run (invariant) | none |
| `svm` | **FAIL-FAST MISSING** — `_ensure_registered_svm_head` requires a `status='current'` registry row per (taxonomy_id, encoder); pipeline error literally says "run `just optimize` to train and promote one" | `just optimize svm` dual-gate pipeline (coverage audit → generator authorship → synth corpus → train/eval → uplift gate vs maxsim) |
| synth generators | UNKNOWN COVERAGE vs 944 — `GeneratorRegistry.coverage_report(category_set)` will quantify; 316+ existing generators were authored for PII/meta-tagging vocabularies, not SDG.DOM/GENERIC | run the coverage audit first; `evolve-generators` for gaps |
| taxonomy identity | `classify.taxonomy_id="default"` — SDG must be isolated (e.g. `sdg-a6ab350`) so heads/collections can't collide with production | config + optimize runs keyed to it |
| embedding-atlas projection | STALE — golden baseline exists only for release v0_3; `build_baseline_projection.py` defaults to the /raid v0_3 dir and its `--verify-atlas` checks the 540-term glossary (stale until P5 regen) | re-aim at the preview; defer Atlas verify |
| `just optimize agent` | N/A for this path — Hive-oriented reference curation | later, if ever |

## The contamination boundary (needs pre-registration, like the name freeze)

Training inputs for the SDG stack must come from **vocabulary metadata
only** (labels, descriptions, `example_values` hints → synth generators).
Off-limits as training signal:
- release `corpus_columns.sample_values` (that IS the eval),
- corpora `ddl/*/base_rows.parquet` values (same generative process =
  near-duplicate eval values),
- DDL `COMMENT` JSON (`template_id`/`bfo_anchor` — literally the key).

Same logic as the name_match freeze: measured inference, not leakage.
Propose this line to Ægir alongside the ablation pre-registration.

## Proposed sequence (the real ⑤, before any scored claim)

1. `classify.taxonomy_id` isolation for SDG + vocab injection wiring
   (pipeline entry that loads `load_sdg_vocabulary` when the aegir source
   is selected).
2. Generator coverage audit vs the 944 (cheap, quantifies the SVM lift
   needed): `GeneratorRegistry.coverage_report`.
3. `just optimize maxsim` — build + promote the SDG Qdrant collection.
4. `just optimize svm` — synth-from-vocab corpus, train NHSVM head,
   promote under the SDG taxonomy_id (Gate A/B as designed).
5. Pattern map: SDG remap for the anchored patterns that have SDG
   equivalents (email/uuid/url map naturally into SDG.GENERIC.*), or
   explicitly accept + record the dead channel for the first run.
6. Blind pipeline run over the preview → `--from-run` emission →
   `just score-atelier`. THAT number is the first honest one, and it
   pins at P5 for the scored claim.
7. Rebuild the golden baseline projection against the preview for the
   embedding-atlas visual check (Atlas verify deferred to post-P5).

Estimated shape: 1–2 is hours; 3–4 is the long pole (generator authorship
for uncovered SDG codes; GPU training is available — 6× 4090 per
preflight); 5–7 are small.
