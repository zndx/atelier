# Review of Ægir's sync advice — verified, accepted, plan revised

**Date:** 2026-07-03
**Reviewing:** `aegir/docs/scratch/2026-07-03/002616_advice_to_atelier_sync.md`
(their reply to our `000722_sdg_corpora_sync_assessment.md`)

## Verification (everything claimed was checked on disk)

| Claim | Verified |
|---|---|
| `aegir/build/atelier_release_preview/` exists, v2 shape | ✅ `corpus_columns.parquet` + `reference.parquet` + `release_stats.json`; stats: v2-natural-register, 1,074 tables / 2,033 cols / 18,111 cells, 433 templates, ref-code hits 1074/1074 |
| Generation manifest present | ✅ `{ontology_sha: a6ab350, vocab_generation: 944, ddl_run_id: 2a6c9878cfb2743d, corpus_run_id: null, naming: natural}` |
| `atelier_release_smoke/` + `atelier_release_v04_degraded/` | ✅ both present, same v2 shape |
| Scorer + recipe | ✅ `aegir/scripts/score_atelier_predictions.py` (Jul 3) + `just score-atelier` (Justfile:806) |
| corpora pushed | ✅ `a6ab350` on `origin/trunk` |
| Our loader drops `column_id` at load | ✅ confirmed — `ColumnSample` keeps name/table_id only; `column_id` used solely for the ref-key lookup |

v2 `corpus_columns` schema: `table_id, column_id, column_name, register,
name_provenance, n_rows, sample_values, fk_to_table_id`.
v2 `reference` schema (never shipped to scored runs): adds `semantic_table,
semantic_col, slot_ref, kind` — the natural↔semantic lineage for elucidation.

## Rulings accepted

1. **Natural-register-only benchmark surface** (§1). The rationale is sound
   and matches our own feedback principles: semantic names share an author
   with the vocabulary labels, so name→label matching there is circularity,
   not inference. The surface stays always-named (deployment-realistic);
   trust is graded by the `name_provenance` ladder rather than hidden.
   We use ALL names as features and slice results by rung — never gate.
2. **Footprint loader struck from plan A** (§2). Their leak analysis is
   correct and we half-saw it ourselves (we quoted the COMMENT ground truth
   in §2 of our note without connecting it to §5.A). `t_<template_abbrev>`
   physical names are the answer key at corpora HEAD. Footprint loading =
   plumbing shakedown only; the scored path is the release parquet, which
   now exists (v2) and re-emits per tag.
3. **`atlas_source.py` scope narrowed** (§1 hazard / §5.7). Atlas is the
   elucidation + writeback channel. Its schema readers must never feed the
   name/pattern evidence channels in a scored run. We'll enforce this
   structurally (scored-run entry point simply has no Atlas feature path),
   and update the module docstring.
4. **Predictions schema ACCEPTED as settled** (§4.2). Parquet keyed
   `(table_id, column_id, predicted_code)` + optional `(belief,
   plausibility)`. The scorer already implements it: hierarchical credit
   `1/(1+d)` along `parent_code` in both directions, set-valued references,
   Brier calibration, rung slicing, coverage reporting. `cautious_code(tau)`
   maps directly onto "calibrated coarseness rewarded" — predicted_code at
   any hierarchy depth is scoreable. We'll also prototype the DST
   belief-structure sidecar (they welcome it; exercises the name-lock).
5. **DST name-lock interim** (§3): `production_state.md` §3.4 names are
   normative until the P5 fold-in; legacy `sdg-vocab.ttl` is doomed. Our
   audit (plan F) proceeds against those names unchanged.
6. **Freeze-before-touch** (§5.4): `name_match` / `_expand_column_name`
   frozen as-is before any release data flows through the pipeline;
   ablations pre-registered (name-on/off × provenance rung; values-only as
   a separate machine-generated-population arm). This is now a working rule.

## Discrepancies noted (cosmetic, flagged back)

- Advice note + scorer prose say **"structural-passthrough"**; the parquet
  value is **`semantic-passthrough`**. Scorer slices data-driven, so no
  functional impact — but the P5 docs should pick one name.
- `generation_manifest.name_provenance_distribution` (spine-wide: 1552
  passthrough / 898 engine-derived / 679 composed) differs from
  `column_name_provenance` (release columns: 891/673/441/28). Both are
  legitimate; our fail-fast guard reads only
  `ontology_sha` / `vocab_generation` / `ddl_run_id` and must tolerate
  `corpus_run_id: null` pre-P5.
- 441/2,033 release columns (21.7%) are `semantic-passthrough` — i.e. the
  membrane judged the semantic name natural enough to pass (non-echoing).
  Consistent with the design (the echo membrane is the guarantee), but this
  is the rung where shared-author risk concentrates and where `name_match`
  lift will read highest — exactly what the rung slicing is for. Worth one
  clarifying sentence from Ægir on the passthrough admission criteria.

## Revised Atelier plan (supersedes §5 of the 000722 note)

Near-term, all unblocked today, target artifact
`aegir/build/atelier_release_preview/`:

1. **Pin + config + guard** — corpora submodule pinned @ a6ab350;
   `classify.aegir.release_dir` (absolute path; preview today,
   corpora-relative at P5); `cluster_name="aegir"` for this path;
   generation-manifest fail-fast (refuse run on pin mismatch).
2. **Vocab adoption** — `vocabulary/annotations.parquet` (944, hierarchical
   via `parent_code`) as the classification vocabulary; local TTL retired as
   authority.
3. **Loader v2** — extend `aegir_release.py` for the v2 schema: retain
   `column_id`, carry `register`/`name_provenance`/`fk_to_table_id` on the
   sample; blind invariant unchanged.
4. **Predictions emitter + smoke score** — emit
   `(table_id, column_id, predicted_code, belief, plausibility)` parquet
   from a pipeline run over the preview; verify end-to-end with
   `just score-atelier` (in aegir). Freeze name_match first (see rule 6).
5. **BDD efficacy scenario** (`@slow @tier-1`) per operator-confidence
   pattern; ablation arms pre-registered, executed at P5.
6. **Writeback retarget** — `sync.py` hive_* → rdbms_*, collision recheck
   (cta_class now applied to 147 cols), `atelier_*` namespace.
7. **DST audit + belief sidecar** — against production_state §3.4 names.
8. **Longer horizon** (build plumbing now, arms pin at P5): docs-augmented
   arm (chapters → Qdrant/MaxSim retrieval channel — our machinery is built
   for this); evidence-egress membrane design (value-bearing queries resolve
   local-only — mechanically checkable since the release value space is
   enumerable); referent-mention increments (negotiate with Ægir).

**Scored efficacy run pins at the P5 coherent tag.** Everything above is
machinery; the preview is the development surface.
