# Structural-baseline embedding-atlas projection from the Ægir release

Date: 2026-06-30 (follow-on to `142608_aegir_atlas_integration_phase0.md`)

## Goal

A **golden-path baseline with no pipeline run**: a `build/results/<run_id>/`
that looks exactly like a real classification run, but with every prediction
set to its KNOWN reference code. Purpose — visually confirm the plumbing is
wired correctly (Ægir DDL corpus → Atelier results artifact → embedding-atlas
UI → Atlas correspondence) *before* standing up the classification stack, and
give a fixed artifact the real run can be diffed against later.

## Why it's well-grounded (verified against live Atlas)

The curated correspondence already exists in the release's `reference.parquet`
and is 1:1:1: **146 `source_table` ↔ 146 `SDG.*` codes ↔ 146 templates**, every
`source_table` mapping to exactly one code (0 tables with >1 code), all slots
`Class`. Each of the 146 templates resolves to a live Atlas glossary term
(**146/146**) and an `rdbms_table` footprint entity (`footprint.<source_table>
@aegir`, spot-checked present). So the baseline isn't invented — it's the
release's own answer key, projected into the results schema.

## What was built

- **`scripts/build_baseline_projection.py`** — reads the release
  (`corpus_columns.parquet` + `reference.parquet`), sets every column's
  `predicted_code = reference_code` at full belief
  (`belief=plausibility=confidence=1.0`, `conflict=uncertainty=0`,
  `matches_reference=True`), and writes a complete run dir:
  - `atelier_embeddings.parquet` — the embedding-atlas artifact
  - `classifications.json`, `settings_snapshot.json` (`source_id=
    aegir-baseline/footprint`) → `_is_complete_run` True, so the gateway
    auto-sync registers it as a run in the UI
  - `atlas_correspondence.csv` — one row per source_table linking
    `reference_code` → `footprint.<source_table>@aegir` (Atlas table qn) →
    glossary term, so the Atlas associations are explicit and checkable.
  - `--per-template N` caps instance tables per code (smaller visual);
    `--atlas-url` enriches `evidence` from the live glossary definitions;
    `--verify-atlas` round-trips a sample of footprint tables.
- **`pipeline._write_parquet`** gained `precomputed_xy=` — the baseline injects
  a deterministic **code-clustered** layout (golden-angle spiral centers +
  per-column hash jitter). Necessary because a uniform-belief golden run would
  collapse the PCA fallback to a point; also handy for deterministic tests.
  Default `None` preserves the existing UMAP/PCA path exactly.
- Tests: `tests/classify/test_baseline_projection.py` (golden invariants,
  clustering, schema parity via a tiny synthetic release). 13 new tests pass.

## Verified output (sample run, `--per-template 4 --atlas-url … --verify-atlas`)

- 1024 columns / 584 tables / **146 distinct codes**; 540 glossary labels read.
- Schema parity: **all 35 pipeline fields present, none missing** (uses the
  pipeline's own writer → cannot drift from the real artifact).
- Golden invariants hold on every row; hover text e.g.
  `"Hipaa Safeguard Admin - HIPAA_SAFEGUARD_ADMIN"`; evidence pulled from the
  glossary definition.
- Layout: mean within-cluster distance 0.63 vs between-cluster 113.5 (**180×**)
  — clean, separable clusters for at-a-glance verification.
- Atlas verify: **10/10** sampled footprint tables present.

Artifact left on disk (gitignored): `build/results/aegir-baseline-sample/`.

## How to use it

1. Point the embedding-atlas viewer / Atelier Embeddings page at
   `build/results/aegir-baseline-sample/atelier_embeddings.parquet`. Expect 146
   clean clusters, colored by `predicted_label`, hover showing the SDG label.
2. Cross-check `atlas_correspondence.csv` against the Atlas UI — each cluster's
   `reference_code` ↔ its `footprint.<table>@aegir` entity + glossary term.
3. When the stack is up, run the real pipeline over the same release and diff
   its `atelier_embeddings.parquet` against this baseline — same keys/schema, so
   any divergence in `predicted_code` / `matches_reference` is pipeline signal.

## Notes / next

- Full baseline (all 16516 columns) is the same command with `--per-template 0`
  (default); the 1024-col sample is just the lighter visual.
- Atlas writeback (apply baseline codes as `atelier_*` tags on the
  `rdbms_column` entities) is the natural complement — needs the `sync.py`
  rdbms_* parameterization noted in the Phase-0 note. Then both the baseline
  and the real prediction sit on the same Atlas entity.
- Baseline rows key on the release's (obfuscated) `tbl_NNN`/`x` names — the same
  keys the pipeline ingests — so the diff is apples-to-apples. The *real* names
  live one hop away via `atlas_correspondence.csv` → footprint entity.
