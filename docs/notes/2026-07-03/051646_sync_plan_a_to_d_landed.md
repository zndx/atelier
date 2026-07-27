# sdg-corpora sync ①–④ landed — golden smoke scored 1.0 by Ægir's scorer

**Date:** 2026-07-03
**Follows:** `000722_sdg_corpora_sync_assessment.md` + `045947_aegir_advice_review.md`
(plan approved; items ①–④ of the revised plan implemented this session).

## What landed

### ① Pin + config + guard
- **`external/sdg-corpora` submodule** added, pinned at `a6ab350` (the
  founding artifact), cloned `--reference` the local Ægir checkout so
  objects are borrowed, origin is `git@github.com:zndx/sdg-corpora.git`.
- **HOCON** (`config/base.conf` → `config.py`): `classify.aegir.corpora_dir`
  (default `external/sdg-corpora`), `classify.aegir.release_dir` (empty =
  fail-fast when invoked; `ATELIER_AEGIR_RELEASE_DIR` env), and
  `classify.aegir.cluster_name` (default `aegir`) — scoped so
  `governance.cluster_name` stays `cm` for CAI. Verified round-trip via
  `load_config()`; `just preflight` passes.
- **Generation-manifest guard** (`check_release_pin` in
  `classify/aegir_release.py`): refuses pre-v2 releases (no
  `generation_manifest`), `ontology_sha` mismatches against the pinned
  corpora HEAD (git-resolved, prefix-tolerant, `expected_ontology_sha`
  override for gitless environments), and `vocab_generation` drift against
  the actual `annotations.parquet` row count. `corpus_run_id: null`
  tolerated pre-P5; `ddl_run_id` logged for provenance.
  `ReleasePinError` on all refusals.

### ② Vocabulary adoption
- **`load_sdg_vocabulary(corpora_dir)`** reads
  `vocabulary/annotations.parquet` through the same shared builder
  (`_build_category_set_from_records`) every other vocabulary source uses —
  identical `embedding_text` / tree semantics. `example_values` (the
  domain_hypernym column-name hints) feed `embedding_text` via the
  `specifics` channel. Explicit `parent_code` wins over dot-derivation
  (cross-branch parents like `SDG.DOM.* → SDG.ICE` are preserved).
- Verified against the pin: **944 codes, 4 forest roots** (`SDG.GENERIC`,
  `SDG.ICE`, `SDG.INDEPENDENT_CONTINUANT`, `SDG.PROCESS`).

### ③ Loader v2
- `ColumnSample` gains `column_id` (the predictions join key), `register`,
  `name_provenance`, `fk_to_table_id` — populated by the release loader,
  `None` elsewhere; `to_dict()` emits them only when set so non-Ægir
  artifacts keep their shape. Rung fields are for **slicing results,
  never gating features**.
- `load_aegir_release_samples` handles v2 and v0.3 identically (v0.3 rows
  simply lack the register columns). Blind invariant unchanged and tested.
- Module docstring rewritten: the "obfuscated names" framing was
  v0.3-speak — v2 is an always-named natural-register surface with the
  provenance ladder grading trust.

### ④ Predictions emitter + smoke
- **`scripts/emit_aegir_predictions.py`** writes the settled handoff
  schema `(table_id, column_id, predicted_code[, belief, plausibility])`.
  Modes: `--golden` (predictions == reference at full belief — plumbing
  check only, stamped as such) and `--from-run RUN_ID` (joins
  `classifications.json` back to release ids via `corpus_columns`,
  counting unjoinable rows). Pin guard runs first; `--allow-unpinned`
  downgrades to a loud warning for legacy-emission plumbing work.
  Set-valued references (`|`, P5+) collapse to first member in golden mode.
- **End-to-end smoke against the real preview**
  (`aegir/build/atelier_release_preview`, 1,074 tables / 2,033 cols):

  ```
  pin OK: ontology_sha=a6ab350 vocab_generation=944
  golden predictions: 2033 rows
  → just score-atelier (aegir):
     overall: leaf_accuracy 1.0, hierarchical_score 1.0, miss_rate 0.0,
              brier_calibration 0.0, coverage 1.0, n_unkeyed 0
     by_name_provenance: all four rungs (engine-derived 891, composed 673,
              semantic-passthrough 441, degraded-mechanical 28) at 1.0
  ```

  The loop Ægir opened is closed: pin guard → blind loader → predictions
  parquet → their scorer, keyed correctly, sliced by rung.

## Tests
27 new/updated tests in `tests/classify/test_aegir_release.py` (v2
provenance carry, v0.3 back-compat, vocab hierarchy + fail-fast, five pin
guard refusal/acceptance cases) and
`tests/classify/test_emit_aegir_predictions.py` (golden mirror, set-valued
collapse, run→id join with unjoined counting, parquet schema, empty
refusal). Full `tests/classify` + `tests/governance`: **116 passed**.

## Deliberately not done (pins at P5 / next steps)
- Real blind pipeline run + `--from-run` emission (freeze
  `name_match`/`_expand_column_name` first — pre-registration rule).
- BDD `@slow @tier-1` efficacy scenario (plan ⑤).
- Writeback retarget + collision recheck (⑥), DST audit + belief sidecar
  (⑦), docs-augmented arm / egress membrane / referent-mention (⑧).
- Plan G docs sweep (CLAUDE.md got the submodule line only).

## Working-tree state
Staged: `.gitmodules` + `external/sdg-corpora` (submodule add). Modified:
`config/base.conf`, `src/atelier/config.py`, `src/atelier/classify/sampler.py`,
`CLAUDE.md` (+ the pre-existing 06-30 WIP diffs). New:
`scripts/emit_aegir_predictions.py`, extended `aegir_release.py`, tests.
Nothing committed — awaiting review/commit decision.
