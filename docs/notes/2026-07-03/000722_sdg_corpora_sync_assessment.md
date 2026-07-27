# sdg-corpora sync assessment — bringing Atelier in line with Ægir's relational corpus

**Date:** 2026-07-03 (session start 2026-07-02 evening)
**Trigger:** Ægir landed a fresh round of sdg-corpora (submodule at
`~/local/src/zndx/aegir/corpora`, remote `git@github.com:zndx/sdg-corpora.git`,
HEAD `a6ab350` "the FOUNDING ARTIFACT — signals.zndx.org, 828 classes + 7,956
certified individuals"). This note reconciles Atelier's 2026-06-30 integration
WIP against the current Ægir direction and lays out the sync work.

Sources: three parallel deep-reads — (1) sdg-corpora tree, (2) Ægir direction
docs (`atelier-evals.txt`, `docs/current/src/signals_programme.md`,
`end_to_end_and_meta_harness.md`, `phase_gate_governance_ddl.md`,
`ontology/production_state.md` §3.4, `docs/scratch/2026-07-02/*`), (3) Atelier's
own uncommitted 2026-06-30 work.

---

## 1. The contract, stated plainly

From `corpora/README.md` ("Releases (for Atelier's git data source)") and
`aegir/scripts/build_atelier_release.py` (S6 RELEASE stage of Ægir's DAG):

> A tagged release packages `vocabulary/annotations.parquet` + `ontology/` +
> the populated DDL tables — **without** the per-column reference codes.
> Atelier pins the release and classifies columns into the SKOS vocabulary
> **blind** (values + vocab only); the reference (column→code, deterministic
> from the spine) is held back as the scoring key → independent, pre-training
> efficacy feedback on the corpus, and a clean measure of Ægir's downstream
> lift over it.

Atelier's role in the Signals programme component map: **"independent
pre-training efficacy gate (blind classification, reference withheld)"**.
Scoring is Ægir-side (`scripts/score_realization_cpa.py` shares the same
held-out key with their reasoner-CPA path) — Atelier emits predictions; Ægir
scores both Atelier and its own realize-as-CPA against one answer key.

## 2. What sdg-corpora actually contains at HEAD

- `vocabulary/annotations.{csv,parquet}` — **944 rows**, literally the Atelier
  ReferenceCategory shape: `code, label, abbrev, notation, parent_code,
  taxonomy, description, common_names, example_values`. Roots: SDG.DOM (503),
  SDG.GENERIC (361), SDG.ICE (76), + BFO/CCO anchors. 83 `domain_hypernym`
  rows carry pipe-separated column-name hints in `example_values`.
  `vocabulary.ttl` is the SKOS twin (944 concepts, scheme
  `https://signals.zndx.org/sdg/scheme`).
- `ontology/sdg-ontology.{owl,omn}` — the founding artifact: 828 classes,
  7,956 HermiT-certified individuals (**values are class-typed OWL
  individuals**, membrane-admitted, 92 clashing withheld), namespace
  `https://signals.zndx.org/sdg#`, BFO 2020 + CCO grounding,
  `HERMIT_CERTIFICATE.md` as checkable verdict. NB `ontology/sdg-vocab.ttl`
  inside corpora is legacy (old `signals360.example.org` namespace).
- `ddl/b4bbc02f7800b0ea/` — the relational footprint as parquet:
  `ddl_statements` (CREATE TABLEs w/ ground truth in SQL `COMMENT` JSON:
  `template_id`, `bfo_anchor`, `family`), `base_table_index`, **`base_rows`**
  (materialized cells: `table_name, row_ix, col_name, value, is_pk, is_fk,
  fk_target_table`), `cross_family_fks` (351), `views` (974),
  per-dialect validation (Trino ∩ Spark). 623 tables.
- `corpus/collections/` — 121 topic collections, 1,977 chapters with RI-true
  tables woven inline, 4,116 `t_*.sql` (623 unique; shared tables duplicated
  per collection). `manifest.json` per collection is the join key.

**⚠ Version skew at HEAD:** ontology+vocabulary were regenerated 2026-07-02
(828/944); ddl+corpus still reflect the 623-template v0.4 run (2026-06-28).
Only git tag is `seed-baseline-v0.3`. The v0.4 standalone blind
`columns.parquet` benchmark is "being rebuilt" (absent — the v0.3 extractor's
JSON block is no longer emitted). A naive "sync everything at HEAD" mismatches
label space against table footprint.

**⚠ Timing:** corpus scale-up is **HELD** Ægir-side pending the "Convert 1"
authenticity rework (`aegir/docs/scratch/2026-07-02/160034_convert_priority_
authentic_cas.md`) — the next release's relational structure (FKs, value
pools) will differ. Build machinery now; pin the efficacy run after Convert 1
lands and Ægir cuts a coherent tag.

## 3. Where Atelier's 2026-06-30 WIP stands against this

The WIP (uncommitted on trunk) is solid plumbing but encodes a premise that
has moved:

| 2026-06-30 premise | Current reality |
|---|---|
| Source of truth = `/raid/checkpoints/aegir-artifacts/atelier_release_v0_3/` checkpoint | Source of truth = **pinned sdg-corpora tag** (git submodule); /raid checkpoint is a stale v0.3 emission (146/540 templates) |
| Vocabulary = live Atlas glossary, 540 terms | Vocabulary = `annotations.parquet`, **944 codes**, content-derived and moving; glossary is a projection, not the source |
| `cta_*` typedefs registered but NOT applied → no scoreboard collision | **`cta_class` now applied to 147 columns** + `domain_foundation` on tables (DDL-native projector, gate PASS) — collision check must be redone |
| Local `sdg:` prefix / atelier-vocab.ttl relevant | Namespace stabilized to `https://signals.zndx.org/sdg#`; local TTL is a pre-migration snapshot, not authoritative |
| Blind columns from `corpus_columns.parquet` | At corpora HEAD, blind columns reconstruct from `ddl/<run>/base_rows.parquet`; release parquet returns with the next `build_atelier_release.py` emission |

What survives intact:
- `classify/aegir_release.py` — release_dir is caller-supplied; the
  `{corpus_columns, reference, release_stats}` contract is exactly what
  `build_atelier_release.py` emits. Keep as-is for release-parquet loading.
- `governance/atlas_source.py` + `AtlasClient.get_glossary_terms` — Atlas
  stays useful as the *names/FK-graph* channel and writeback target.
- `scripts/build_baseline_projection.py` + `precomputed_xy` — golden-path
  plumbing verification, still valid (against whichever release is pinned).
- All tests (hermetic).

Still-open gaps from the 06-30 session, now confirmed against direction:
- Nothing wired into pipeline/gateway/justfile (modules are dangling).
- `governance/sync.py` writeback still hardcodes `hive_table`/`hive_column`.
- Config: no release-dir key; `governance_cluster_name` defaults `"cm"` while
  all new code assumes `"aegir"`.

## 4. New hard requirements from Ægir

1. **DST name-lock:** Ægir's state-fusion layer consumes Atelier belief
   structures "using these names directly and without translation" —
   `sdg:MassFunction / sdg:BeliefInterval / sdg:Evidence / sdg:Claim`
   (`production_state.md` §3.4). Audit our emitted belief structures.
2. **Natural-name register:** corpus columns carry the "natural" DBA-register
   name variant (canonical deliverable, breaks concept-from-header shortcut);
   ontology-native "semantic" names are Atlas-paired elucidation only. Our
   name-match evidence source will be weak by design — that's the point.
3. **SHARE tier only** (KNOW ⊇ SHARE). Projections (Atlas, corpora) —
   "integrate freely, depend on nothing".
4. **Don't build on** `family_complex.json` (being deleted) or Atlas
   search-by-classification (known-broken filter, deferred).
5. **Terminology adoption:** `rdbms_*` (never hive), `cta_class`/CPA,
   membrane/propose-dispose, convert-1a/1b/1c, natural⊕semantic registers,
   `signals.zndx.org/sdg`.

## 5. Proposed sync work plan (Atelier side)

Ordered; A–D are buildable now against seed-baseline-v0.3 / current shapes,
E–F coordinate with Ægir's next tag.

- **A. Data source repoint** — add sdg-corpora as the pinned source (config
  key, e.g. `classify.aegir.corpora_dir` / release tag pin; submodule once
  the GitHub push happens). Extend loading: keep `aegir_release.py` for
  release-parquet dirs; add a corpora-footprint loader
  (`base_rows.parquet` + `base_table_index` → blind `TableSample`s) for
  HEAD-shape consumption while the release parquet is being rebuilt.
- **B. Vocabulary adoption** — load `vocabulary/annotations.parquet` (944
  codes, hierarchical via `parent_code`) as the classification vocabulary /
  ReferenceCategory set. Retire local TTL as authority. Fail-fast if the
  pinned release's vocab and ddl generations disagree (version-skew guard).
- **C. Config wiring** — HOCON: corpora/release dir, glossary name,
  `cluster_name="aegir"` for this integration path; preflight checks;
  per defaults philosophy, required-on + fail-fast.
- **D. Pipeline wiring + predictions artifact** — a `just`/gateway entry that
  runs `run_classification_pipeline(samples=<blind loader>)` with the adopted
  vocab and emits a predictions artifact keyed by `(table_id, column_id)` (or
  `table_name, col_name` for footprint-shape) for Ægir-side scoring; local
  self-score path retained where a reference is available (v0.3 checkpoint).
  BDD: `@slow @tier-1` efficacy scenario per operator-confidence pattern.
- **E. Writeback retarget** — `sync.py` hive_* → `rdbms_*` parameterization;
  redo the collision check (cta_class now applied); write under `atelier_*`
  classification namespace.
- **F. DST sdg: name audit** — verify emitted belief structures match the
  sdg: vocabulary names verbatim.
- **G. Docs/terminology sweep** — CLAUDE.md + mdbook: sdg-corpora contract,
  namespace, natural/semantic registers, 944-code moving vocab, KNOW⊇SHARE.

## 6. Open questions for Ægir coordination

- When does Convert 1 land + a coherent tag get cut (vocab and ddl/corpus at
  the same generation)? That's the pin for the first real efficacy run.
- Predictions handoff format: settle the exact schema Ægir's scorer wants
  (parquet columns, code granularity — leaf vs `cautious_code(tau)` levels).
- Will `build_atelier_release.py` re-emit `corpus_columns.parquet` per tag,
  or should Atelier's footprint loader be the canonical path?
- Glossary↔annotations relationship going forward: is the Atlas glossary
  regenerated from the 944-row vocab, or still the 540-term snapshot?
