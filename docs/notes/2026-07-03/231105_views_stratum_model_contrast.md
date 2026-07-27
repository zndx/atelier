# R4: views as scored stratum + the pre-registered ± model contrast

**Date:** 2026-07-03 (follows 224719_frame_ruled_shape_measured.md)
**RH direction:** Atelier scores the views Ægir layers onto the relational
core — the central component of the curated text+view corpus the bespoke
H-Net+RWKV trains on. Endgame: show that model contributes positive DST-
ensemble lift on in-distribution relational constructs.

## Grounding (verified in corpora)

`ddl/b4bbc02f7800b0ea/views.parquet`: 974 views with `sql`,
`verbalization` (the text half of the text+view pairs), `columns_json`
(explicit per-column lineage `view_col → base_table.base_col` —
derivation-aware reference keying is deterministic for Ægir's realize
layer), `rows_json` (materialized values → blind-surface material),
`base_tables_json`/`fk_json` (topology). View widths median 4 / p90 8 /
max 9 — already richer than base tables (median 1). One quality flag
passed along: sampled verbalizations show degenerate slot-fills.

## What went into the spec of record (aegir …225241…md)

- **§0 extended** with the endgame: the gate must be able to measure the
  H-Net+RWKV's ensemble contribution, hence views as a stratum.
- **R4**: view columns in the blind surface (+`construct` base|view and
  `derivation` identity|rename|aggregate|computed|join_key slice fields);
  per-column references derived THROUGH the lineage (identity inherits
  the base column's ref — depends on R1; aggregates key to
  measure/aggregate hypernyms, set-valued where derivation spans);
  verbalizations ship as the docs-arm channel only, never inline with the
  names+values blind surface; **view holdout doubly binding** (views ARE
  the model's training data — gate views must come from held-out
  partitions or the measurement collapses into memorization).
- **Ensemble-contribution protocol (pre-registered)**:
  `lift(X) = score(Arm X ⊕ model) − score(Arm X)` per arm; primary
  endpoint = view-stratum hierarchical lift on held-out partitions
  ("in-distribution" scoped as distribution-not-instance); secondary
  slices by derivation class (hypothesis: max lift on
  aggregate/computed, where surface signal is weakest and ontology-
  informed priors matter most), rung, kind; guardrails: base-column
  no-regression + Brier non-degradation; preconditions: the
  source-independence memo (Ægir's §5.2 commitment) argued against each
  existing source, model outputs as vocabulary-code mass functions
  (the reserved 7th DST slot), both sides scored by Ægir per the
  handoff standard. OOD generalization (GitTables/SOTAB) explicitly a
  separate later claim.

## Atelier-side implications (build queue, after R1/R4 land)

- Loader: `construct`/`derivation` fields alongside register/rung
  (trivial extension of the v2 loader).
- Fusion frame: the 7th evidence-source slot's mapping contract (model
  score vector → mass function over vocabulary codes) — design with the
  independence memo in hand.
- Referee + arms run unchanged over view columns (they're just columns
  with richer siblings) — the working-set builder needs zero changes
  beyond the new fields.

## Conclusion-strength discipline

The claim being built is deliberately scoped: "positive ensemble lift on
in-distribution relational constructs" (held-out sdg-corpora views), not
"the model improves classification generally." The scope IS the design —
per the synthetic-conclusion-strength feedback rule.
