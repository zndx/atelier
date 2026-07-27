# Referee diagnostic read + THE frame finding; Arm I initiated

**Date:** 2026-07-03 (follows 205344_curation_complete.md)

## Handoff standard (RH proposal — ADOPTED)

Atelier emits predictions; Ægir scores. Adopted as standing practice for
structural authenticity — our session no longer reads the key for scoring.
Exception recorded: ONE pre-declared diagnostic read below (score + an
aggregate structure read of the key), taken before adoption, no referee
tuning derived from it.

## The diagnostic read

Referee decisions emitted via `emit_aegir_predictions.py
--from-agent-mediated sdg` (2,004 rows) → `just score-atelier`:

| metric | value |
|---|---|
| hierarchical_score | **0.0225** |
| leaf_accuracy | 0.011 |
| miss_rate | 0.965 |
| Brier (conf mean 0.921) | 0.829 |
| coverage | 0.986 |

All four provenance rungs equally poor — including semantic-passthrough,
the designed null control. That uniformity is itself the tell: not a
name-signal failure, a task-definition failure.

## The frame finding

Aggregate structure of the key: **one code per table** (1,074/1,074
single-coded; 433 distinct codes = the template spine; all under
`SDG.GENERIC.*`/`SDG.ICE.*`; per-row `kind` ∈ entity/junction/dimension/
fact/eav). The keyed task is *table-class realization projected onto
columns* (CTA-at-table-granularity). The referee answered *per-column
semantic typing* (mass in `SDG.DOM.*` — which the key never uses). The
0.0225 measures the frame mismatch, not referee competence at either task.

Corollaries:
- The 503 `SDG.DOM.*` hypernym codes are apparently hint/retrieval
  vocabulary, not answer space — consistent with them being LLM-derived
  enrichment. (To confirm in the frame ruling.)
- Referee over-confidence (0.92 on wrong-frame answers) is a real
  calibration lesson independent of the frame: confidence prompting needs
  tempering, and Brier will police it once frames align.
- The curation artifacts remain valuable: per-column semantic labels are a
  coherent projection (the deployment task); if the ruling is table-class,
  the referee re-runs table-wise (1,074 decisions, richer context, cheaper
  than tonight's run).

## Frame question → Ægir (running note §13)

Pre-P5 gate task: every-column-inherits-table-class (then our arms need a
table-level projection layer), or per-column semantics with the current
key a spine-derivation simplification (their `rout_stop_address` example
reads per-column; set-valued refs at P5)? **Ruled before Arm I's expensive
steps.**

## Arm I initiation (frame-independent step done)

Generator coverage audit vs the 944: **271 covered (all `inferred`),
673 missing** — missing dominated by template-class codes, which
value-generators cannot cover by construction (a table-class isn't a
column-value distribution). Reinforces the frame question; the SVM
channel is per-column by nature and the ensemble likely needs a
table-projection layer under the table-class frame.

Holding Arm I's GPU-expensive steps (synth corpus authorship, NHSVM
training, maxsim collection) for the frame ruling — hours of build aimed
at the wrong frame is the exact waste the ruling prevents.
