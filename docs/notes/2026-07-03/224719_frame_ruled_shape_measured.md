# Frame ruled per-column; corpus-shape gap measured; Ægir asks posted

**Date:** 2026-07-03 (follows 223613_referee_diagnostic_frame_finding.md)

## Ruling (RH)

**Per-column semantic typing with first-class hypernyms** is the gate's
objective — the deployment task (1M+-scale metadatasets, per-column
ensemble evidence). Consequences:
- Tonight's referee reference is the RIGHT frame → Arm T stands as-is.
- The preview key (one code per table) keys a different task; Ægir asked
  to re-derive references per column (slot_ref lineage, set-valued for
  junction-FK/role columns, hypernyms as legitimate reference levels).
- The 0.0225 diagnostic is fully explained; no referee re-run needed for
  frame reasons. Standing lesson: confidence 0.92 → Brier 0.83 — temper
  confidence prompting; Brier will police it once frames align.

## Corpus-shape measurement (posted to Ægir §14 with the asks)

Sampled column-count distributions per table:

| corpus | n | mean | median | p90 | p99 | max | ≥20 | ≥50 |
|---|---|---|---|---|---|---|---|---|
| sdg preview | 1,074 | 1.9 | 1 | 4 | 4 | 5 | 0% | 0% |
| GitTables | 3,000 | 15.9 | 11 | 32 | 174 | 440 | 24.6% | 3.1% |
| SOTAB | 2,000 | 8.8 | 8 | 15 | 27 | 29 | 4.4% | 0% |
| SchemaPile | 34,742 | 6.5 | 5 | 13 | 28 | 315 | 3.1% | 0.2% |

The x/y slot-template structure is near-degenerate for our task: median
ONE classifiable column — no sibling-context signal, no wide-table stress
(prompt batching, retrieval interference), zero clickstream-class tail.
Asked Ægir to target long-tailed mixes calibrated to these corpora (their
C25 subtree-mixing spec is the natural mechanism) and pointed at the
local reference datasets on /raid.

## Arm I plan under the ruling

Resumes against per-column codes; scored runs wait for the re-keyed
release (handoff standard — Ægir scores):
1. Synth corpus for per-column-meaningful codes: 271 inferred-covered +
   `evolve-generators` authorship prioritized by the codes the referee
   actually used (149) ∪ hypernym families; template-class codes excluded
   (not per-column targets).
2. NHSVM head trained synth-only, promoted under an SDG taxonomy_id.
3. Maxsim Qdrant collection from vocabulary embedding_text.
4. Cold blind run → emission → hand to Ægir for scoring when the
   per-column key exists.

## Session state

Long day: engine landed + federated identity, referee reference produced
(blind-audited), handoff standard adopted, frame ruled, shape gap
quantified. GPUs free, leases clear, all sibling-session channels updated
via the running-observations note (§1–14).
