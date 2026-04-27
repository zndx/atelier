<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Phases 3-5 Complete: Synth Framework + Belief Paths + DST Refinement

## Summary

Completed the final 3 phases of the 6-phase ontology refinement plan.
All 73 BDD scenarios pass (9 new scenarios added).

## Phase 3: Unified Synth Framework (Task #206)

### New Files
- `src/atelier/classify/synth_generators.py` — Single source of truth for 316+ value generators
- `src/atelier/classify/synth_registry.py` — Extensible registry with 3-layer coverage (hand-coded > template > inferred)

### Refactored Files
- `src/atelier/classify/synth.py` — Uses shared GENERATORS + registry parameter + `generate_for_vocabulary()`
- `scripts/generate_sample_source.py` — Uses shared generators via importlib (avoids numpy chain)

### Key Results
- 316/316 leaf categories now have hand-coded generators (was 300 before adding 16 Phase 2 generators)
- Registry adds template + inferred layers for custom vocabularies → >80% coverage on any vocabulary
- `generate_for_vocabulary()` convenience API for onboarding workflow

## Phase 4: Belief Paths + Meta-Tagging Overlay (Task #207)

### belief.py Additions
- `belief_path()` — Traces [Bel, Pl] from predicted leaf to root. Bel increases monotonically ascending.
- `cautious_code(tau)` — Returns deepest code where Bel > threshold. Answers: "at what level is evidence unambiguous?"

### evaluation.py Addition
- `epistemic_evaluation()` — Per-depth belief metrics (mean_bel, mean_pl, mean_gap, count), cautious_accuracy, mean_commitment_depth, belief_convergence

### New File: meta_tagging_overlay.py
- `META_TO_ICE` — 130+ hand-aligned mappings from meta-tagging numeric codes to ICE.* codes
- `translate_ground_truth()` — Converts meta-tagging annotations to ICE codes
- `build_blended_vocabulary()` — Composes base + meta-tagging into blended vocabulary
- `mapping_coverage_report()` — Reports mapping coverage across data files

### Pipeline Integration
- `_classify_column()` now returns `belief_path` and `cautious_code` fields
- Parquet output includes `dst_belief_path` (JSON) and `cautious_code` columns
- `epistemic_evaluation` added to pipeline summary

## Phase 5: DST Iteration Refinement (Task #208)

### bootstrap.py Additions
- `IterationMetrics` dataclass — Structured per-iteration metrics (mean_k, max_k, disagreements, coverage)
- `record_iteration_metrics()` — Records metrics at each bootstrap iteration
- `k_convergence_rate()` — Linear slope of mean K (negative = improving)
- `should_stop_early()` — Detects K plateau (2 consecutive non-decreasing iterations)

### Pipeline Integration
- Convergence loop uses `record_iteration_metrics()` for structured tracking
- Early termination via `should_stop_early()` (proof-of-progress paradigm)
- Summary includes `k_convergence_rate` and `epistemic_evaluation`

## BDD Scenarios Added (Phase 6, Task #209)

| Feature | Scenarios |
|---------|-----------|
| `belief_path.feature` | Leaf-to-root trace, cautious classification, epistemic evaluation |
| `synth_framework.feature` | Registry coverage, vocabulary-driven generation |
| `meta_tagging.feature` | Code translation, hierarchy consistency |
| `bootstrap.feature` | K convergence rate, plateau detection, decrease detection |

Total: 73 scenarios pass (was 64), 0 failures.
