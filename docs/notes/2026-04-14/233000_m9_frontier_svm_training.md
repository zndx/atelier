<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# M9: Frontier-Label SVM Training via R-MDMC Sampling

## What Changed

Implemented progressive SVM retraining on blended synthetic + frontier LLM
labels. After the bootstrap LLM sweep produces high-quality Opus-tier labels,
the SVM is retrained early and incrementally, carrying corpus-specific signal
into the validation and convergence phases.

### Core Addition: `train_svm_on_frontier_labels()` (ml_train.py)

New training function that:
1. Collects frontier labels from `state.labels` where `label_source in ("llm", "llm_revisit")`
2. Builds SVM texts via `build_svm_text(col.name, col.column_type, col.values[:5])`
3. Optionally loads synth data for blending (vocabulary breadth)
4. Concatenates synth + frontier texts/labels
5. Trains `SVMClassifier()` on the blended set
6. Threshold guards: min 20 frontier labels, min 3 distinct classes

### Three-Phase Retraining (pipeline.py)

`_maybe_retrain_svm()` helper encapsulates retrain + hot-swap:

1. **Post-sweep**: After first LLM sweep (always) — SVM available for first ML validation
2. **Iterative**: During programmatic revisit loop (≥10 new labels heuristic)
3. **Final**: Before final classification pass (only if not converged)

Hot-swap via `ml_inference.reset()` + `configure_paths(svm_path=..., catboost_path=...)`.

### New Agent Tool: `retrain_svm` (agent_loop.py)

6th tool for the agent convergence loop. The agent decides when to retrain
during convergence — maximum control over timing. Handler calls
`train_svm_on_frontier_labels()` and hot-swaps the model.

Required threading `cfg` (AtelierConfig) through `_dispatch_tool()`.

### Config Knobs

```hocon
classify.bootstrap {
  frontier_svm_retrain = true       # Default on
  frontier_svm_min_labels = 20      # Minimum frontier labels to trigger
}
```

Wired through: `config/base.conf` → `config.py` (`_HOCON_MAP` + `AtelierConfig`)
→ `bootstrap.py` (`BootstrapConfig` + `bootstrap_config_from_cfg()`)

### Summary Audit Trail

Pipeline result dict includes:
- `svm_retrained_on_frontier`: bool
- `svm_frontier_training_samples`: int (when retrained)
- `svm_frontier_model_path`: str (when retrained)

## Why

### DST Independence Preserved

The independence claim is central to DST evidence fusion:
- **Training signal**: Opus (frontier model, used in LLM sweep)
- **Bulk LLM in DST fusion**: Sonnet/Haiku (subagent model)
- **SVM feature space**: sparse TF-IDF (orthogonal to dense embeddings)

Different models + different features + different inductive biases = genuine independence.

### Why Blend (Not Replace)

- Synth data covers all vocabulary categories (breadth)
- Frontier labels capture corpus-specific patterns (depth)
- Categories absent from the frontier sample still have synth coverage

### Why SVM Only (Not CatBoost)

CatBoost uses dense sentence-transformer embeddings that overlap with
cosine's feature space. Retraining CatBoost on frontier labels would
weaken independence for two evidence sources simultaneously.

## Files Modified

| File | Change |
|------|--------|
| `src/atelier/classify/ml_train.py` | `train_svm_on_frontier_labels()` with synth blending |
| `src/atelier/classify/pipeline.py` | `_maybe_retrain_svm()` + 3 call sites + summary audit trail |
| `src/atelier/classify/agent_loop.py` | `retrain_svm` tool (6th) + handler + dispatch + cfg threading |
| `src/atelier/classify/bootstrap.py` | Config/state fields + factory wiring |
| `src/atelier/config.py` | HOCON map entries + AtelierConfig fields |
| `config/base.conf` | `frontier_svm_retrain` + `frontier_svm_min_labels` |
| `features/agent/bootstrap.feature` | 1 new @slow scenario |
| `features/agent/step_defs/bootstrap_steps.py` | 2 step definitions |
| `docs/src/architecture/classification.md` | M9 section (replaced Future stub) |
| `docs/src/architecture/agents.md` | 6th tool + retrain_svm documentation |
| `docs/src/architecture/synth.md` | Frontier training section + diagrams |

## Verification

- 97 BDD tier-0 scenarios: all pass (0 failures)
- New @slow scenario validates frontier SVM end-to-end (runs with `just behave`)
