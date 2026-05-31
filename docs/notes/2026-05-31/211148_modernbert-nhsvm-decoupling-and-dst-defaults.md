# ModernBERT+NHSVM: kill scripts/-import coupling + align DST-mix defaults

Date: 2026-05-31

## Context

Cross-referenced `tmp-critique.txt` (prior-session critique of the
qdrant/maxsim + "mBERT-NHSVM" work) against `trunk`. Validated findings,
then acted on the tech-debt the critique surfaced.

Critique cross-reference verdict (focused on ModernBERT+NHSVM):

- **"mBERT" is a misnomer** — confirmed. The only encoder anywhere is
  `answerdotai/ModernBERT-base` (monolingual English, 768-dim). No
  multilingual model exists. Name still misleads.
- **Encoder coupling smell** — confirmed and *worse than at critique
  time*: it had become the production inference path while still being
  imported from `scripts/` via a `sys.path.insert` hack.
- **Kronecker-on-dense singular matrices** — confirmed verbatim
  (`factorized_nhsvm.py` docstring: TF-IDF 98.93% vs ModernBERT 4.26%
  on the materialized expansion; factorized per-node form dodges it).
- **STALE claim**: production/experimental roles have *flipped*. The
  dense ModernBERT factorized head is now the production-default SVM
  channel (`config: classify.svm.source = "registered"`, registry-loaded
  `NHSVMHeadAdapter`); the TF-IDF + Crammer-Singer path is now the
  `per_vocab_legacy` rollback.
- New caveat the critique predates: the dense head needs a post-hoc
  `softmax_temperature` (structured-hinge logits "flatten under the
  287-way softmax").

## Changes made

1. **Extracted the encoder** → `src/atelier/optimize/svm/encoder.py`
   (canonical `encode_modernbert`, `MODEL_ID`, `EMB_DIM`).
   - Removed the `sys.path.insert(scripts/)` hack from
     `classify/factorized_nhsvm.py::NHSVMHeadAdapter._encode` and
     `optimize/svm/reference.py::encode_with_cache` — both now import
     from the package.
   - `scripts/reflect_nhsvm_modernbert.py` re-exports from the core
     module so the research scripts that
     `from reflect_nhsvm_modernbert import encode_modernbert` keep working.
   - **Added a process-wide model cache** (`lru_cache` on
     `(model_id, device)`). The runtime adapter encodes one column at a
     time (`batch_size=1`); previously each call reloaded the ~600 MB
     encoder from `from_pretrained`. This is load-bearing for making the
     channel viable as a default. (Follow-up: batch-encode at the adapter
     level instead of per-column.)

2. **Aligned DST-mix defaults** (`config/base.conf`):
   - `classify.svm.enabled = false → true`. The intended trained/embedding
     DST mix is **CatBoost + MaxSim + modernBERT-NHSVM**; MaxSim was
     already `true`, CatBoost rides the non-negotiable `fit_to_llm`
     invariant, SVM was the missing one.
   - Kept `source = "registered"` (fail-fast). Per operator decision: if
     the pipeline is triggered before `just optimize` trains+promotes a
     head, the run errors loudly rather than degrading.

3. **Diminished the legacy TF-IDF SVM path** (slated for removal):
   - `_ensure_per_vocab_svm` (pipeline.py): DEPRECATED docstring banner +
     runtime `logger.warning` at entry.
   - `SVMClassifier` (svm_classifier.py): LEGACY/deprecated class docstring.
   - base.conf: `per_vocab_legacy` and `auto` source modes marked
     DEPRECATED / slated-for-removal.

## Verification

- `pytest tests/classify/test_nhsvm_cs.py` → 8 passed.
- Import checks: encoder loads, both production consumers decoupled
  (no `sys.path` / `reflect_nhsvm_modernbert` references), script
  re-export resolves to `atelier.optimize.svm.encoder`.
- `load_config()` → `classify_svm_enabled=True`, `source='registered'`,
  `classify_maxsim_enabled=True`.

## Operator-facing implication (flagged)

With `svm.enabled=true` + `source=registered` now the default, any
pipeline run without a promoted head fail-fasts. This includes BDD
`@slow` pipeline scenarios and fresh dev/CI environments. To run there,
either promote a head (`just optimize svm`) or set
`ATELIER_CLASSIFY_SVM_ENABLED=false` / `ATELIER_CLASSIFY_SVM_SOURCE=auto`
in those environments.

## Not done (future)

- Delete `_ensure_per_vocab_svm` + the TF-IDF Kronecker paths in
  `svm_classifier.py` once a promoted head is guaranteed in every
  environment.
- The Aegir (HNet/RWKV) `encoder_id` branch in `_encode` + DST
  independence guardrail (separate registry encoder key + corpus
  provenance assertion so an Aegir head can't be trained on the synth
  enrichment corpus). The `encoder_id` dispatch in `_encode` is the
  intended plug point.
