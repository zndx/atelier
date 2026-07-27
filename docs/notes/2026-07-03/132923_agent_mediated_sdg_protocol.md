# Agent-mediated SDG adaptation — the honest-baseline protocol

**Date:** 2026-07-03
**Trigger:** RH: `just agent` (agent-mediated blind classification) drives
the rest of `just optimize`; adapt the full sequence to the SDG corpus
while maintaining the blind contract end-to-end.

## The chain as built (verified)

1. **`just optimize agent`** — `build_agent_mediated.py` assembles
   `build/data/agent_mediated/working_set.json` (per-column values via
   embedding_text, vocabulary, prior predictions/xlsx context) → Agent SDK
   runs `/curate-agent-mediated` → **`agent_mediated.json`** + `audit.json`
   + `training_weights.json`. The skill calls this "the most epistemically
   load-bearing operation in the project — the referee against which all
   pipeline improvements are measured." It already has a `calibrate`
   scope (taxonomy-adaptive bootstrap) for the no-priors case.
2. **`just optimize maxsim`** — `semantic_optimize.py` APO loop: edits
   Qdrant enrichment payloads only, scored against the **frozen**
   `agent_mediated.json` (hash-guarded start/end; refuses to terminate on
   drift), rescue@k over failure rows from a prior run's
   classifications.json. The critic cannot propose reference changes.
3. **`just optimize svm`** — `run_corpus_expansion_pipeline.sh` Phases
   A–E: coverage audit → agent generator authorship → synth corpus →
   NHSVM train/eval → refinement; Gate A (TARGET_ACCURACY ≥ 0.95 on
   validate vs the reference) + Gate B (cosine⊕SVM mutual-affirmation
   uplift). Training modes: `synth-only` / `reference-primary` /
   `reference+synth`.

So the agent-mediated referee is the root of the whole adaptation tree —
exactly the structure the SDG protocol needs.

## The two-arm insight (what "honest baseline" means here)

Two distinct integrity properties, one hard and one declarative:

**1. Answer-key isolation (hard, mechanical).** The agent + optimize
chain must never observe: `reference.parquet`, corpora
`ddl/*/base_rows.parquet` / DDL `COMMENT` JSON, `naming_map.parquet`,
the Atlas semantic register, or Ægir tree paths generally. Mechanism:
the SDG working-set builder is the *single ingress* — it reads exactly
two files (`corpus_columns.parquet`, `annotations.parquet`), inlines
everything the agent needs (values ride in the working set already), and
records input sha256s in metadata. The skill enumerates allowed inputs;
a post-hoc blind-integrity audit greps the agent transcript + optimize
artifacts for forbidden markers (`reference.parquet`, `semantic_col`,
`template_id`, `naming_map`, aegir paths).

**2. Transductive adaptation (legitimate, must be declared).**
`reference-primary` training and maxsim payload optimization adapt the
stack to the *blind inputs* using self-generated labels. That is not
leakage — it is Atelier's deployed behavior (`just optimize` IS in-situ
domain adaptation; an operator would do exactly this on their corpus).
But it specializes the model to the eval distribution, so it must be a
**declared arm**, not the hidden default. Hence two pre-registered arms:

- **Arm I (inductive / cold):** synth-from-vocab SVM only, unoptimized
  vocab-built Qdrant payloads, no agent referee in the loop. "OOTB
  Atelier."
- **Arm T (transductive / adapted):** agent-mediated blind curation →
  frozen referee → maxsim APO + `reference+synth` SVM → re-run. "Atelier
  as deployed, full adaptation, zero key access."

Arm T − Arm I is the measured value of the optimize machinery itself —
a lift Ægir gets for free from the protocol, alongside their
rung-sliced ablations.

Gate A numbers are agreement-with-referee (pseudo-labels), an internal
steering metric — never quoted as accuracy. Accuracy is only ever
Ægir's scorer against the held-back key.

## Protocol sequence

- **Stage 1 — Arm I**: SDG taxonomy_id isolation; vocab→Qdrant
  collection; generator coverage audit → `evolve-generators` for gaps →
  synth-only NHSVM head promoted under the SDG taxonomy; pattern-map
  decision (SDG remap or measured-dead); blind run → emit → score.
- **Stage 2 — `just agent` (SDG mode)**: working set from the blind
  release surface only; agent classifies blind into the 944 (calibrate
  scope; hierarchical integrity — parent codes are first-class, which
  is also what Ægir's `1/(1+d)` credit rewards); output under an SDG
  namespace of `build/data/agent_mediated/`.
- **Stage 3 — Arm T**: `just optimize maxsim` (frozen SDG referee) +
  `just optimize svm --training-mode reference+synth`; blind re-run →
  emit → score.
- Both arms reported; scored claims pin at P5.

## Asks back to Ægir

1. **Physically separate the key from the blind surface**: preview dirs
   currently hold `reference.parquet` beside `corpus_columns.parquet` —
   an agent pointed at the release dir would see the key sitting there.
   Ship future previews as `<release>/` (blind) + `<release>.key/`
   (reference), mirroring "the answer key is a physically separate
   file" one level up. (P5 sealed runs ship no key at all, so this is
   preview-era hygiene.)
2. Confirm the two-arm framing lands in the shared pre-registration
   (Arm I / Arm T alongside the rung + values-only ablations).

## Build list for `just agent` SDG mode

- `build_agent_mediated.py --sdg-release <dir>` (or a sibling
  `build_agent_mediated_sdg.py`): working set from
  `load_aegir_release_samples` + `load_sdg_vocabulary`; sha256 ingress
  manifest; refuses paths containing `reference.parquet`.
- Skill argument surface: `calibrate` scope against SDG working set;
  blind-contract preamble (enumerated inputs, forbidden files).
- `scripts/audit_blind_integrity.py`: transcript + artifact scan for
  forbidden markers; runs at the end of `just agent` and before any
  emit.
- Namespace the agent_mediated artifacts per taxonomy_id so the SDG
  referee can't collide with hive-poc curation
  (`build/data/agent_mediated/<taxonomy_id>/...`).
