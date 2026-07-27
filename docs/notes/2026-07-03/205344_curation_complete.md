# SDG agent-mediated reference COMPLETE — task 8 closed

**Date:** 2026-07-03 (follows 180626_sdg_curation_launched.md)

## Result

Full blind curation of the preview on the local `referee`
(Nemotron-3-Nano-30B-A3B-BF16, GPUs 0–3, ~2.5h wall):

- **1,074/1,074 tables · 2,033 decisions · 2,004 resolved (98.6%)**
- 29 unresolved (response pathologies; retryable via resume after
  clearing them from agent_mediated.json if wanted)
- **Blind-integrity audit CLEAN** — ingress hashes unchanged, zero
  forbidden markers across working set / decisions / audit trail
- 149 distinct codes used; confidence mean 0.921, median 0.95, 41 below
  0.7; 22 vocabulary-validation retries
- All four provenance rungs covered — 891 engine-derived / 673 composed /
  441 semantic-passthrough / 28 degraded-mechanical (matches the release
  census exactly, so rung-sliced analysis has full support)
- Top codes: AGENT_ROLE (273), AGENT_ROLE_CATEGORY (220), IDENTIFIER
  (220), AGENT_ENTITY (205), IDENTIFIER_SCHEME (143) — plausible for a
  DDL corpus dominated by role/agent/identifier columns.

Artifacts: `build/data/agent_mediated/sdg/{working_set,agent_mediated,
audit,review_state}.json` (gitignored build data).

## Late defect found + fixed

Table-parallel workers mutating `decisions`/`audit`/`state` while
`_persist` serialized them → "dictionary changed size during iteration" at
table 1072/1074 (the pipe to `tail` masked the non-zero exit — background
invocations should avoid trailing pipes). Resume machinery absorbed the
crash (2 tables re-run); all mutations now share `_io_lock` with persist.

## Ops

Engine shut down; GPUs 0–3 and the shared-dir lease released; §12 of the
running-observations note updated for the Ægir session (window free).

## Next (Arm T stage 3 + Arm I)

1. Adapt `just optimize maxsim` (semantic_optimize.py) to the SDG referee:
   frozen-oracle path already keyed to `agent_mediated.json` — needs the
   SDG taxonomy_id namespace + a Qdrant collection built from the SDG
   vocabulary, and a pipeline run dir to source failure rows from (i.e.
   an Arm I cold blind run comes FIRST).
2. Arm I cold stack: generator coverage audit vs the 944 →
   `evolve-generators` for gaps → synth-only NHSVM head promoted under the
   SDG taxonomy_id → cold blind run → `--from-run` emission → diagnostic
   score.
3. Then `just optimize svm --training-mode reference+synth` with the
   referee reference + `training_weights` derived from audit confidences.
