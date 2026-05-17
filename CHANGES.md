<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Atelier Changelog

All notable changes to this project are recorded here.  The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to a relaxed semver (pre-1.0 minor bumps may carry breaking
changes; the upgrade notes call them out).

---

## v0.4.0-rc2 — 2026-05-11

Staging branch for the second `v0.4.0` release candidate.  Reconciles
the three deployment branches (`release/v0.3.0`, `release/v0.4.0`,
`deploy/v0.4.0-rc1`) onto trunk so the Cloudera proprietary header,
the algorithmic engine work, and the ontology / SOTAB strategy land
together for the rc2 soak.

### What landed from the merge

- **Cloudera proprietary header** swept across all source + docs
  (from `release/v0.3.0` + `release/v0.4.0`).
- **Algorithmic engine reconciliation** from `deploy/v0.4.0-rc1` —
  parent-aware DST frame, hierarchical cosine mass, cross-subtree
  cautious_code, ontology priors, governance cost model, cautious-
  review three-way decision, per-vocabulary synth-trained SVM,
  R7–R10 audit remediations, Embeddings Canvas reviewer's guide,
  FSM Pipeline Phases walkthrough, Nautilus mid-run watcher.
- **Ontology IRI canonicalization** — CCO shorthand vs canonical
  IRI clarifications in the ontology README, TTL header, and
  CCO module inventory note.
- **SOTAB v2 Coverage Strategy** architecture doc + Ægir handoff
  note (vocabulary / synth / grounding work moves to Ægir; Atelier
  becomes a consumer of trained artifacts).

### Standing issues (block GA)

`just behave` surfaces ~11 pre-existing failures + ~17 errors that
travel with the deploy-branch content.  None are merge-induced
(verified — the only merge-induced drift was a stale
`meta_tagging_steps` import, fixed at HEAD).  Outstanding triage:

- **Bootstrap mass-function assertion** (`bootstrap.feature:18`) —
  test expects LLM mass > 0.8 at confidence 0.9, but the
  `8a5f3de` LLM-discount recalibration (0.10 → 0.15) makes the
  actual mass 0.765.  Update the test threshold.
- **Terminal line editor errors** (12 scenarios) — pass in
  isolation, error in the full suite with
  `RuntimeWarning: coroutine 'TerminalSession.handle_input' was
  never awaited`.  Inter-feature event-loop pollution; isolate the
  offending hook.
- **Evidence-independence ontology-prior scenarios** (2 errors) —
  need to confirm new step bindings load against the merged
  classify surface.
- **Infra scenarios** (health_postgres, devenv_logs, application
  stack) — require a fully-up devenv stack at run time.
- **Classify failures** (classification / coverage_guarantees /
  experimentation / gpu_acceleration) — likely additional
  threshold drift from the discount recalibrations.

GA promotion requires all five clusters resolved or explicitly
deferred with `@known-failure` tagging.

### Upgrade notes

No upgrade actions from rc1.  Version bump only.

---

## v0.4.0-rc1 — 2026-04-28

First release candidate for `v0.4.0`.  Bumps `v0.3.0-rc1` after CAI
soak surfaced a substantive line of algorithmic work that warrants a
new minor rather than an `rc2`.  Carries the entire algorithmic
engine reconciled from the `rch/deploy` CAI workspace branch on top
of the `v0.3.0-rc1` UX surface, plus the rc1-soak production fixes
from `deploy/v0.3.0-rc1`.

### Headline — algorithmic engine

- **DST as iterative refinement** — bootstrap loop reframed as
  fixed-point iteration with explicit numerical-methods primitives
  (Banach 1922, Saad 2003 §4.1, Robbins-Monro 1951, Brandt 1977
  multigrid, Smets 1993).  Documented in
  [`docs/src/architecture/dst-evidence-independence.md`](docs/src/architecture/dst-evidence-independence.md)
  (689 lines).
- **Unified residual norm + contraction rate** — L2 combination of
  four normalized components (mean(gap)/gap_threshold,
  frac_unclear/clarity_target, mean(K)/k_threshold, frac(indep-tier
  disagreement)).  `bootstrap.residual_norm` and
  `bootstrap.contraction_rate` surface in `IterationMetrics` and
  `iteration_history`.
- **Hierarchical cosine mass + cross-subtree cautious_code** (Shafer
  §3, Smets §6) — when the LLM votes confidently in one subtree and
  no leaf decisively fits in the cosine-favored subtree, the system
  surfaces an honest "subtree X is the right place but no leaf fits"
  promotion via Smets least-commitment.
- **Parent-aware DST frame** — the LLM is free to vote at any
  hierarchy level; the DST frame honors internal-node codes; the
  headline picker treats every node as a tag candidate; parents
  render as first-class taxonomy rows.
- **Cosine reliability shaping (Haenni-Hartmann)** — margin-aware
  mass allocation; the cosine source's reliability scales with the
  top-1 vs top-2 margin instead of a fixed discount.
- **Ontology priors** — public substrate threaded through embedding
  text + LLM prompt + SAGE feature.
- **Governance cost model** — Type-II-aversion in the LLM system
  prompt; ICE-only sensitivity map (drop schema-assuming Branch A);
  invocative-rubric replaces prescriptive-checklist phrasing.
- **Discount calibration**: cosine 0.30 → 0.20, llm 0.10 → 0.15;
  SVM discount default 0.55 (Denoeux 2008 non-distinct-evidence
  framing — the incremental SVM trains on LLM labels).
- **Indep-tier revisit gate** — fires an LLM revisit when
  `{cosine, pattern, name_match}` agree on a code at meaningful
  consensus mass that disagrees with the LLM, even when DST K
  doesn't trip.

### Headline — operations & UX

- **Nautilus mid-run pipeline watcher** — daemon thread polls FSM +
  `BootstrapState.batch_audit`, fires structured `InterventionRecord`
  via callback when the run stalls, sweeps too long, or accumulates
  failures.  Pairs with halving retry (per-batch) and supervisor
  overwatch (post-run).  See
  [`docs/src/architecture/nautilus.md`](docs/src/architecture/nautilus.md).
- **start-app.sh self-heal on PGlite OOM** — Node `--max-old-space-
  size=8192` (was the implicit ~4 GB default that OOM-killed under
  thousand-column runs).  Backgrounds gRPC + uvicorn with `wait -n`;
  exits with diagnostics on critical-child death so
  `scripts/startup_app.py` restarts the stack.
- **Atlas taxonomy CLI** + `just sync-taxonomy` recipe.
- **Snapshot orchestration framework** — `scripts/snap_*.py`:
  concurrent per-table subagent fan-out against Bedrock Sonnet,
  manifest-driven resumability.
- **Bootstrap-secrets shell-quoting** + **SOPS decryption format
  fix** for `.env.cai.enc`.
- **Settings UI captions** — per-choice `captions` dict for
  `review_backend` and `shap_method`.

### Pipeline & classification

- `cautious_code` filters to threshold-cleared rows only.
- Persistent state file for auto-start source resolution.
- Customer-derived encoding scrub from universal vocabularies +
  provenance audit.

### BDD coverage added

- `features/agent/evidence_independence.feature` (317 lines)
- `features/agent/governance_cost_model.feature` (85 lines)
- `features/agent/fusion_strategy.feature` (22 lines)
- `features/gateway/auto_start.feature` (43 lines)

### Database

- No new migrations.  All schema work from v0.3.0-rc1 carries through.

### Known regressions (re-introduce before v0.4.0 GA)

- **Throttle-aware retry** — trunk's `state.throttle_count` +
  `ThrottledError` handling (commit `fd39ebb`) was dropped during
  the bootstrap.py reconciliation in favor of rch/deploy's algo
  base.  The `sweep_throttled` UI progress field is no longer
  populated (Status.tsx gracefully treats it as 0).  Re-port the
  throttle handling on top of rch/deploy's bootstrap.py before
  promoting to GA.

### Lineage notes

- 25 algorithmic-engine commits originally authored as
  `Test005 user005` (CAI workspace identity) rewritten to
  `Ryan Hill <rch@zndx.org>` with `Co-Authored-By: Claude` trailer.
- 5 cherry-picked soak-fix commits from `deploy/v0.3.0-rc1`
  similarly rewritten.

### Upgrade notes (v0.3.0 → v0.4.0)

1. No new database migrations.
2. **SVM discount default** is now `0.55` (was `0.20`) per Denoeux
   2008 non-distinct-evidence framing.  Operators with overlay
   overrides should re-validate.
3. **Discount recalibration**: cosine `0.30 → 0.20`, llm
   `0.10 → 0.15`.
4. UI rebuild: `just build-ui`.
5. Re-port throttle-aware retry before GA (see Known regressions).

---

## v0.3.0-rc1 — 2026-04-27

First release candidate for `v0.3.0`.  Cut from trunk at SHA `f33bae4` after
131 commits since `v0.2.0`.  Soak target: 72 hours on devenv + one CAI
smoke before promoting to `v0.3.0` GA.

### Headline features

- **ML Artifact Management + Extend Classification** (M1–M4) — trained
  CatBoost / SVM / UMAP bundles are now first-class entities in
  `ml_artifact_sets`, registered in PG, listed in the UI, and replayable on
  new data through a streamlined Extend pipeline that skips the LLM sweep
  and DST iteration.  See
  [`docs/src/architecture/ml-artifacts.md`](docs/src/architecture/ml-artifacts.md).
- **Belief-gap convergence pivot** — the bootstrap loop converges on
  `mean(Pl − Bel)`, not on K conflict.  The agent's `declare_converged`
  schema gained a required `convergence_kind` enum (`gap_threshold_met` /
  `iterative_convergence`).  Earlier K-centric convergence language is gone
  from system prompts and code paths.
- **Cautious-Code Review** — agent-mediated Seam A backoff that re-examines
  low-confidence predictions before they ship.  Replaces the unnamed
  `seam_a_review` scaffolding with a documented module.
- **Overwatch agent loop + report viewer** — Opus-class instance reviews
  finished pipeline runs and writes a markdown recommendation; UI surfaces
  the report inline with feedback affordances.
- **Governance SDK integration** — Atlas + Ranger clients for
  taxonomy/classification sync (gated by `cfg.governance_auto_sync &&
  cfg.has_atlas`); CDP control-plane discovery via `cdpcurl`; S3 probe in
  the health-check skill.
- **Web Terminal Agent panel** — pick model / provider with live TTFT +
  tok/s, readline-style line editor with arrow-key history.
- **Settings page (Phase 1 + 2)** — full parameter surface: 46 controls
  across 5 tabs, adaptive focus, per-run snapshot; reads/writes the
  in-memory overlay through `GET/PATCH /api/settings`.
- **Phase-gate validation** — 97.8% on meta-tagging, beating the LLM
  baseline (Phase Gate Report #2 in `docs/notes/`).

### Pipeline & Classification

- Renamed the *frontier SVM* to the **incremental SVM** in user-facing
  surfaces — UI tooltips, overlay labels, module docs, log lines, BDD
  scenario + matcher pair, and architecture docs.  The unrelated
  *frontier model* (Opus-class capability tier) and *frontier columns*
  (Monte Carlo stratification) usages are untouched.  On-disk filenames
  (`svm_frontier.pkl`), HOCON config keys, and function names are
  preserved for backward compatibility.
- TreeSHAP per-feature attribution via structured CatBoost input.
- CatBoost fit-to-LLM mode — training surface agrees with the oracle.
- Yager fusion strategy alongside Dempster (`classify.fusion_strategy`).
- Pattern evidence: 8 new patterns + LLM aliases; identifier-shape
  patterns (UDID / ICCID / IMEI); category-enum detection + sibling
  domain clustering.
- Pipeline invariants enforced at entry: `max_iterations >= 2` and
  `catboost.fit_to_llm = true` are non-negotiable.
- Reproducibility: fit-to-LLM CatBoost persisted alongside the run
  parquet; reference parquet shipped pre-classified for first-run UX.
- LLM-abstention rescue via CatBoost extrapolation; reasoning-trace
  capture for prescriptive revisit (+9pts iterative gain on attribution
  pilot).

### UI & UX

- Service Status state machine on Landing — `connecting` (yellow) →
  `connected` / `degraded` → `disconnected` (red, 10s alarm window) →
  back to `connecting` (no thrash).
- ML Artifacts panel + radio Active column on the Status page; Data
  Source panel revamp drops the inline `[active]` chip in favor of a
  leftmost radio column.
- Operator-overridable app display name (defaults to "Atelier";
  `ATELIER_APP_DISPLAY_NAME` for CAI rebrands).
- Live LLM-sweep visibility — heartbeat remap + sub-phase fields surface
  per-batch progress without polling tricks.
- Reference-column exclusion toggle on the Status page.

### CAI Deployment & Operations

- `CDSW_APP_PORT` plumbed end-to-end (gateway exec, Vite proxy,
  embedded terminal); local-dev still defaults to `8090`.
- SOPS + age encrypted CAI deployment defaults — minimal AMP config
  surface (4 required vars after the consolidation).
- `bake-everything-in` posture for fresh CAI deployments — Opus / Sonnet
  / Haiku ARNs, overwatch settings, and ground-truth fixture all
  delivered via the encrypted dotenv.
- LICENSE prep + snapshot scripts.
- `apply_cloudera_header.py` — proprietary header stamper for release
  branches.  Idempotent, with `--dry-run` / `--check` modes.
- `build_source_archive.sh` — self-contained tarball for offline CAI
  deployments (main repo + embedding-atlas submodule, no
  hermes-agent).

### Database

- New migration `20260427000000_ml_artifact_sets.sql`:
  - Adds `ml_artifact_sets` table with partial unique index
    `idx_ml_artifact_sets_one_active ON (is_active) WHERE is_active`.
  - Adds three columns on `datasets`: `artifact_set_id`,
    `parent_dataset_id`, `run_kind` (`'classify' | 'extend'`).
- Operators upgrading from `v0.2.0` must run `just migrate` after
  pulling.

### Documentation & Roadmap

- New: [Pareto Capability Evolution
  (Roadmap)](docs/src/architecture/pareto-capability-evolution.md) —
  research-shaped capstone integrating Active Learning, APO, and GEPA
  on top of the existing pipeline.
- New: [ML Artifacts & Extend
  Classification](docs/src/architecture/ml-artifacts.md).
- Updated: classification pipeline glossary clarifies *incremental SVM*
  vs *frontier model* vs *frontier columns*.
- Phase Gate Report #2 — iterative revisit, reasoning capture, honest
  uncertainty (in `docs/notes/2026-04-19/`).

### Refactors

- `frontier SVM` → `incremental SVM` (terminology only; no code or
  filename rename).
- `seam_a_review` → `cautious_review`.
- `ground_truth` → `curated_reference` (parity scripts + fixture rename).
- FAIR-aligned overwatch review prompts (no suppressive language).

### Stability & Bugfixes

- LLM_SWEEP hang remediation: heartbeats, attempts gate, deadline,
  circuit breaker.
- Auto-start stability: connectivity gating, Anthropic timeouts,
  off-loop seeding, nautilus auto-cancel, daemon-thread death recovery.
- Operator-initiated cancel + env-seeded source race fix.
- Throttle-aware retry — sleep at the same batch size, do not halve.
- SVM retrain no longer wipes the fit-to-LLM CatBoost install.
- AMP metadata env defaults no longer override HOCON thesis defaults.
- Anthropic credential validator no longer treats Bedrock ARNs as model
  IDs; SDK v0.1.56 `thinking=adaptive` workaround.
- 65536 max_tokens on Bedrock; SDK session-continuity error handling.
- Pipeline uses source's connection/database, not HOCON defaults.
- Embedding warmup + offline mode — defense-in-depth on first load.

### Tests & BDD

- Mock `declare_converged` supplies the now-required `convergence_kind`.
- Tier-0 and tier-1 BDD coverage for ML Artifact Sets, Extend pipeline,
  and gateway artifact-set endpoints.
- Settings full feature with 46-control round-trip.
- Settings page phase 2 BDD coverage (adaptive focus, per-run snapshot).

---

## Upgrade notes (v0.2.0 → v0.3.0)

1. **Run `just migrate`** to apply
   `20260427000000_ml_artifact_sets.sql`.
2. **Rebuild the UI**: `just build-ui`.
3. **Convergence semantics**: pipelines that explicitly set
   `classify.bootstrap.k_threshold` should switch to
   `classify.bootstrap.gap_threshold` (default `0.10`).  K is now
   diagnostic-only.
4. **Agent SDK**: any external code calling `declare_converged` must
   supply `convergence_kind` (`"gap_threshold_met"` or
   `"iterative_convergence"`).
5. **Terminology**: tooltip / overlay strings reference the
   *incremental SVM* now.  Internal names (filenames, config keys,
   function names) are unchanged for compatibility.
6. **CAI port**: deployments that previously hardcoded gateway port
   `8090` should switch to honoring `CDSW_APP_PORT`.

## Known issues

- Pyright reports import-resolution errors for `numpy`, `pyarrow`,
  `joblib`, `umap` — these resolve correctly in the runtime venv.
  Type-check from inside `devenv shell` to silence.
- Aegir leaderboard integration deferred (see Pareto Capability
  Evolution doc).

---

## v0.2.0-m10 — 2026-04-19 (Phase Gate #2 milestone)

Marks the **10th M-numbered milestone** — the validation maturity gate
where the iterative DST-fusion loop was demonstrably consulting injected
evidence.  Carries 76 commits since v0.2.0.  Headlines:

- **Belief-gap convergence pivot** (replaces K-conflict gating)
- **Cautious-Code Review** (Seam A backoff) — agent-mediated review of
  over-specified DST predictions before they ship
- **TreeSHAP per-feature attribution** via structured CatBoost input
- **Phase-gate validation: 97.8%** on meta-tagging — beats the LLM
  baseline; documented in `docs/notes/2026-04-19/`
- **Reasoning-trace citation analyzer** — directional evidence that
  pass-2 reasoning consults injected evidence terms.  +9 pts iterative
  gain on the attribution pilot via prescriptive revisit
- **Live LLM-sweep visibility** on the Status page
- **Curated-reference terminology** (renamed from `ground_truth`)
- **Governance SDK integration** (Atlas + Ranger + CDP discovery)
- **/health-check skill** with `chk` terminal alias
- **Settings page Phase 1 + 2** (46 controls, 5 tabs)
- **Pipeline invariants enforced** at entry (`max_iterations >= 2`,
  `catboost.fit_to_llm = true`)

The full commit list for this window is captured in the v0.3.0-rc1
section above — v0.2.0-m10 is a stable intermediate milestone that
v0.3.0-rc1 builds on (M1–M4 ML Artifact Sets + Extend Classification
all land *after* this milestone).

This tag's history: originally `v0.2.0-phase-gate-2` (working name),
then briefly `v0.2.1`; final name is `v0.2.0-m10` to fit the
M-numbered milestone series and establish a `v0.2.0-mN` pattern for
intermediate milestone releases.

---

## v0.2.0 — 2026-04-16

Vocabulary routing, LLM robustness, rich terminal, UX polish.
