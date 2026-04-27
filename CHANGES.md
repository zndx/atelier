# Atelier Changelog

All notable changes to this project are recorded here.  The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this
project adheres to a relaxed semver (pre-1.0 minor bumps may carry breaking
changes; the upgrade notes call them out).

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

## v0.2.0 — 2026-04-13

Phase-gate-2 baseline.  See `docs/notes/2026-04-19/170000_phase_gate_2.md`
for the full report.
