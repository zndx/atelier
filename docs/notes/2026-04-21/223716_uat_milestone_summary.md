<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Atelier — 3-Week Milestone Summary for UAT

**Window:** 2026-04-05 → 2026-04-21 (17 days, 219 commits, all on `trunk`)
**Repository:** [zndx/atelier](https://github.com/zndx/atelier)
**Author:** Ryan Hill (rch@zndx.org)

## Executive summary

Atelier is an agentic classification workbench built for Cloudera AI (CAI).
Starting from an empty repo on 2026-04-05, the project landed a
Dempster-Shafer evidence-fusion classification pipeline backed by a
316-class BFO-grounded vocabulary, GPU-accelerated SAGE/SHAP explanations,
an iterative-revisit agentic convergence loop with reasoning capture,
full Cloudera governance integration (Atlas + Ranger), and a single-form
CAI deployment that installs itself end-to-end from a fresh git clone.

The April 19 parity run against UAT's reference tables produced
**96.95% exact accuracy on synthetic tables with hints** and **~97% on
tables without hints when ignoring ID columns** — materially ahead of
the 93% LLM-only baseline the UAT reviewer cited and well above the 36%
third-party AMP baseline that triggered the re-evaluation.

This document is a milestone-oriented digest; the full commit-level
release notes are in **Appendix A**.

## Milestones

### M0 — Foundation (Apr 5 – Apr 11)

Project scaffolding: devenv-first development environment, gRPC core +
FastAPI gateway + React 19 UI, PostgreSQL (PGlite embedded for CAI,
external PG for dev), Qdrant vectors, Anthropic Claude Agent SDK with
Bedrock provider support. HOCON as single source of truth for config.
CAI AMP deployment hardened: Node.js via nvm, PGlite over pgserver,
protobuf coexistence with CML system packages, self-healing install
fallback. Keystone agents with XYFlow orchestration canvas. Embedded
ghostty-web WASM terminal wired to the Agent SDK.

### M1 — Classification Pipeline v0 (Apr 12 – Apr 13)

Dempster-Shafer evidence-fusion pipeline landed in five labeled
milestones (M0 → M5 in the git history):

- **M0**: DST pipeline skeleton with pignistic probability ranking,
  `HierarchicalClassification` with belief / plausibility / conflict.
- **M1**: LLM bootstrap convergence loop — LLM sweep → ML validation →
  disagreement revisit.
- **M2**: ML classifiers (CatBoost + SVM), synthetic training data,
  backend abstraction.
- **M3**: E2E validation framework, SAGE global feature importance.
- **M4**: SHAP explanations, configurable DST discounts, thread-safe
  model loading.
- **M5**: Real-data validation baseline, deprecated-term handling, 1-pg
  architecture doc.

Six evidence sources active (name-match, pattern, cosine, LLM, CatBoost,
SVM). Fusion strategies pluggable between Dempster (conflict-normalized)
and Yager (redirect conflict to ignorance).

### M2 — Universal Vocabulary + Scale (Apr 14)

- **BFO-grounded 316-class ICE vocabulary** using Prudhomme et al.
  (2025) PROV-to-BFO mapping criteria, CCO-mediated alignment
  (InformationEntityOntology + AgentOntology + ExtendedRelationOntology).
  ICE trichotomy: Designative (names/IDs), Descriptive
  (descriptions/measurements), Prescriptive (software/specs).
- **Monte Carlo sampling** (`monte_carlo.py`, `row_sampler.py`) for
  stratified frontier LLM sweeps + label propagation on the residual.
- **GPU acceleration** for SAGE, PermutationSHAP, CatBoost training via
  custom vectorized kernels; NVIDIA driver symlink pattern for CAI.
- **R-MDMC pilot**: row-level Monte Carlo sampling for the frontier SVM.
- **Agent-driven convergence loop**: Claude directs the bootstrap via the
  Agent SDK rather than a hard-coded state machine.
- Source-aware pipeline routing with OOTB sample auto-import and 25
  mixed-domain sample tables under `data/sample/`.

### M3 — Convergence + Robustness (Apr 15)

- **Belief-gap convergence** — `mean(Pl − Bel)` replaces K as the
  primary convergence measure. K remains diagnostic.
- **Frontier-label SVM** retrained during the bootstrap loop via R-MDMC
  sampling of accumulated LLM labels.
- **Pattern detection audit** — 8 validators (Luhn, IPv4, phone, currency,
  etc.) with graduated mass and confusable-pair handling; fixed phone
  false positives that had been inflating DST conflict across all columns.
- **Hive annotation source auto-discovery** on gateway startup.
- **Persistent terminal sessions** with ring-buffer replay.

### M4 — Governance + Overwatch (Apr 16)

- **Governance SDK** — Atlas + Ranger clients with URL normalization,
  CDP control-plane discovery via `cdpcurl`, optional Atlas taxonomy +
  entity-tag sync. Production audit with 7 remediated findings.
- **Health-check skill** — parallel subagent probes of the full CAI ML
  platform (cmlapi, MLflow, S3 Object Store via Data Connections, sibling
  AI Studio AMPs). `chk` terminal alias.
- **Overwatch foundation** — native CLI install, GitHub App wiring,
  dedicated Opus instance for autonomous post-run analysis.
- **Conversation continuity** across terminal sessions, 128K max output
  tokens, SOPS+age encrypted deployment defaults.

### M5 — Thesis Validation + Settings UX (Apr 17)

- **Phase-gate validation: 97.8% on meta-tagging** — beats the third-party
  AMP LLM-only baseline cited in the UAT review.
- **CatBoost fit-to-LLM regime** — trains CatBoost on
  `(embedding_text, llm_predicted_code)` pairs after the LLM sweep so
  SHAP/SAGE attributions explain the *current* LLM's decisions rather
  than a disagreeing pre-trained surrogate.
- **Settings page** — 46 operator-tunable parameters across 5 tabs with
  per-run snapshots.
- **Adaptive focus** — UI surfaces the parameters most relevant to the
  current FSM state.
- **100% coverage target** default — converges only when every column
  has a prediction.
- **Meta-tagging source** — local private-dir mount; data never commits.
- **Thesis-aligned defaults** — overwatch auto-enabled when
  `ANTHROPIC_API_KEY` is present, vocab tables excluded from
  classification, `predicted_annotation` exposed alongside
  `predicted_label`.

### M6 — CAI Deployment Readiness (Apr 18)

- **Web Terminal Agent panel** — pick model/provider with live
  TTFT (time-to-first-token) and tokens/sec metrics; readline-style
  line editor with arrow-key history.
- **SOPS-delivered ground truth** — meta-tagging reference fixture
  SOPS-encrypted alongside model ARNs + overwatch settings.
- **Four-var AMP form** — all operator-tunable CAI deployment
  parameters collapsed to 4 required env vars in the AMP metadata; the
  rest live in the encrypted dotenv.
- **Pipeline self-remediation** — halving retry with per-column
  fallback, nautilus mid-run supervisor, GPU auto-detect, Bake-everything
  install (pre-built embedding-atlas dist, pre-cached vocabulary).
- **Model catalog** — Opus 4.7 / Sonnet 4.6 / Haiku 4.5 as the default
  three-tier stack.

### M7 — Reproducibility + Authoritative Reference (Apr 19 – Apr 20)

- **Authoritative curated reference** (`curated_reference.csv`) —
  re-frames UAT labels as *provisional*. Built from generator-derived
  evidence + spot-checked corrections. Drives objective UAT-vs-Atelier
  delta analysis.
- **SVM ablation** — 7 arms (naive stop-word removal, Crammer-Singer,
  full-vocabulary training, etc.) with a leak-sanitized Hive-import
  dataset generator.
- **SOTAB Schema.org attribution pipeline** — pilot against the SOTAB
  CTA benchmark with iterative-revisit driver.
- **Iterative reasoning capture** — GLM-4.7 reasoning trace → prescriptive
  revisit prompt → **+9 accuracy points on iterative gain**.
- **Abstention rescue** — CatBoost extrapolation for rows the LLM
  abstained on.
- **FAIR-aligned review prompts** — removed prescriptive
  "do not recommend X" language; replaced with transparency /
  explainability / reproducibility principles.
- **(Target, tolerance) health signals** — asymmetric direction
  (undershoot vs overshoot depending on signal semantics). Replaces
  binary 100% green/red gates.
- **Reference-column exclusion toggle** on the Status page with a
  production-paired-column integrity explanation.
- **Name-index derivation fix** — parent codes were being assigned to
  leaf columns; 6 more columns rescued.
- **Fit-to-LLM CatBoost persistence** — trained CatBoost saved alongside
  the run parquet so the ML-only replay path is reproducible from a
  run directory alone.

### M8 — Stabilization Blitz (Apr 21)

A concentrated stabilization pass driven by two parallel CAI-deployment
investigations:

- **API key whitespace strip** — leading space in `ANTHROPIC_API_KEY`
  from AMP form produced `LocalProtocolError: Illegal header value`.
  Now stripped at config load.
- **Auto-start stability** — `asyncio.to_thread` wrapper around all
  synchronous seed functions so Hive JDBC probes cannot block the
  event loop during lifespan startup.
- **Explicit timeouts** — Bedrock (`connect=15s, read=180s`), Anthropic
  (`httpx.Timeout(connect=15.0, read=180.0)`), nautilus watcher,
  auto-start probes. Bedrock SDK's 600s default was the silent
  blackhole vector.
- **Silent daemon-thread death recovery** — per-batch FSM heartbeats,
  `BaseException` catch, force-reset on `/api/fsm/cancel`.
- **LLM_SWEEP hang remediation** — heartbeats inside the halving
  recursion (FSM `updated_at` now advances mid-tree), attempts counter
  gates separate from success-calls counter, consecutive-failure
  circuit breaker (~2 min trip on a 15s connect-timeout),
  `SweepDeadlineError` subclass for log-readability.
- **Design-invariant enforcement** — `max_iterations >= 2` and
  `fit_to_llm = true` are non-negotiable; attempts to set them below
  the floor raise `ValueError` at pipeline entry.

## Numbers that matter

| Metric | Value | Source |
|---|---:|---|
| Exact accuracy, synth tables with hints | **96.95%** | UAT reproduction, `meta-tagging-clean.zip` (Apr 19) |
| Exact accuracy ignoring ID columns | ~97% | UAT reproduction (tables w/o hints) |
| Phase-gate validation, meta-tagging | 97.8% | Internal run, Apr 17 |
| Third-party AMP baseline cited in UAT review | 36% | UAT acceptance review doc |
| LLM-only baseline (Claude Opus) | 93% | Same review |
| Iterative-revisit accuracy gain | +9 pts | GLM-4.7 reasoning capture (Apr 19) |
| Vocabulary size | 316 classes | BFO-grounded ICE (Apr 14) |
| Operator-tunable parameters | 46 | Settings page (Apr 17) |
| Evidence sources active | 6 | DST pipeline (Apr 12+) |
| Commits in the 3-week window | 219 | `git log --since=2026-03-31` |
| Required CAI AMP form fields | 4 | SOPS-encrypted deployment (Apr 18) |

## Architecture at a glance

```
┌─────────────────────────────────────────────────────────────┐
│  React 19 + Ant Design + XYFlow (Vite → ui/dist for CAI)    │
│  ─ Landing  ─ Status  ─ Settings  ─ Embeddings  ─ Canvas    │
│  ─ embedding-atlas (fork, pre-built dist committed)         │
└──────────────────────────┬──────────────────────────────────┘
                           │ REST /api/*
┌──────────────────────────▼──────────────────────────────────┐
│  FastAPI Gateway (src/atelier/gateway.py)                   │
│  ─ source + dataset registry, FSM run orchestration         │
│  ─ Agent SDK terminal (Claude Agent SDK + Bedrock)          │
└──────────────────────────┬──────────────────────────────────┘
                           │ gRPC :50051
┌──────────────────────────▼──────────────────────────────────┐
│  Atelier gRPC Core (proto-first)                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Classification Pipeline (atelier.classify/)            │ │
│  │ ─ Dempster-Shafer evidence fusion (6 sources)          │ │
│  │ ─ Bootstrap convergence: LLM sweep → ML → revisit      │ │
│  │ ─ Monte Carlo frontier + label propagation             │ │
│  │ ─ CatBoost fit-to-LLM + frontier SVM retraining        │ │
│  │ ─ SAGE + PermutationSHAP (GPU kernels)                 │ │
│  │ ─ Nautilus mid-run supervisor + Overwatch post-run     │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Governance (atelier.governance/)                       │ │
│  │ ─ Atlas client (HTTPBasicAuth via CDPUrlResolver)      │ │
│  │ ─ Ranger client, auto-sync gated by config             │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────┬──────────────────────┬────────────────────┬──────┘
           │                      │                    │
┌──────────▼─────┐  ┌────────────▼──────┐  ┌──────────▼────────┐
│  PostgreSQL    │  │  Qdrant           │  │  CAI Data         │
│  (PGlite       │  │  (embedded        │  │  Connections      │
│   embedded for │  │   binary for CAI) │  │  ─ Hive           │
│   CAI; pg5533  │  │                   │  │  ─ S3 (Object)    │
│   for devenv)  │  │                   │  │  ─ Atlas, Ranger  │
└────────────────┘  └───────────────────┘  │  ─ MLflow, cmlapi │
                                           └───────────────────┘
```

**Deployment tiers**: devenv (Nix) → `just` (portable task runner) →
plain Python/bash scripts (CAI AMP, no devenv or just available).

## Current state (2026-04-21 end of day)

- **UAT reproduction**: in-flight on CAI with 38 vCPU / 153 GiB /
  A10G GPU profile, auto-started against `hive-poc/default`.
- **OOTB reference parquet bundle**: in progress — a pre-classified
  parquet will ship under `data/sample/atelier_embeddings.parquet` so
  first-time users can explore the Embeddings page before their first
  run completes. Landing card wiring in the same PR.
- **Design invariants**: enforced at pipeline entry
  (`max_iterations >= 2`, `fit_to_llm = true`) with project-memory
  directive.
- **Sweep deadline**: disabled by default; opt-in via
  `ATELIER_BOOTSTRAP_SWEEP_DEADLINE_S`. Consecutive-failure breaker
  (LS-4) and attempts cap are the primary brakes against a blackholed
  endpoint.

## Known limitations / open questions

- **CAI pod memory** on the default small profile cannot hold the full
  ~700-column sweep; requires a scaled profile. Today's resolution is
  operator-scaled pod; a future `CAI_LEAN_MODE` preset is possible if
  smaller profiles become a requirement.
- **Tables-discovered vs Hue**: the `SHOW TABLES` service-principal view
  can show more tables than Hue's user-scoped view (stale cache or
  Ranger filtering). Investigated for `hive-poc/default` and confirmed
  benign (7 `*2` UAT variant tables that Hue had not refreshed).
- **Landing-page Entities**: 0 mid-run is expected — the `datasets`
  row is registered at run completion. Bundle PR (in progress) adds a
  reference row for the OOTB sample so the card is informative even
  before any run completes.

---

## Appendix A: complete release notes

All 219 commits in the 3-week window, grouped by calendar day (newest first). Commit hashes are on the `trunk` branch; resolve with `git show {hash}` in a checkout of [zndx/atelier](https://github.com/zndx/atelier).

### 2026-04-21

- `0c0170f` feat(pipeline): enforce design invariants at entry — max_iterations≥2, fit_to_llm=true
- `0285ded` fix(bootstrap): disable sweep_deadline_s by default — LS-4 is the real brake
- `fed0cf0` feat(bootstrap): surface LLM-sweep brakes in HOCON + Settings UI
- `e067be0` refactor(ui): move reference-column production-naming note to source comment
- `9fcb8a9` fix(bootstrap): hang-in-LLM_SWEEP remediation — heartbeats, attempts gate, deadline, breaker
- `fcc501b` fix(pipeline): auto-start stability — Anthropic timeouts, off-loop seeding, nautilus auto-cancel
- `14c6713` fix(config): strip whitespace from string config values
- `b004926` fix(pipeline): remove auto-start validation gate — fail honestly instead
- `4c7a400` fix(pipeline): auto-start connectivity gate + explicit Bedrock timeouts
- `4d9df76` fix(pipeline): recover from silent daemon-thread death + per-batch heartbeat
- `fec0ce7` feat(pipeline): operator-initiated cancel + env-seeded source race fix
- `0ab7f91` fix(cai): source-id dedup, post-filter table count, network-failure distinction, terminal header clear
- `febda64` fix(cai): install sops so encrypted deployment defaults actually load
- `6c666a6` fix(terminal): run 'chk' health check in-process, not via SDK
- `077a88a` ui(status): smoke-test staleness threshold 10min → 24h
- `b35c7f6` overwatch(prompt): asymmetric tolerance — configured contract, not symmetric band

### 2026-04-20

- `29512e6` overwatch(prompt): health as (target, tolerance) pairs, not binary 100%
- `d4829f4` refactor(overwatch): FAIR-aligned review prompts, no suppressive language
- `b02ac4d` test(classify): pin reference-column prefix list + drift-catching context
- `2854937` ui(status): informative caption for reference-column Include mode
- `b0fd685` feat(classify): reference-column exclusion toggle on Status page
- `7bf0b2c` refactor(classify): reference_code terminology + CAI LLM-coverage guarantees

### 2026-04-19

- `5f0895e` docs(notes): Phase Gate Report #2 — iterative revisit, reasoning capture, honest uncertainty
- `eb2b49a` feat(attribution): light-touch reasoning-trace citation analyzer
- `2477ddb` refactor(parity): rename ground_truth → curated_reference; Ontology-Annotation atlas label; terminology table
- `de1d638` feat(attribution): capture GLM-4.7 reasoning + prescriptive revisit prompt → +9 pts iterative gain
- `4ee1b22` feat(attribution): rescue LLM-abstention rows via CatBoost extrapolation
- `6362fd7` feat(attribution): SOTAB Schema.org pilot + iterative revisit driver
- `3e294b5` docs: post-hoc corrections reframing UAT labels as provisional
- `0959f67` feat(parity): authoritative ground truth + objective UAT vs Atelier delta
- `9ce5fab` feat(parity): add Arm J — train SVM on full 296-class vocabulary
- `7e44164` fix(eval): name-index derivation assigned parent codes to leaf columns
- `21457fb` feat(parity): add Crammer-Singer SVM arms (H, I) to the ablation
- `46e6f0e` feat(parity): extend SVM ablation to 7 arms — naive stop-word removal hurts
- `2db930d` feat(parity): generator for leak-sanitized Hive-import dataset
- `53f3d8d` fix(eval): sanitize GT leakage in meta-tagging source + SVM ablation + ML-only reproducibility driver
- `1b2d2c2` feat(reproducibility): persist fit-to-LLM CatBoost alongside the run parquet
- `f523f50` feat(parity): Phase-1 do-no-harm driver + authoritative UAT parity run
- `4cd2700` feat: prefer in-repo UAT snapshot for meta-tagging; tolerate Hive-export headers

### 2026-04-18

- `fd5c86c` feat: Web Terminal readline-style line editor with arrow-key history
- `15d680d` fix: Web Terminal provider routing + auto-start wiring + Opus-only catalog
- `bb65364` fix+feat: classify source layering + Status page UX polish
- `302b512` feat: Web Terminal Agent panel — pick model/provider with live TTFT + tok/s
- `efa7373` fix: ground truth loader resolves annotation mnemonics via vocabulary
- `64c5cd1` chore: swap Sonnet 4 → 4.5 across all three Sonnet-slot defaults
- `d97e1fd` chore: bake Opus/Sonnet/Haiku model ARNs + overwatch settings into .env.cai.enc
- `b51dd47` chore: ship meta-tagging ground-truth fixture (SOPS-encrypted)
- `263acae` feat: SOPS-delivered ground truth + shrink AMP config to 4 required vars
- `e3ddaa4` feat: ATELIER_CLASSIFY_MODEL alias surfaces classify LLM on AMP config page
- `22442b1` feat: pipeline self-remediation — halving retry, nautilus, supervisor overwatch, GPU
- `3d87c95` fix: AMP metadata env defaults were overriding HOCON thesis defaults
- `a905f37` feat: bake-everything-in for fresh CAI deployment

### 2026-04-17

- `7313d13` fix: SVM retrain was wiping the fit-to-llm CatBoost install
- `bb80212` fix: default llm_columns_per_call 50 → 25 (truncation guard)
- `382172d` fix: .claude/settings.json so project slash commands resolve on CAI
- `8968ff2` fix: terminal — chk went silent on Bedrock after the first query
- `cc68385` feat: exclude vocab tables + predicted_annotation + thesis-aligned defaults
- `1edbb92` fix: anthropic credential validator was using Bedrock ARN as the model
- `3eb5d2f` feat: CAI Data Platform — unified dropdown with file:// scheme
- `be5333d` fix: work around SDK v0.1.56 bug mapping thinking=adaptive
- `28667f1` fix: Opus 4.7 thinking format — model-aware adaptive vs legacy
- `f4ef910` feat: phase-gate validation — 97.8% on meta-tagging (beats LLM baseline)
- `2bac7bc` feat: CatBoost fit-to-LLM mode — training surface agrees with the oracle
- `a579c03` feat: meta-tagging source — local private-dir mount
- `b6527c6` feat: 100% LLM coverage + pattern quarantine for phase-gate thesis
- `4b75616` docs: UAT acceptance review — 36% vs 93% third-party baseline
- `5abf3b6` docs: phase 3 caption polish — Endsley L3 projection over L2 restatement
- `ed4ba2a` feat: settings page phase 2 — adaptive focus + per-run snapshot
- `f16ecac` feat: settings page phase 1 — full parameter surface (46 controls, 5 tabs)
- `0cfa899` docs: operator guide for SOPS+age encrypted deployment defaults
- `23d94bd` feat: overwatch enabled by default when ANTHROPIC_API_KEY is set
- `cf940ed` feat: bootstrap coverage target → 100% default
- `0328409` fix: reasoning_budget opt-in + auto-strip when OpenAI-compat backend rejects it
- `153081b` feat: settings page, synthetic source, GPU acceleration, log-health BDD
- `cabdaa5` feat: configurable Yager fusion strategy alongside Dempster
- `9703474` docs: replace K-centric convergence language with belief-gap
- `078c84d` feat: categorical enum detection + sibling domain clustering
- `a470ddd` feat: pipeline hardening — 8 new patterns, abbreviations, LLM aliases
- `f5075e9` feat: overwatch report viewer with markdown rendering + feedback

### 2026-04-16

- `61c05ca` feat: overwatch agent loop — pipeline analysis + recommendations
- `3cb7b9f` feat: overwatch foundation — native CLI install, config, GitHub App
- `714796f` fix: remove silent max_tokens clamp — let Bedrock errors propagate
- `94dcdbd` fix: Bedrock 65536 max_tokens limit + SDK session continuity error
- `1387961` fix: pipeline uses source's connection/database, not HOCON defaults
- `e1b21b9` chore: update uv.lock for cdpcurl dependency
- `69ac244` feat: URL normalization in CDPUrlResolver + CDP discovery endpoint
- `eba8634` feat: CDP control plane discovery via cdpcurl + S3 in health-check
- `a41b4e3` feat: health-check probes S3 Object Store via CAI Data Connections
- `46107a6` fix: governance SDK production audit — 7 issues remediated
- `731363a` feat: governance SDK integration — Atlas as system of record
- `9cec27f` feat: add governance package (Atlas + Ranger clients) and doc notes
- `e817207` feat: health-check discovers sibling AI Studio AMPs
- `fd2810f` feat: health-check skill uses parallel subagents for discovery
- `c0d2332` feat: health-check probes full CAI ML platform (cmlapi + MLflow)
- `9111231` feat: conversation continuity + 128K max output tokens
- `70f8ef4` feat: /health-check skill with `chk` terminal alias
- `d937c79` feat: SOPS+age encrypted CAI deployment defaults
- `6cd89a2` chore: version bump script + fix missed 0.1.0 references
- `c0ff5cc` chore: remove R-MDMC-Grok-Review.md from repo root
- `1f0a3f8` feat: v0.2.0 — vocabulary routing, LLM robustness, rich terminal, UX polish

### 2026-04-15

- `1b53665` fix: retry OOTB sample seed on DB startup race
- `cc4db75` docs: comprehensive rewrite — belief-gap convergence, pattern audit, terminal sessions
- `4ebad6a` feat: belief-gap convergence replaces K as primary convergence measure
- `cc9acf7` feat: graduated pattern mass, confusable pairs, pattern theta 0.25
- `410ce39` fix: pattern detection audit — validators, Luhn, IPv4, date, currency
- `063e3a0` fix: phone pattern false positives inflating DST conflict across all columns
- `872092d` fix: truncate Bedrock ARN in Configuration panel, show full ARN on hover
- `eef6063` fix: SVM drops singleton classes instead of crashing on StratifiedKFold
- `47c32bf` fix: Bedrock runtime — tool-use structured output, ARN-aware regions, fail-fast sweep
- `735c9e8` feat: persistent terminal sessions with ring buffer replay
- `cfbba85` feat: auto-discover Hive annotation sources on gateway startup
- `695298d` docs: reconcile counts across classification, data-sources, scenarios
- `8cacc32` fix: useRef requires explicit initial value in React 19 types
- `90df396` docs: comprehensive introduction rewrite with verified numbers + MathJax
- `d5e557e` fix: fold db-bootstrap into grpc-server startup
- `e76a89b` fix: make all SQL migrations idempotent (IF NOT EXISTS)
- `500717f` feat: frontier-label SVM training via R-MDMC sampling (M9)
- `8d3cd19` feat: SVM signals alignment, reasoning budget, db bootstrap, docling

### 2026-04-14

- `1d2d33e` feat: archive/unarchive for data sources and datasets + onboarding BDD
- `c025d60` fix: FSM same-state transition, Embeddings 500, Vite proxy noise
- `55e2187` feat: R-MDMC pilot — row-level Monte Carlo sampling
- `1cbe976` docs: comprehensive documentation overhaul
- `772e58b` feat: Monte Carlo sampling for scalable classification
- `e664dc9` feat: GPU-accelerated embeddings via NVIDIA driver symlink pattern
- `3ecf162` fix: agent loop post-validation fixes from live Cerebras+Sonnet run
- `744d3b6` feat: agent-driven classification convergence loop
- `c863b38` feat: synth framework, meta-tagging overlay, gateway integration, agent config
- `ec9b7f8` feat: propagate source_id through FSM run lifecycle
- `1df309a` fix: source-aware dataset registration + audit remediation
- `18c5e09` docs: data sources architecture + proposed MLflow/Hive integrations
- `e44ce06` feat: source-aware pipeline routing with OOTB sample auto-import
- `14182f5` feat: generate 25 mixed-domain sample tables with opaque column names
- `0c8655d` feat: expand vocabulary to 300 BFO-grounded leaves via CCO trichotomy
- `0bc7035` feat: data source + dataset versioning model for OOTB onboarding
- `de6abd9` feat: CCO-mediated BFO alignment for classification vocabulary
- `a885c8c` refactor: eliminate use_mock, single pipeline code path
- `df7f1af` feat: BFO-grounded universal vocabulary with audit hardening
- `81cb65b` docs: Atlas/BFO reorientation research and hierarchy notation options
- `89d7460` fix: normalize Hive column keys, validate vocabulary cache, add /api/vocabulary/stats
- `e51f37a` feat: true cardinality via COUNT(DISTINCT), embedding_text in parquet, conflict coloring

### 2026-04-13

- `c0695b3` feat: expose all operator-facing config in AMP metadata
- `a5c8cc7` fix: add missing operator-facing env vars to AMP metadata
- `6d2f458` fix: PGlite version bump + maxConnections for CAI stability
- `34e9af8` docs: work notes for classification pipeline and LLM backend sessions
- `bb5f419` feat: unified classification pipeline with structured output LLM backends
- `3d6fcc9` fix: show only active dataset in Embeddings card on landing page
- `8620f8f` feat: seed keystone agents migration + global dataset selector
- `aaac1b8` fix: retry SDK smoke test on cold-start failure
- `a6187d5` fix: add libpq to devenv LD_LIBRARY_PATH, atlas-compatible parquet output
- `047bf40` fix: make Embeddings card navigable with /embeddings index route
- `2a95c9c` fix: PGlite wedge, DAO pool resilience, dataset auto-registration, @slow tags
- `23f7f47` feat: real data validation baseline, deprecated term handling, architecture one-pager (M5)
- `48566e0` feat: SHAP explanations, configurable discounts, thread-safe models (M4)
- `7abb0c9` feat: E2E validation framework, SAGE feature importance (M3)
- `37f16aa` ci: add GitHub Actions workflow for mdbook docs deployment
- `b2123bf` feat: ML classifiers, synthetic data, and backend abstraction (M2)
- `d46f25a` feat: LLM bootstrap convergence loop (M1)

### 2026-04-12

- `681225a` fix: audit remediation — schema inversion, pignistic probability, HierarchicalClassification
- `d247c8c` feat: Dempster-Shafer classification pipeline (M0)
- `7d8d007` fix: proof-of-progress PGlite readiness + gateway PG probe retry
- `441d5fe` feat: animated braille spinner while SDK query is thinking

### 2026-04-11

- `994f8c8` feat: web terminal pause/redirect via Ctrl-C interrupt
- `55a2fad` feat: CAI Data Platform connection testing
- `fd8bcde` fix: surface real CLI stderr + tri-state Service Status card
- `8eb4636` fix: web terminal — grant tool permissions + handle multi-line paste
- `9b2bd0d` refactor: promote CLI feature flags to HOCON-bound config
- `9763438` fix: pin Bedrock sub-model overrides + disable experimental betas
- `0ae92e2` fix: Bedrock auth flag + JSON error envelopes for robust CAI runtime

### 2026-04-10

- `4461151` fix: prevent gateway crash from blocked event loop + install [agents] extra
- `c212107` fix: harden CAI deploy path — SDK skill discovery, seed idempotency, build heap
- `b23a832` fix: symlink claude CLI to ~/.local/bin on CAI
- `1e369b1` feat: keystone agents + skills + CAI deployment hardening

### 2026-04-09

- `23817e8` feat: XYFlow orchestration canvas with keystone agent topology
- `1fc3a16` fix: use TCP port check for PGlite health instead of psycopg
- `c1caf6a` fix: drop psycopg[binary] to eliminate C extension segfault on CAI
- `e0dc555` fix: harden CAI deployment for clean CI re-deployment
- `17bf2ed` fix: kill orphaned PGlite before restart to prevent EADDRINUSE
- `2df19c0` fix: use port 5440 for PGlite to avoid CAI platform Postgres on 5432

### 2026-04-08

- `2ab245a` fix: start PGlite/Qdrant before config resolution in startup
- `bf0e176` fix: use SQLAlchemy bootstrap for migrations instead of dbmate CLI
- `1636016` fix: use tarball for embedding-atlas to avoid symlink resolution issue
- `1f3fe69` fix: harden load_nvm.sh for CAI runtimes
- `5610927` fix: pre-built embedding-atlas dist for CAI deployment

### 2026-04-07

- `668a5e4` docs: session notes for preflight, model discovery, embedded terminal
- `7cddeef` feat: embedded terminal with ghostty-web WASM and Claude Agent SDK
- `a8bfc58` feat: model discovery, rename Embeddings Viewer to Embeddings, default to Opus
- `7656d69` feat: preflight checks, devenv hardening, config materialization
- `d279dc4` fix: handle nested event loop in SDK smoke test
- `05f1c2c` feat: Agent SDK integration with Bedrock provider support, Embeddings Viewer polish

### 2026-04-06

- `8d08f32` fix: DAO returns dicts to avoid detached session errors, automate AMP setup
- `af0062f` feat: integrate Embeddings Viewer with GitTables dataset pipeline
- `590878a` feat: BDD scaffolding + scenario-oriented documentation
- `6bb12a1` revert: remove protobuf <6 cap (proto stubs need 6.x runtime)
- `f2d0efe` fix: pin pglite 0.3.8 to match pglite-socket 0.0.13 peer dep
- `f5b60e2` fix(cai): cap protobuf <6 to coexist with CML system packages
- `2c60e8e` fix: correct pglite-socket version (0.0.13, not 0.4.0)
- `0a585e9` chore: add ripgrep to devenv, nix-in-CML roadmap notes
- `78cc575` fix(cai): install Node.js via nvm (RAG Studio pattern)
- `23d8e96` feat: replace pgserver with PGlite for CAI embedded PostgreSQL
- `9fc1f8d` fix(cai): target Python 3.12 runtime (pgserver lacks 3.13 wheels)
- `5c91263` feat(cai): adopt AMP create_job/run_job pattern
- `25a7e48` fix(cai): install into system python instead of virtualenv
- `f759d5a` Add self-healing install fallback to start-app.sh
- `9cf5966` Fix CAI startup: activate venv, use python directly
- `2aa7654` Fix uv not on PATH in CAI application sessions
- `72b218d` Update CLAUDE.md: CAI/CML/CDSW naming guide, devenv-first, full stack
- `879e437` Rewrite README as devenv-first with clear local vs CML separation
- `e20afcd` Update README with full local dev and CML deployment procedures
- `cec9635` Fix CAI deployment issues: HTTPS submodules, inline migrations
- `f7f6e40` Add Qdrant binary deployment for CAI (RAG Studio pattern)
- `97cac28` Add PostgreSQL infrastructure with pgserver for CAI
- `13f0890` Add documentation, CLAUDE.md, and update README with deployment guide
- `7988f53` Add CAI deployment files and task runner
- `83a4783` Add React frontend with Vite, Ant Design, and XYFlow
- `d81f4b8` Add gRPC service, gateway, and data layer
- `9c500b8` Add Python project and HOCON configuration system
- `6d39645` Configure devenv with Python 3.12, Node.js 22, and process management
- `5e2800a` Add git submodules for embedding-atlas and hermes-agent

### 2026-04-05

- `08eaaac` Initial commit

---

*Commit counts* (conventional-commit prefix): **93 feat**, **75 fix**, **14 docs**, **7 chore**, **6 refactor**, **2 ui**, **2 overwatch**, **1 test**, **1 revert**, **1 ci**, plus 17 initial-scaffolding commits without prefix = **219 total**.

*Generated 2026-04-21; source-of-truth is `git log` on the `trunk` branch. To reproduce:* `git log --since="2026-03-31" --pretty=format:"%h %ad %s" --date=short`.
