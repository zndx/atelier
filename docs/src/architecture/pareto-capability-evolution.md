<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Pareto Capability Evolution (Roadmap)

> **Status**: research-shaped capstone milestone. No incremental rollout —
> we ship it whole when the pieces converge.

This document proposes a long-horizon evolution of the Atelier
classification pipeline from a single-config bootstrap loop into a
**multi-objective, population-based search** over the policy space (LLM
prompts, classifier hyperparameters, fusion strategy). The framing is
rooted in three bodies of work — **Active Learning**, **Automatic
Prompt Optimization (APO)**, and **GEPA** — each of which already maps
cleanly onto a piece of what we ship today.

## Why this is a capstone, not a feature

The current bootstrap loop is *already* an active-learning system, just
informally named. We sweep with an Opus oracle, fuse with Dempster-
Shafer, revisit disagreements, retrain incrementally — all under a
single configuration. Operators have started asking the next question:
*could we have run with a tighter belief gap, fewer LLM tokens, deeper
cautious predictions?* Each answer requires re-running with different
settings. We need a search procedure that can carry this load without
forcing operators to hand-tune one knob at a time.

We ship this when the prerequisite pieces converge:

- The reasoning model in `overwatch/agent.py` stabilizes as a reliable
  proposer of *structured* configuration edits (prompt diffs and JSON
  patches over the config tree, not free-form advice).
- We have enough corpus diversity in `data_sources` to evaluate
  candidates against generalization, not point estimates on one source.
- A persistent population store (the *config leaderboard*) is in place
  so evolution survives gateway restarts and CAI session boundaries.

Until those land, individual ideas in this doc may be borrowed in
isolation (e.g. an APO-only loop that evolves a single sweep prompt
against accuracy).  The **capstone is the integrated whole** — the
borrowed pieces alone do not constitute "Pareto Capability Evolution".

## Foundations

### Active Learning — the paradigm we already implement

Active learning minimizes label cost by querying an oracle on examples
the model is most uncertain about (Settles 2009).  Mapped onto the
Atelier pipeline:

| Active Learning concept | Atelier component |
|---|---|
| Oracle | Opus during sweep + revisit (`pipeline.py::_llm_sweep`, `_llm_revisit`) |
| Labeled pool (T_K) | Synth corpus + curated reference + accumulated LLM labels |
| Unlabeled pool (T_U) | Discovered source columns awaiting classification |
| Query strategy | Belief-gap-driven revisit selection (largest `Pl − Bel`) |
| Query-by-committee | Disagreement between CatBoost-fit-to-LLM and the synth-trained SVM (via the ICE→user alignment) |
| Pool vs. stream | Pool-based — Monte Carlo stratification picks each batch |
| Stopping criterion | `mean_gap < gap_threshold` OR `max_iterations` reached |
| Cold-start mitigation | Synth pre-training + pattern evidence on first sweep |

The active-learning incorporation of new oracle labels is
concentrated in the ``catboost`` source (``fit_to_llm`` mode trains
on the live LLM labels mid-run).  The SVM was previously also part
of this active-learning loop via the M9 ``frontier_svm`` retrain,
but that path was excised on 2026-05-04 (commits 8627c2c, 5199379,
cc59d01) for the independence reasons documented in
``ontology_alignment.py``.  The SVM now contributes a label-stable
TF-IDF view that complements the live-LLM-aligned CatBoost view.

### Automatic Prompt Optimization — APO and GEPA

Both APO (Microsoft Agent Lightning) and GEPA (Lakhotia et al., ICLR
2026) optimize LLM prompts via *reflection-driven mutation*: the LLM
diagnoses its own failures in natural language and proposes prompt
edits, evaluated against held-out tasks.  They differ on search shape:

| Dimension | APO | GEPA |
|---|---|---|
| Search structure | Beam (default width 4) | Pareto frontier (open-ended population) |
| Objective | Single scalar reward | Multi-objective, non-dominated sorting |
| Mutation | Textual gradient → LLM-edit | LLM reflection + cross-candidate recombination |
| Targets | One prompt template at a time | One or more prompts; full system policy |
| Scope | "Pick the best system prompt" | "Discover diverse strategies and combine them" |
| Sample efficiency | Not benchmarked vs. RL | 35× fewer rollouts than GRPO; +6–20% over MIPROv2 |

For Atelier, **APO is the right shape for narrow optimizations** (tune
*one* sweep prompt against accuracy on a known corpus).  **GEPA is the
right shape for the capstone**: we have multiple operator-relevant
objectives (accuracy, calibration, cost, coverage, latency), and we
benefit from preserving complementary policies rather than collapsing
to a single configuration.

We treat APO and GEPA as **peer techniques**.  APO is invoked when one
objective clearly dominates and beam search is sufficient; GEPA is
invoked when objectives trade off and the frontier's diversity is
itself the asset.  Both share the same reflection-engine plumbing.

### The synthesis — Pareto Capability Evolution

The capstone integrates AL, APO/GEPA, and population-based search into
one loop:

1. **Active learning** drives label acquisition *within* each candidate
   run (the existing bootstrap loop, unchanged).
2. **Reflection-driven mutation** drives *proposal* of new pipeline
   configurations: prompt edits, classifier knobs, fusion swaps.
3. **Pareto sorting** decides which configurations survive into the
   next generation.

The reflection model is the same Opus instance already wired for
overwatch — it reads the convergence report of a finished run and
proposes targeted edits to the configuration that produced it.

## Pipeline policy space

Mutation targets the configuration tuple, not just the prompt:

- **LLM prompts**: sweep template, revisit template, classification
  subagent system prompt.
- **Classifier hyperparameters**: CatBoost depth, learning rate, class
  weights; SVM C and kernel; SVM-vs-LLM blend ratio in DST mass
  construction.
- **Fusion strategy**: Dempster vs. Yager; gap threshold; bel-floor;
  pignistic vs. cautious decision rule; cautious depth threshold.
- **Search budget**: sweep batch size, max bootstrap iterations, Monte
  Carlo stratification fraction, revisit triggers.
- **Pattern evidence weights**: per-pattern mass discount, evidence
  layering order.
- **Embedding choice**: MiniLM-L6 (today's default) vs. BGE-large vs.
  E5-mistral — bounded by the embedding-model identity check we
  already enforce on Extend runs.

Hard invariants encoded elsewhere (e.g.
`classify.bootstrap.max_iterations >= 2`,
`classify.catboost.fit_to_llm = true`) remain non-negotiable —
mutations that violate them are rejected before evaluation, never
committed to the population.

## Objectives (Pareto axes)

| Objective | Source | Direction | Why operators care |
|---|---|---|---|
| Mean Bel of correct prediction | curated reference | maximize | core accuracy |
| Mean Pl − Bel | EVALUATING report | minimize | calibration tightness |
| LLM tokens / converged column | sweep accounting | minimize | governance budget |
| Cautious accuracy @ depth-N | `epistemic_evaluation` | maximize | hierarchy faithfulness |
| Vocab coverage @ τ | `classifications.json` | maximize | "did we touch every leaf?" |
| Pipeline duration | `fsm_runs.{started_at, updated_at}` | minimize | iteration speed |

A configuration enters the frontier if no other configuration beats it
on *every* axis (non-dominated sorting).  The frontier is open-ended in
size; crowding-distance pruning bounds it under operator-defined caps.

## Population store ("config leaderboard")

A persistent backing store records:

- Each evaluated configuration as a row, keyed by hash of the config
  tuple (bit-stable across host reboots).
- Every objective score per evaluation, with provenance back to the
  `fsm_runs.id` that produced it.
- Lineage edges: which configuration mutated to which, via what
  proposer (APO-style critic vs. GEPA-style recombiner) and what diff.
- Frontier membership over time, so an operator can see which
  configurations entered, dominated others, or were pruned.

This is conceptually a **leaderboard** — operators sort and filter by
any axis or weighted combination — and structurally a write-once
registry that supports re-evaluation as new corpora arrive.  A frontier
that holds against corpus A may not hold against corpus B; the registry
preserves both views without conflating them.

The store interfaces with existing tables: it points at
`ml_artifact_sets` rows (the bundle a winning config produced still
ships through Extend Classification) and `fsm_runs` rows (each
evaluation is one FSM run).  It does **not** duplicate them — there is
one source of truth for artifacts, and the leaderboard layers
search-state on top.

## Reflection loop — concrete shape

Per generation:

1. **Sample a parent** from the current frontier, weighted by either
   crowding distance (favor diversity) or recency (favor live operator
   priorities).  A small fraction of generations sample a *dominated*
   ancestor instead, to escape local frontier traps.
2. **Diagnose** by feeding the parent's run report to the reflection
   model.  The report includes the final classifications, per-axis
   objective scores, the convergence trace, and any cautious-review
   findings.
3. **Propose edits** as a structured patch (JSON) against the
   configuration tuple — e.g. `{"classify.bootstrap.gap_threshold":
   0.05, "classify.svm.blend_ratio": 0.6}` — or a textual prompt diff
   when the target is a prompt template.
4. **Evaluate** by instantiating the patched config, running it as an
   FSM run, and recording scores into the leaderboard.
5. **Update the frontier** via non-dominated sort; admit the new
   configuration if it is non-dominated; prune incumbents whose
   crowding distance falls below a threshold.

Mutation diversity is encouraged via **dual proposers**: one focused on
accuracy/calibration (the reflection model with a "be conservative"
system prompt), one focused on cost/latency (the same model with an
"aggressively shrink the budget" system prompt).  The frontier
preserves both styles rather than collapsing to whichever proposer
happened to find an early local optimum.

## What this retires

- **"Frontier SVM" terminology** *and the M9 retrain it described*.
  The mid-loop ``train_svm_on_frontier_labels`` retrain that gave the
  "frontier SVM" its name was excised on 2026-05-04 (commits 8627c2c,
  5199379, cc59d01) for the source-independence reasons documented
  in `ontology_alignment.py`.  The SVM is now trained once on synth
  with ICE.* labels and translated into the user vocabulary at
  inference time via the LLM-mediated alignment.  "Frontier" the
  word is freed for the Pareto sense used elsewhere in this doc.
- **Single-config tuning by hand**.  Today operators tweak `base.conf`
  or the runtime overlay and re-run.  The capstone replaces that loop
  with population-based search; the overlay UI surfaces *frontier
  picks* and lets operators promote one to active rather than asking
  them to choose individual values.
- **The "single best" mental model**.  Operators learn to think in
  trade-offs: "the accuracy-leader spends 4× tokens; the budget-leader
  loses 6 points of cautious accuracy at depth-3" — and the system
  surfaces both rather than averaging them away.

## Non-goals (explicitly deferred)

- **Multi-tenant scheduling under CAI quotas**.  The search loop assumes
  single-tenant compute on the host's GPU.  Quota-aware scheduling is
  a separate concern.
- **Cross-corpus warm-start**.  A frontier from corpus A is not
  automatically transplanted to corpus B.  The leaderboard preserves
  both, but transfer learning across taxonomies is research in its
  own right.
- **Re-training the embedding model in-loop**.  Embedding-model
  identity is locked per artifact set (already enforced for Extend
  runs); evolution can *swap* the embedding model only by spinning up
  a fresh population, not via mutation within an existing one.
- **Online / streaming evaluation**.  Pool-based AL is the operating
  mode.  Streaming evaluation as columns arrive continuously is a
  candidate for v2 — the leaderboard would persist while the pool
  grows.

## Open research questions

- **Cold start for the proposer**.  The reflection model needs at
  least one finished run before it can propose edits.  Bootstrap with
  N random perturbations of the default config?  Use APO-style beam
  search for the first generation, then expand into Pareto?
- **Noisy oracle problem**.  AL assumes the oracle is roughly correct.
  Opus is excellent but not infallible.  The cautious-review pass
  catches some errors, but whether the leaderboard should down-weight
  configurations whose convergence relied on later-overturned LLM
  labels is open.
- **Convergence detection for the meta-loop**.  When does evolution
  stop?  Frontier-stability heuristics (no admissions in K generations)
  versus operator-driven termination versus budget-exhausted.
- **Reflection-model agreement**.  APO's textual-gradient critic and
  GEPA's recombination critic are both LLM-driven.  Do they propose
  meaningfully different edits, or do they collapse to the same
  suggestion?  Worth empirical study before committing the
  architecture.
- **Reproducibility under stochastic LLM outputs**.  Two evaluations
  of the same config can disagree on objective scores.  How much
  smoothing (multi-seed averaging) is required before non-dominated
  sorting becomes stable?

## Cross-references

- [Classification Pipeline](./classification.md) — the AL loop being
  generalized.
- [Synthetic Data & Training](./synth.md) — synth provides the
  labeled-pool floor.
- [ML Artifacts & Extend Classification](./ml-artifacts.md) — winning
  configurations produce artifact sets that flow through the existing
  Extend pipeline.
- [GPU Acceleration](./gpu.md) — population-based search amplifies the
  payoff of fast per-evaluation rollouts.
- [Proposed Integrations](./integrations.md) — neighboring roadmap
  items that may interact with the leaderboard surface.

## References

- Settles, B. (2009). *Active Learning Literature Survey*. Computer
  Sciences Technical Report 1648, University of Wisconsin–Madison.
- Lakhotia, K. *et al.* (2025). *GEPA: Genetic-Evolutionary Pareto-
  frontier Adaptation*. arXiv:2507.19457.  ICLR 2026 (Oral).
- Pryzant, R. *et al.* (2023). *Automatic Prompt Optimization with
  "Gradient Descent" and Beam Search*. arXiv:2305.03495.
- Microsoft Agent Lightning, *APO Algorithm Documentation*,
  https://microsoft.github.io/agent-lightning/latest/algorithm-zoo/apo/
