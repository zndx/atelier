<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Pareto Capability Evolution — milestone framing

Operator question that started the thread: "Why do I see 'SVM frontier'
in the status UI?  That doesn't make objective sense."

Followed by: explore the original intent, then sketch a roadmap entry
informed by GEPA (arXiv:2507.19457) and Agent Lightning's APO, plus
Active Learning as the conceptual base.

## What the SVM "frontier" actually is

`src/atelier/classify/ml_train.py:81-176` trains the SVM on the
*moving frontier of newest LLM labels* (`label_source in ("llm",
"llm_revisit")`) blended with synth for vocab coverage.  Retrained
three times during bootstrap (`pipeline.py:898, 1113, 1197`):

- post-sweep (so an SVM is available before first ML validation),
- iteratively (after each revisit batch, ≥10 new labels),
- final (one last retrain before convergence).

DST independence is preserved because the SVM lives on sparse TF-IDF
features and is trained on Opus-tier labels, while the LLM mass
function in DST fusion uses the Sonnet/Haiku subagent model.

So "frontier" here is *temporal* — the moving boundary of newest
oracle labels — not Pareto-frontier in the optimization sense.  The
naming has been mildly confusing because GEPA et al. use "frontier"
for the Pareto set of non-dominated candidates.

## Three external bodies of work

- **Active Learning** (Settles 2009) — pool-based, uncertainty-driven
  oracle queries.  Maps cleanly onto our existing bootstrap loop;
  we've been doing AL without saying so.
- **APO** (Microsoft Agent Lightning) — beam search (width 4) with
  textual-gradient critique → LLM edit.  Single scalar reward, no
  Pareto.
- **GEPA** (Lakhotia et al., ICLR 2026) — Pareto frontier of prompt
  variants, evolved via reflection + recombination.  Multi-objective.
  35× fewer rollouts than GRPO; +6-20% over MIPROv2.

GEPA is the right shape for our capstone: we have multiple
operator-relevant objectives that genuinely trade off (accuracy,
calibration, cost, coverage, latency).

## Decisions made this session

- **Capstone framing, no incremental rollout**.  We know where we're
  going; we ship it whole when the pieces converge.
- **APO kept as a peer reference** alongside GEPA.  APO is the right
  shape for narrow single-objective optimizations within the same
  reflection plumbing.
- **No vendor names in the doc** for the leaderboard backing store —
  framed as a generic *config leaderboard* concept.
- **Rename "frontier SVM" → "incremental SVM"** as a follow-up
  cosmetic change.  On-disk filenames stay (`svm_frontier.pkl`) for
  backward compatibility; only UI tooltips, docstrings, and log
  lines need touching.

## Artifacts written

- `docs/src/architecture/pareto-capability-evolution.md` — full
  capstone milestone doc (foundations, policy space, objectives,
  reflection loop, leaderboard, retirements, non-goals, open
  questions, cross-refs, references).
- `docs/src/SUMMARY.md` — entry added between ML Artifacts and
  Proposed Integrations.

## Follow-ups (not done this session)

- Glossary nudge in `docs/src/architecture/classification.md`
  clarifying that today's "frontier SVM" is the incremental SVM and
  the word "frontier" is being held for the future Pareto sense.
- UI tooltip rename in `ui/src/pages/Status.tsx` (the SVM chip in
  the ML Artifacts panel currently says "SVM frontier").
- Empirical study: do APO-style and GEPA-style critics propose
  meaningfully different edits given the same parent run, or do
  they collapse to the same suggestion?  Worth understanding before
  committing the dual-proposer design.
