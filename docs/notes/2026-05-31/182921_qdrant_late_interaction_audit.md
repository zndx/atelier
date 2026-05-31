# Qdrant late-interaction audit — "cosine" vs MaxSim

**Date:** 2026-05-31
**Scope:** `colbert_encoder.py`, `late_interaction_bridge.py`, `enrichment/qdrant_writer.py`,
`mass_functions.py` (`late_interaction_to_mass`, `_cosine_reliability`), `enrichment/loop.py`,
pipeline cosine-channel wiring.
**Trigger:** terminology question — "cosine" language where ColBERT actually uses MaxSim.

## Architecture as verified (correct)

- **Encoder** (`colbert_encoder.py`): BERT backbone + 768→128 `linear.weight` projection.
  Per-token vectors are **L2-normalized** (`F.normalize(projected, p=2, dim=-1)`, line 115).
  Special tokens stripped; content tokens only.
- **Both sides use the same encoder** — entity (query) = `ColumnFeatures.to_embedding_text()`,
  annotation (doc) = `compose_annotation_text()`.
- **Collection** (`qdrant_writer.ensure_collection`): single `colbert` multivector field,
  `distance="Cosine"` (default, never overridden — only call site is `loop.py:177`),
  `MultiVectorComparator.MAX_SIM`.
- **Query**: `query_points(query=token_vectors, using="colbert")` — Qdrant computes MaxSim natively.
- **Fail-fast**: `LateInteractionUnavailable` on every degraded path; only operator-explicit
  disable returns `None`. Legacy single-vector `cosine_to_mass` removed 2026-05-25; no silent
  fallback. Matches the no-silent-DST-degradation rule.

## The terminology question (headline)

**MaxSim ≠ cosine.** MaxSim = `Σ_q maxᵈ ⟨q_tok, d_tok⟩`. The *per-token* comparison is cosine
(Qdrant Distance=Cosine + vectors L2-normalized), but the *aggregate* is a sum of max
token-cosines — a different quantity from a single-vector cosine similarity. Calling the result
"cosine mass" / `try_compute_cosine_mass` / "cosine evidence source" conflates the aggregate
operator with its per-token metric.

**Why it persists:** the DST evidence *slot* was historically single-vector cosine (MiniLM).
Late-interaction replaced the *implementation* but kept the *slot name* "cosine" throughout:
module docstring, `try_compute_cosine_mass`, `_cosine_reliability`, config namespace
`classify.cosine.*`, `classify_cosine_union_focal_*`. So "cosine" is now an *evidence-channel
label*, not a description of the math.

**Where it is accurate:** the per-token metric genuinely is cosine, and the normalized score
(MaxSim / n_query_tokens) is a *mean per-token max-cosine* — cosine-like, ~[0,1]. The
MaxSim-specific spots are already careful (`late_interaction_to_mass` docstring "Convert ColBERT
MaxSim scores", attribution `maxsim_score`, `ranking_basis: qdrant_maxsim`). The drift is only at
the channel-level names.

**Recommendation:** pick one convention.
- (a) Rename the channel `cosine` → `late_interaction` and reserve "cosine" for the retired
  single-vector path. Cleaner, but touches operator-visible config keys (`classify.cosine.*`,
  `ATELIER_*`) — a breaking rename of the same flavor as the "frontier" cleanup.
- (b) Keep "cosine" as the channel name and document, once, that its current implementation is
  ColBERT MaxSim with a cosine per-token metric. Cheaper, roll-forward-friendly.

## Correctness findings

### F1 (substantive — wants validation): reliability sigmoid calibrated for the wrong scale
`_cosine_reliability` centers at `tau_abs=0.40, sigma_abs=0.10`; its docstring explicitly cites
"sentence-transformer embeddings" (~0.30 noise, ~0.50 clear match) — i.e. **MiniLM single-vector
cosine**. The input is now **mean-per-token-max-cosine from ColBERT MaxSim**, a differently
distributed quantity: each query token almost always finds a well-aligned doc token, so even
weakly-related pairs score high and compressed. Likely effect: `alpha_abs` saturates near the
`ceiling=0.70`, weakening the "is this noise?" discrimination the sigmoid exists to provide.
The bridge comment claiming the `/n_query_tokens` normalization "recovers the scale
`_cosine_reliability`'s sigmoid was calibrated for" papers over the encoder/metric change.
→ Recalibrate `tau_abs`/`sigma_abs` against the empirical ColBERT-MaxSim score distribution on
UAT data (this is exactly an `optimize/calibration/sweep.py` job), or confirm the distributions
overlap before trusting the current centers. Tuning concern, not a crash.

### F2 (minor — comment accuracy): the "[0,1]" normalization claim
Bridge lines 296–299 assert normalized score ∈ [0,1]. With L2-normalized vectors + Cosine,
per-token max-cosine ∈ [−1,1], so MaxSim/n_q ∈ [−1,1] theoretically; in practice max-over-doc
is ~always positive, landing in (0,1]. Downstream sigmoid handles negatives gracefully (low α,
no crash). Fix the comment to "[−1,1] theoretical, ~[0,1] practical" or clamp explicitly.

### F3 (minor — robustness/clarity): double normalization
Encoder L2-normalizes AND collection uses Distance.COSINE (Qdrant normalizes again). Harmless
(idempotent) but the "Cosine" choice does nothing the encoder didn't; DOT would be equivalent.
Worth a one-line comment at the `ensure_collection` distance arg so a future reader doesn't
remove the encoder normalization assuming Qdrant covers it (or vice versa).

### F4 (observation, supports F1): the divisor is monotonic for ranking
`/n_query_tokens` uses query-token count, constant across the candidate set for a single column
query → it's a pure rescale that does **not** change ranking, only the absolute scale fed to the
sigmoid. (Good: doc length doesn't inflate MaxSim either, since it's max-over-doc, not sum.) So
the divisor exists *solely* to hit the sigmoid's expected scale — which reinforces F1 as the one
place this matters.

## Bottom line
Implementation is sound and the fail-fast posture is correct. The "cosine" naming is a channel
label inherited from the retired single-vector source, not a math error — but it is pervasive and
worth resolving one way or the other (F1 recommendation a/b). The one finding with operational
teeth is **F1**: the reliability sigmoid's calibration targets MiniLM cosine and is now fed
ColBERT MaxSim-derived scores; validate/recalibrate against UAT score distributions before
treating the discount as well-tuned.
