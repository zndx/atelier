# Late-Interaction Cosine + DST Pipeline Audit

**Date**: 2026-05-19  
**Scope**: Commits 929e29e..b37ff3e (P1–P7 remediation arc, ~10 commits)  
**Reviewer**: Claude Code spot-check

---

## Summary

The remediation work is **architecturally sound and mechanically correct**
at the DST-theory level.  However, end-to-end verification against
known failure modes reveals a **critical integration defect**: the
late-interaction cosine source contributes **vacuous mass (Θ=1.0) on
every column** due to a code-namespace mismatch between the Qdrant
collection and the pipeline's FrameOfDiscernment.

**The entire late-interaction evidence source is inert in production.**
The pipeline reports `cosine_path="late_interaction"` and `status="ok"`
but the cosine channel contributes nothing to fusion.

---

## FINDING 0 (CRITICAL): Code namespace mismatch — cosine is dead

### Symptoms

- `late_interaction_to_mass` returns vacuous mass (Θ=1.0) on every column
- `channel_conflict_k=0.0` always (no non-Θ mass to conflict)
- `subtree_concentration=None` always (no top-1 leaf)
- No WARNING emitted — the bridge considers this a "success"
- Legacy single-vector fallback NOT triggered (bridge returns "ok")
- The pipeline runs with 5 evidence sources that WORK (pattern, name,
  LLM, CatBoost, SVM) plus 1 that silently contributes nothing (cosine)

### Root Cause

Two independent code namespaces with zero overlap:

| Component | Code format | Example | Count |
|---|---|---|---|
| Qdrant collection (`annotations_default_v1`) | Numeric dot-codes | `1.1.1.9.3.1` | 511 points (281 unique codes) |
| Pipeline `FrameOfDiscernment` (from live annotations table) | ICE.* hierarchical | `ICE.SENSITIVE.PID.CONTACT.EMAIL` | 316 leaves |

The enrichment script enriched the taxonomy loaded from
`hive-poc/default.annotations` which uses numeric dot-codes.  The
pipeline's `load_sample_vocabulary()` loads the ICE.* taxonomy.

In `_late_interaction_positive_mass`, the code path:
```python
if s.code in frame.singletons:      # "1.1.1.9.3.1" in {"ICE.SENSITIVE...": ...} → False
elif s.code in frame.internal_nodes:  # same → False
```
…rejects all 281 scored tags.  `in_frame` is empty → vacuous.

### Evidence of the Problem

The registry row confirms the last enrichment *failed*:
```
summary: 'enriched=0 cache_hit=0 verifier_failed=0 generator_failed=296'
```
The 511 points are from a prior build against the OLD taxonomy.  The
current ICE.* taxonomy has never been successfully enriched into Qdrant.

Even with a successful enrichment, the codes stored in Qdrant
(`payload['code']`) would need to match the frame's singleton keys.

### Mapping Potential

- **Mnemonic → Abbrev**: Only 20/234 Qdrant mnemonics match frame abbrevs
  (the two taxonomies diverged significantly)
- **Label → Label**: Only 23/511 points match frame category labels exactly
- These are far too sparse for a translation layer to work

### Required Fix

The Qdrant collection must be **re-enriched against the ICE.* taxonomy**
that the pipeline actually uses.  The enrichment source loader must read
from the same vocabulary that `load_sample_vocabulary()` returns.
A code-translation shim in the bridge would be fragile (partial coverage,
maintenance burden) and violates the architecture's "Qdrant is source of
truth" principle.

### Why This Wasn't Caught

1. Commit 929e29e's "verification" measured K=0.0005 and
   subtree_concentration=0.0 — both are exactly what a vacuous mass
   produces (no mass to conflict, no leaf to aggregate).  The test
   checked the bridge returned *something*, not that it returned
   *evidence*.
2. No assertion in the pipeline that the cosine source contributes
   non-vacuous mass.  A vacuous mass merged via Dempster's rule is
   an identity operation — it's as if the source doesn't exist.
3. The bridge's status codes distinguish "Qdrant down" from "scoring
   error" but don't detect "scored successfully but nothing in frame."

---

## Findings

### FINDING 1 (Medium): `parent_path` role is misnamed — actually format-corroboration

**Location**: `src/atelier/classify/late_interaction.py:346-352`

The role named `parent_path` computes `pattern_query × view.value_patterns`
(MaxSim of the column's pattern-summary embedding against the annotation's
format-pattern vectors).  The *name* suggests hierarchical context matching,
but the computation is format cross-verification.

The weight key `weight_parent_path` reads as if it's about hierarchical
path matching, when it's actually about pattern-to-format corroboration.
Causes confusion during tuning.

**Recommendation**: Rename the role to `format_corroboration` or
`pattern_x_format`, and the weight to match.

### FINDING 2 (Low): Dead code in `_dedup_top_n`

**Location**: `src/atelier/classify/multi_vector_features.py:195-197`

A comprehension with a walrus operator that is immediately overwritten on
line 199.  The `noqa: F841` suppression suggests awareness that something's
off.  Working implementation is correct (lines 199-207).  Delete the dead
block.

### FINDING 3 (Medium): `is_enabled()` docstring says "Default off"

**Location**: `src/atelier/classify/late_interaction_bridge.py:62`

Stale from when the feature was first introduced.  `config/base.conf`
sets `enabled = true` and `AtelierConfig` defaults to `True`.

### FINDING 4 (Medium — remediated): `description_view` stored but never scored

Each `AnnotationView` carries `description_view` and every Qdrant point
stores it, but `score_column_against_index` never references it.  The
description embedding has discriminative signal being left on the table.

**Remediation**: Added a `description` role that scores
`name_query × description_view` (column name embedding against annotation
description embedding).  Weight `weight_description = 0.10`.  Positive
weights rebalanced to sum to 1.0.

---

## Blind Spots (Calibration-Level, Not Logic Bugs)

### Anti-examples cannot overpower confident positive signal

Positive ceiling α=0.80, negative budget β=0.30 → max K ≈ 0.24.  If false
positives survive anti-example evidence, raise `negative_beta` toward 0.45.

### Verifier attenuation scales both channels equally

A poorly-verified annotation contributes less in BOTH directions.  Audit
the live collection's `verifier_pass_rate` distribution; if median < 0.8,
enrichment quality is bottlenecking.

### MaxSim normalization dilutes mixed-type columns

Columns with diverse samples get diluted cosine scores even when some
samples are excellent matches.  Mixed-type columns (the hardest targets)
will systematically underperform on cosine evidence.

---

## Verified Correct

- `dempster_combine` / `yager_combine` — mass normalization, K accumulation
- `combine_multiple` — sequential combination with Smarandache cumulative K
- `top1_margin` disjoint-FE traversal (commit 1fee0be fix)
- `HierarchicalClassification.from_combined_evidence` — headline selection,
  internal-node/leaf interaction
- `late_interaction_to_mass` channel decomposition + Yager fallback on K=1
- Bridge self-supply embedder (commit 929e29e)
- Enrichment loop: content-addressed caching, verifier retry, MaxSim
  collection schema
- Bootstrap convergence: gap/bel/margin gates, independent-tier revisit,
  plateau detection, budget exhaustion

---

## End-to-End DST Verification (Attempted)

Ran 6 known failure cases from
`Atelier-Results-vs-Prompt-solution-522d89ae.xlsx` through the live
scoring pipeline.  ALL returned vacuous mass (Θ=1.0) due to Finding 0.
Cannot validate whether late-interaction fixes those errors until the
enrichment is rebuilt against the ICE.* taxonomy.

Test cases attempted:
- `social_profiles.page_ref` (SYSURL→PRSNURL)
- `streaming_activity.query_ref` (TRANSID→SRCHQ)
- `streaming_activity.device_ref` (DEVATTR→DEVNAME)
- `system_monitoring.network_addr` (IPADDR→DEVMACADDR)
- `ecommerce_orders.event_date` (DATE→TRANSDATE)
- `ecommerce_orders.destination_ref` (SHIPADDR→ADDRFULL)

Embedding-level simulation (bypassing Qdrant, using direct model.encode)
shows the enriched descriptions WOULD discriminate on several of these:
- `page_ref`: PRSNURL desc beats SYSURL desc by Δ=0.24 on name_query
- `query_ref`: SRCHQ desc beats TRANSID desc by Δ=0.11 on name_query
- `network_addr`: DEVMACADDR desc beats IPADDR desc by Δ=0.08 on name_query

But `device_ref` goes the WRONG way on `name_query × description_view`
(DEVATTR wins by Δ=0.17) while the reference says DEVNAME — confirming
that `name_query × description_view` is not a universal improvement and
the `label_view` concatenation already captures the useful signal on many
cases.  The `sample_queries × description_view` pairing is more promising
(disambiguates via value content) but also not universal (Case 2: neither
TRANSID nor SRCHQ descriptions match search-query values).
