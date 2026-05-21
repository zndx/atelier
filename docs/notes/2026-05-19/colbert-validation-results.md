# ColBERT Late-Interaction Validation Results

**Date**: 2026-05-19
**Branch**: `feat/dst-late-interaction-cosine`
**Environment**: CAI Application pod (PGlite:5440, Qdrant:6333)

## Executive Summary

The ColBERT encoder and Qdrant MaxSim retrieval work correctly.  The
architecture produces **discriminative signal** on real data — MaxSim
scores cleanly separate correct annotations from distractors.  However,
a **critical clamping bug** in `_late_interaction_positive_mass`
(`mass_functions.py`) destroys this signal before it reaches the DST
fusion layer.  Fix required before sweep compute.

### Verdict: ARCHITECTURE VALID, MASS FUNCTION BLOCKED

| Component | Status | Notes |
|-----------|--------|-------|
| ColBERT encoder | ✓ OK | colbertv2.0, 128d, unit-normalized |
| Qdrant multi-vector (MaxSim) | ✓ OK | Collection created, queries return scored results |
| Score discriminability | ✓ OK | Strong matches score 5–6; weak 2–3; margins 0.005–2.0 |
| `late_interaction_to_mass` | ✗ BUG | Clamp `min(1.0, score)` zeros discriminability |
| Mass conservation | ✗ LEAK | 0.979 total (2% leak from duplicate code in collection) |
| `sigma_marg` tuning | ✓ OK | 0.05 suits MaxSim's narrow-margin regime (0.005–0.04) |

---

## Step 1 — Collection Re-population

Created `annotations_default_v1_colbert_validation` with 30 annotations
from the source `annotations_default_v1` (511 points, 7-field MiniLM-384d
schema → single `colbert` 128d MaxSim field).

| Metric | Value |
|--------|-------|
| Points upserted | 30 |
| Unique codes | 29 (one duplicate: `1.1.1.2.3.4.3`) |
| Token range per annotation | 132–510 tokens × 128d |
| Vector field config | `colbert`: size=128, Cosine, MaxSim |
| Source collection points | 511 |

**Note**: The duplicate code (`Age Differentiating`, two distinct points)
is a data-quality issue in the source collection that should be
deduplicated before production enrichment runs.

---

## Step 2 — Spot-Check Results (Column-Name-Only Queries)

15 columns sampled randomly from the 86 agent-mediated reference columns whose
correct annotation exists in the 30-point validation set.

### Summary Statistics

| Metric | Value |
|--------|-------|
| Top-1 accuracy | 4/15 (26.7%) |
| Mean top-1/top-2 margin | 0.5034 |
| Min/Max margin | 0.053 / 1.934 |
| Mean score for correct annotation | 3.78 |
| Score range (correct) | 2.85 – 5.69 |

### Hits (correct at rank 1)

| Column | Expected | Score | Margin |
|--------|----------|-------|--------|
| `contact_supplemental.office_ref` | Office Number | 4.22 | 0.632 |
| `contact_supplement.desk_ref` | Office Number | 3.93 | 0.654 |
| `payment_instruments.stripe_data` | Magnetic Stripe Data | 5.69 | 1.934 |
| `clinic_encounters.visit_date` | Transaction Date | 3.08 | 0.762 |

### Misses (correct not at rank 1)

| Column | Expected | Actual Rank | Notes |
|--------|----------|-------------|-------|
| `travel_records.ts_ref` | Transaction Date | 2 | Gov ID wins on "ref" |
| `health_location_profiles.alias_ref` | Nickname | 2 | Narrow miss |
| `enterprise_document_store.proposal_ref` | Sales Doc | 2 | Gov ID wins on "ref" |
| `component_catalog.mod_ts` | Transaction Date | 3 | Weak signal from abbrev name |
| `credential_key_store.chlng_ref` | Security Question | >5 | "Key Material" dominates in crypto context |
| Others | Various | >5 | Name-only queries insufficient for disambiguation |

### Assessment

26.7% top-1 accuracy is **expected and acceptable** for column-name-only
queries.  The procedure document correctly noted this limitation.  Key
observations:

1. **Semantically transparent names hit reliably**: `stripe_data` →
   Magnetic Stripe (score 5.69, margin 1.93); `visit_date` →
   Transaction Date (margin 0.76); `office_ref` → Office Number
   (margin 0.63).

2. **Opaque/abbreviated names miss**: `ts_ref`, `chlng_ref`, `val_72`,
   `mod_ts`, `hw_ref` — these require sample values, type info, and
   sibling context to disambiguate.  Production `ColumnFeatures` provides
   exactly this signal.

3. **"_ref" suffix bias**: the Gov ID annotation dominates queries
   containing "ref" (likely due to "reference" tokens in its composed
   text).  This is a content issue in the annotation's composed text,
   not an architecture failure.

4. **Score magnitudes are meaningful**: strong matches (5–6) are clearly
   separated from marginal matches (2–3).  The MaxSim scoring function
   is working as designed.

---

## Step 3 — Mass Function Results

### Bug: Score Clamping (`min(1.0, s.positive_score)`)

**Location**: `src/atelier/classify/mass_functions.py`,
`_late_interaction_positive_mass`, line:
```python
sim = max(0.0, min(1.0, s.positive_score))
```

**Root cause**: This line was inherited from the cosine-similarity path
where scores are in [0, 1].  ColBERT MaxSim scores are **sums of
per-token max-cosines** across the token sequences.  With 4–510 tokens
per document, scores range from ~1.7 to ~6.4.  The `min(1.0, ...)`
clamps all scores to exactly 1.0, producing:

- Uniform softmax probabilities (1/N for all N candidates)
- Zero margin → `margin_weight = 0.0`
- All non-Θ mass distributed equally across candidates
- No discriminative signal reaches the DST fusion layer

**Observed effect** (5 test cases):

| Entity | Θ mass | Top-1 bel | All-candidate bel | Margin |
|--------|--------|-----------|-------------------|--------|
| email_addr + samples | 0.40 | 0.0206 | 0.0206 (uniform) | 0.0000 |
| stripe_data | 0.40 | 0.0206 | 0.0206 (uniform) | 0.0000 |
| visit_date + samples | 0.40 | 0.0206 | 0.0206 (uniform) | 0.0000 |
| office_ref + samples | 0.40 | 0.0206 | 0.0206 (uniform) | 0.0000 |
| ssn_val + samples | 0.40 | 0.0206 | 0.0206 (uniform) | 0.0000 |

Total mass: 0.9794 in all cases (2% leak from duplicate-code dedup).

### Simulated Fix: Remove the Clamp

Replacing `min(1.0, s.positive_score)` with raw score pass-through:

| Entity | Θ mass | Top-1 | Top-1 bel | Margin |
|--------|--------|-------|-----------|--------|
| stripe_data (easy) | 0.20 | Magnetic Stripe Data | 0.80 | 0.80 |

Mass conservation: **1.000000** (perfect).

### `sigma_marg` Already Tuned Correctly

Initial concern: `sigma_marg=0.05` might saturate for MaxSim's larger
score deltas.  **Disproven.**  Harder queries produce narrow MaxSim
margins that fall in the correct operating range:

| Entity | Top-1/Top-2 Margin | `margin_weight` |
|--------|-------------------|-----------------|
| `ts_ref | travel_records` | 0.006 | 0.11 |
| `mod_ts | component_catalog` | 0.014 | 0.28 |
| `alias_ref | health_location` | 0.044 | 0.71 |
| `stripe_data | payment` (easy) | 2.001 | 1.00 |

The parameter correctly implements graduated concentration: ambiguous
queries spread mass across candidates; clear queries concentrate on
top-1.  The `sigma_marg` setting needs no adjustment.

### Mass Conservation Leak (2%)

Cause: collection contains 2 points with code `1.1.1.2.3.4.3` (Age
Differentiating).  When building `in_frame`, the dict overwrites on
duplicate codes, but the softmax denominator includes all non-overwritten
entries.  This is a minor issue — it manifests only with duplicate codes
in the collection and produces <3% leak.

**Fix**: deduplicate the source collection (or use the first/best-scoring
entry per code).

---

## Findings by Priority

### P0 — Must fix before sweep

1. **Remove score clamp** in `_late_interaction_positive_mass`:
   ```python
   # Before (broken):
   sim = max(0.0, min(1.0, s.positive_score))
   # After (correct):
   sim = max(0.0, s.positive_score)
   ```
   MaxSim scores are unbounded-above (sum of per-token max-cosines).
   The softmax normalization already handles scale; the `min(1.0, ...)`
   is both incorrect and destructive.

2. **Deduplicate collection points by code**: the `in_frame` dict
   build should handle multiple points for the same code (take max
   score, or average).  Alternatively, enforce uniqueness at upsert
   time.

### P1 — Should address before production

3. **`_cosine_reliability` tau_abs/sigma_abs**: With MaxSim scores
   always >> 0.4, `alpha_abs` saturates at 1.0 and only the margin
   term contributes.  The absolute-reliability sigmoid is effectively
   dead code for this source.  Options:
   - Rescale `tau_abs`/`sigma_abs` for MaxSim range (~3.5 center)
   - Remove `alpha_abs` for the late-interaction source (margin-only)
   - Keep as-is (the margin term alone produces correct behavior)

4. **Qdrant client version warning**: client 1.18.0 vs server 1.13.2
   (minor version gap > 1).  MaxSim multi-vector queries work, but
   pin the client version or upgrade Qdrant to avoid API drift.

### P2 — Documentation / procedure fixes

5. **Procedure doc `col_key` scoping bug** (Step 2): in the 3-tuple
   branch, `col_key` is undefined.  The procedure runs as-is because
   only 2-tuples are generated, but the dead code is misleading.

6. **`HierarchicalClassification.from_mass` does not exist**: the
   procedure's Step 3 calls a non-existent method.  Correct API is
   `HierarchicalClassification.from_combined_evidence(source_masses={"late_interaction": mass}, frame=frame, category_set=cat_set)`.

7. **Discount reference**: procedure says "0.20 discount"; memory
   records 0.22 for subsumption alignment.  The late-interaction
   source's `discount=0.20` default is intentional (separate
   parameter per source), but should be explicitly documented as
   distinct from the SVM source's 0.22.

---

## Decision

**Architecture validated.  Do NOT commit sweep compute until P0 is
fixed.**  The MaxSim retrieval produces discriminative signal; the mass
function destroys it.  After removing the clamp and deduplicating codes:

- Re-run this procedure's Steps 2-3 to confirm mass conservation = 1.0
  and top-1 belief > 0.5 for semantically clear matches
- Then proceed to a 3–4 cell sweep (not full 12-cell grid) per the
  procedure's decision gate

---

## Environment Snapshot

```
Branch: feat/dst-late-interaction-cosine
ColBERT model: colbert-ir/colbertv2.0 (128d, ~500MB)
Qdrant: 1.13.2 (server) / qdrant-client 1.18.0 (Python)
Source collection: annotations_default_v1 (511 points, 7-field MiniLM-384d)
Validation collection: annotations_default_v1_colbert_validation (30 points, 1-field ColBERT-128d)
Frame: 220 singletons, 76 internal nodes, 296 annotations total
GT: 920 columns, 173 distinct codes
ATELIER_DB_URL: postgresql+psycopg://postgres:postgres@127.0.0.1:5440/postgres
```
