# M0.5: Audit Remediation

## Summary

Post-implementation audit of M0 classification pipeline against the signals
reference project and production annotation schema. Found and fixed critical
schema inversion, missing pignistic probability, and absent
HierarchicalClassification output class.

## Audit Findings

### Critical: Schema Inversion
The `ontology` column in annotations is the **human-readable label** (e.g.,
"Payment Card Number"), not a parent grouping. The `annotation` column is the
**formal code / mnemonic** (e.g., "PAN"), not the label. Our code had these
inverted. Mock annotations masked this because they set both fields identically.

### Major: Missing Pignistic Probability
Pipeline used raw singleton mass for ranking instead of the decision-theoretic
pignistic transform BetP({x}). This matters when confusable pairs or Theta
carry significant mass.

### Major: No HierarchicalClassification
Signals wraps every classification result in a HierarchicalClassification object
with hierarchy navigation methods. Our pipeline returned raw dicts, preventing
post-hoc queries like "what's the belief at the Financial Data parent level?"

## Changes Made

### taxonomy.py
- Fixed `_build_category_set_from_records()`: `ontology` -> `label`, `annotation` -> `abbrev`
- Added `common_names` field to ReferenceCategory
- Added `specifics` (11th column) handling in embedding text
- Fixed `save_annotations_json()` round-trip serialization
- Fixed parent node builder to use `ontology` for label

### belief.py
- Added `HierarchicalClassification` dataclass with:
  - `belief_at()`, `plausibility_at()`, `interval_at()` hierarchy navigation
  - `uncertainty_gap`, `needs_clarification` properties
  - `from_combined_evidence()` factory (vacuous filtering, pignistic ranking)

### pipeline.py
- Replaced `_best_singleton()` with `HierarchicalClassification.from_combined_evidence()`
- Output dicts now include `needs_clarification`, `evidence` fields
- Removed unused `combine_multiple` import and `_best_singleton` function

### mass_functions.py
- Enhanced `name_match_to_mass()` to match against `common_names` aliases
- Supports both pipe-separated and comma-separated alias formats

### Mock fixtures
- `mock_annotations.json`: ontology != annotation (EMAIL, PAN, SSN, etc.)
- Ground truth codes unchanged and validated

### BDD
- Added 3 scenarios: pignistic probability, HierarchicalClassification, schema mapping
- All 44 tier-0 scenarios pass

### Documentation
- Updated `classification.md` with corrected schema table and HierarchicalClassification section
- Updated milestones table with M0.5
- Added LLM evidence source to planned sources

## Test Results

```
44 scenarios passed, 0 failed, 10 skipped
163 steps passed, 0 failed, 31 skipped
TypeScript: clean compilation
```
