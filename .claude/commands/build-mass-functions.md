# Build Mass Functions

Convert 5 independent evidence sources into Dempster-Shafer basic probability assignments (BPAs). Each converter transforms classifier scores into a `BeliefAssignment` over the frame of discernment.

## Evidence-to-Mass Converters

| Converter | Input | Mass Distribution |
|-----------|-------|-------------------|
| `cosine_to_mass` | Cosine similarities to reference embeddings | Top-k similarities → singleton focal elements; remainder → Theta (ignorance) |
| `catboost_to_mass` | CatBoost class probabilities | Probabilities above threshold → focal elements; low-prob mass → Theta |
| `svm_to_mass` | Calibrated SVM probabilities | Platt-scaled margins → focal elements with confusable pair redistribution |
| `pattern_to_mass` | Pattern match ratios | High-ratio patterns → strong singleton mass; no patterns → all mass to Theta |
| `name_match_to_mass` | Column name ↔ category name similarity | CamelCase-split name matching; exact/fuzzy matches → high singleton mass |

## Confusable Pair Redistribution

When the top-2 singleton masses are close (ratio < 3.0) and both belong to a known confusable pair in the ontology, half of the 2nd-place mass transfers to the pair focal element. This preserves honest ambiguity rather than forcing an arbitrary choice.

## Procedure

1. Collect outputs from `/run-classifiers` and `/detect-patterns`
2. Apply each of the 5 converters to produce a `BeliefAssignment`
3. Validate each BPA: masses non-negative, sum to 1.0
4. Apply confusable pair redistribution where applicable
5. Return all 5 mass functions per column

## Input
- Classifier outputs (cosine, CatBoost, SVM probabilities)
- Pattern detection results
- Column names (for name matching)
- Frame of discernment (category set with confusable pairs)

## Output
JSON per column with 5 mass functions:
```json
{"column": "...", "masses": {"cosine": {...}, "catboost": {...}, "svm": {...}, "pattern": {...}, "name_match": {...}}}
```

## Notes
- Each mass function assigns mass to focal elements (subsets of Theta)
- Theta (the full frame) represents total ignorance — no evidence for any specific category
- Mass functions are the input to Dempster's rule of combination (see `/apply-dempster-rule`)
