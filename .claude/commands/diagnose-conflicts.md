# Diagnose Conflicts

Analyze items with high conflict factor K from Dempster-Shafer fusion to identify contradictory evidence sources and confusable category pairs.

## Conflict Indicators

| Indicator | Threshold | Meaning |
|-----------|-----------|---------|
| K > 0.3 | Advisory | Moderate disagreement between evidence sources |
| K > 0.5 | Warning | Strong contradiction — label assignment is unreliable |
| K > 0.8 | Critical | Evidence sources fundamentally disagree — manual review needed |
| Pl(A) - Bel(A) > 0.4 | Wide interval | High epistemic uncertainty for top candidate |

## Confusable Pairs

The ontology defines known confusable pairs — categories that are semantically adjacent and frequently confused by classifiers. Examples from SIGDG:

- `PersonName` / `UserName` — both contain human-readable identifiers
- `EmailAddress` / `ContactInfo` — email is a subtype of contact
- `Location` / `Address` — geographic vs. postal context

When top-2 fused beliefs both belong to a confusable pair, the conflict is expected and the pair focal element carries meaningful mass.

## Procedure

1. Load fused belief assignments from `/apply-dempster-rule`
2. Filter items where K exceeds the conflict threshold
3. For each high-conflict item:
   a. Decompose K by evidence source pair (which sources disagree most)
   b. Check if top-2 candidates form a known confusable pair
   c. Compute per-source belief intervals to identify the outlier
4. Generate a diagnostic report with recommended actions

## Input
- Fused belief assignments with conflict factors
- Conflict threshold (default: 0.3)
- Ontology confusable pair definitions

## Output
JSON diagnostic per flagged column:
```json
{"column": "...", "conflict_K": 0.62, "top_pair": ["PersonName", "UserName"], "is_confusable": true, "source_disagreement": {"cosine": "PersonName", "catboost": "UserName", "svm": "PersonName"}, "recommendation": "confusable_pair_expected"}
```

## Notes
- Not all conflict is bad — confusable pairs are a natural property of the ontology
- High K from a single outlier source may indicate a calibration issue in that classifier
- Conflict diagnostics feed back into evidence source tuning and GEPA prompt evolution
