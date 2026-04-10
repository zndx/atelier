# Apply Dempster's Rule

Combine multiple mass functions via Dempster's rule of combination to produce a unified belief assignment with uncertainty-aware intervals.

## Dempster's Rule

Given two mass functions m1 and m2, the combined mass is:

```
m12(C) = [ sum of m1(A) * m2(B) for all A,B where A ∩ B = C ] / (1 - K)
```

where K is the **conflict factor**:

```
K = sum of m1(A) * m2(B) for all A,B where A ∩ B = empty
```

## Belief Measures

From the fused mass function, compute three measures per category:

| Measure | Formula | Interpretation |
|---------|---------|----------------|
| **Belief** Bel(A) | Sum of m(B) for B ⊆ A | Lower bound — evidence that strictly supports A |
| **Plausibility** Pl(A) | Sum of m(B) for B ∩ A ≠ empty | Upper bound — evidence that does not contradict A |
| **Pignistic probability** BetP(A) | Weighted redistribution of non-singleton mass | Decision-making probability for label assignment |

The **belief interval** [Bel(A), Pl(A)] captures both support and uncertainty.

## Procedure

1. Load the 5 mass functions per column from `/build-mass-functions`
2. Apply Dempster's rule pairwise: m1 ⊕ m2 ⊕ m3 ⊕ m4 ⊕ m5
3. Compute belief, plausibility, and pignistic probability for each category
4. Rank categories by pignistic probability
5. Flag items with high conflict factor K (see `/diagnose-conflicts`)

## Input
- Mass functions from `/build-mass-functions` (5 per column)
- Conflict threshold for flagging (default: K > 0.3)

## Output
JSON per column:
```json
{"column": "...", "label": "...", "belief": 0.72, "plausibility": 0.89, "pignistic": 0.81, "conflict_K": 0.15}
```

## Notes
- Combination order does not matter — Dempster's rule is commutative and associative
- High K (> 0.5) indicates strongly contradictory evidence — investigate with `/diagnose-conflicts`
- Pignistic probability is used for the final label assignment decision
