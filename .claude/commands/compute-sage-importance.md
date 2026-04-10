# Compute SAGE Importance

Run SAGE (Shapley Additive Global importancE) analysis to measure the global importance of each of the 12 discrete features in the classification pipeline.

## How SAGE Works

SAGE measures feature importance via **marginal contribution**: for each feature, it computes how much classification accuracy drops when that feature is marginalized (replaced with values from other samples). The result is a Shapley-value-based importance score with confidence intervals.

Key components:
- **FeatureMaskModel** — wraps the classifier to accept a feature mask; masked features are replaced with random values from the data distribution
- **Permutation sampling** — SAGE uses permutation-based estimation (not brute-force over all subsets)
- **Loss function** — cross-entropy loss measures classification degradation

## Feature Importance Rankings (typical)

| Rank | Feature | Why Important |
|------|---------|---------------|
| 1 | `sample_values` | Actual cell content is the strongest signal |
| 2 | `column_name` | Human-assigned names carry strong semantic intent |
| 3 | `value_entropy` | Distinguishes categorical from free-text columns |
| 4 | `pattern_signals` | PII patterns are near-deterministic for some categories |
| 5 | `cardinality` | High cardinality suggests identifiers or unique values |

## Procedure

1. Load feature matrix and classification labels
2. Initialize `FeatureMaskModel` wrapping the embedding classifier
3. Run SAGE with permutation sampling (default: 512 samples)
4. Return ranked feature importance with standard errors

## Input
- Feature matrix (output of `/extract-features` for all columns)
- Classification labels (ground truth or fused predictions)
- Number of permutation samples (default: 512)

## Output
JSON with ranked importances:
```json
{"features": [{"name": "sample_values", "importance": 0.142, "std": 0.008}, ...], "method": "permutation", "n_samples": 512}
```

## Notes
- SAGE is a **global** measure — it ranks features across the entire dataset, not per-item
- For per-item explanations, use `/generate-shap-explanations`
- SAGE results inform feature ablation decisions: low-importance features can be dropped to reduce embedding dimensionality
