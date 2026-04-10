# Generate SHAP Explanations

Produce per-item (per-column) explanations of classification decisions using SHAP values. Two complementary methods are available.

## Method 1: CatBoost TreeSHAP

- **Exact** O(TLD) algorithm on the 991-dimensional CatBoost feature space
- Fast: ~2 seconds for 350 items
- Features are raw dimensions — must be grouped back to interpretable names:

| Group | Dimensions | Covers |
|-------|-----------|--------|
| `full_embedding` | 0-383 | sentence-transformer encoding of full column text |
| `values_only_embedding` | 384-767 | encoding of sample values only |
| `discrete_features` | 768-778 | cardinality, null_ratio, entropy, pattern flags (11 dims) |
| `cosine_similarities` | 779-990 | cosine distances to N reference category embeddings |

## Method 2: Embedding PermutationSHAP

- `shap.PermutationExplainer` on the 12-feature embedding classifier
- More interpretable: features map directly to the 12 discrete features
- Slower than TreeSHAP; benefits from GPU encoding
- Uses the same feature mask approach as SAGE

## Procedure

1. Select method based on available models (TreeSHAP if CatBoost trained, else PermutationSHAP)
2. Compute SHAP values for each item in the dataset
3. For TreeSHAP: group 991 dimensions into 4 interpretable feature groups
4. Select top-3 contributing features per item
5. Return per-item explanations with feature contributions

## Input
- Classified dataset with feature vectors
- Method preference (default: `treeshap` if CatBoost available)
- Top-k features to report per item (default: 3)

## Output
JSON per column:
```json
{"column": "...", "label": "EmailAddress", "top_features": [{"feature": "pattern_signals", "shap": 0.34}, {"feature": "sample_values", "shap": 0.28}, {"feature": "column_name", "shap": 0.15}]}
```

## Notes
- SHAP is a **local** (per-item) method complementing SAGE's global importance
- TreeSHAP groups are pre-defined; PermutationSHAP features match `/extract-features` directly
- SHAP explanations feed into the Atlas projection metadata (see `/prepare-atlas-projection`)
