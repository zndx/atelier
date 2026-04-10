# Prepare Atlas Projection

Generate an Embedding Atlas viewer dataset: UMAP/t-SNE projection of column embeddings enriched with classification results, belief intervals, and SHAP explanations.

## Output Schema

The projection parquet file contains one row per column entity:

| Field | Type | Source |
|-------|------|--------|
| `x` | float | UMAP/t-SNE dimension 1 |
| `y` | float | UMAP/t-SNE dimension 2 |
| `label` | string | Predicted category from DST fusion |
| `column_name` | string | Original column identifier |
| `source_table` | string | Originating table |
| `belief` | float | Bel(label) — lower bound of belief interval |
| `plausibility` | float | Pl(label) — upper bound of belief interval |
| `pignistic` | float | BetP(label) — decision probability |
| `conflict_K` | float | Dempster conflict factor |
| `shap_top1` | string | Most important feature for this classification |
| `shap_top1_value` | float | SHAP contribution of top feature |
| `sensitivity` | string | SIGDG sensitivity level (public/internal/confidential/restricted) |

## Procedure

1. Load column embeddings from the vector store or feature matrix
2. Apply dimensionality reduction:
   - **UMAP** (default): `n_neighbors=15, min_dist=0.1, metric=cosine`
   - **t-SNE** (alternative): `perplexity=30, metric=cosine`
3. Merge with classification results from `/apply-dempster-rule`
4. Merge with SHAP explanations from `/generate-shap-explanations`
5. Compute cluster assignments for color coding (HDBSCAN or label-based)
6. Export as parquet to `data/projections/{dataset_id}.parquet`

## Input
- Column embeddings (384-dim sentence-transformer vectors)
- Fused classification results (from `/apply-dempster-rule`)
- SHAP explanations (from `/generate-shap-explanations`)
- Projection method (default: UMAP)

## Output
Parquet file path ready for Embedding Atlas viewer:
```json
{"path": "data/projections/gittables-sample.parquet", "n_points": 2517, "n_clusters": 42}
```

## Notes
- The Embedding Atlas viewer (`external/embedding-atlas`) reads this parquet directly
- Point colors map to predicted labels; hover shows belief intervals and SHAP features
- High-conflict items (K > 0.5) can be visually distinguished via opacity or border
- Projection is deterministic when UMAP `random_state` is fixed
