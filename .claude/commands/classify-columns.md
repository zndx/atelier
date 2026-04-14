# Classify Columns

Run the full Dempster-Shafer classification pipeline on sampled columns.

## Instructions

1. Run the end-to-end pipeline:
   ```python
   from atelier.config import load_config
   from atelier.classify import run_pipeline
   from atelier.classify.sampler import load_fixture_samples
   from atelier.classify.mock_llm import RealisticMockLLMBackend

   cfg = load_config()
   samples = load_fixture_samples()
   gt = {c.name: c.ground_truth for ts in samples for c in ts.columns if c.ground_truth}
   result = run_pipeline(
       cfg, samples=samples, llm_backend=RealisticMockLLMBackend(ground_truth=gt),
   )
   ```

2. The pipeline executes these stages:
   - **LOADING_VOCAB**: Load annotations from hive or cache
   - **DISCOVERING**: List tables in the target database
   - **SAMPLING**: Sample column metadata from each table
   - **CLASSIFYING**: For each column:
     - Extract 12 features (name, type, values, patterns, etc.)
     - Run cosine similarity against reference category embeddings
     - Detect value patterns (email, SSN, credit card, etc.)
     - Match column names against category labels/abbreviations
   - **FUSING**: Combine evidence via Dempster's rule:
     - Up to 3 independent mass functions per column (M0)
     - Belief intervals [Bel(A), Pl(A)] expose epistemic uncertainty
     - Conflict K measures source disagreement
   - **EVALUATING**: Compute accuracy against ground truth (when available)

3. Results are written to:
   - `build/results/{run_id}/classifications.json` — Full per-column results
   - `build/results/{run_id}/atelier_embeddings.parquet` — For embedding-atlas

4. Monitor progress via the FSM status API:
   ```bash
   curl http://localhost:8090/api/fsm/status
   ```

## Evidence Sources (M0)

| Source | Evidence Type | Discount | Notes |
|--------|-------------|----------|-------|
| Cosine | Sentence-transformer similarity | 0.30 | all-MiniLM-L6-v2, softmax'd |
| Pattern | Regex pattern detection | 0.10 | 8 detectors (email, SSN, etc.) |
| Name Match | Column name ↔ category label | 0.00-0.70 | Exact/abbrev/word overlap |

## Interpreting Results

- **confidence**: Mass on the predicted singleton
- **belief**: Lower bound of the belief interval (committed evidence)
- **plausibility**: Upper bound (cannot be ruled out)
- **uncertainty**: Pl - Bel (gap = unresolved ambiguity)
- **conflict**: K from Dempster combination (source disagreement)

High conflict + low confidence → revisit with additional evidence sources.
