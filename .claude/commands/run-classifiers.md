# Run Classifiers

Run three parallel classification methods against column feature vectors: cosine similarity, CatBoost gradient boosting, and SVM with Platt scaling.

## Classifiers

### 1. Cosine Similarity
- Encode column embedding text with sentence-transformers (`all-MiniLM-L6-v2`)
- Compare against pre-computed reference embeddings for each category
- Rank categories by cosine distance; top-k become candidate labels

### 2. CatBoost
- 991-dimensional feature vector: `[full_embedding(384) | values_only_embedding(384) | discrete_features(11) | cosine_similarities(N_ref)]`
- Pre-trained CatBoost model outputs per-class probabilities
- Handles categorical features natively (no one-hot encoding needed)

### 3. SVM (LinearSVC + Platt Scaling)
- TF-IDF vectorization of embedding text
- LinearSVC trained on the same label set
- Platt scaling via `CalibratedClassifierCV` converts margins to probabilities

## Procedure

1. Load pre-extracted `ColumnFeatures` (from `/extract-features`)
2. Encode embedding text with sentence-transformers
3. Run all three classifiers independently
4. Return per-classifier probability distributions over the label set

## Input
- Feature records (output of `/extract-features`)
- Model paths: sentence-transformer, CatBoost checkpoint, SVM checkpoint
- Category set (SIGDG ontology or custom vocabulary)

## Output
JSON with per-column, per-classifier results:
```json
{"column": "...", "cosine": {"label": 0.82, ...}, "catboost": {"label": 0.91, ...}, "svm": {"label": 0.77, ...}}
```

## Notes
- Classifiers are independent — failures in one do not block others
- CatBoost and SVM require pre-trained models; cosine is zero-shot
- All three outputs feed into `/build-mass-functions` as separate evidence sources
