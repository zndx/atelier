# Extract Features

Extract 12 discrete, ablatable features from column samples for multi-method classification. Each feature is independently maskable to support SAGE feature importance analysis.

## Features

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `column_name` | text | Snake/camel-split column identifier |
| 2 | `column_type` | text | SQL/pandas dtype (e.g., VARCHAR, int64) |
| 3 | `sample_values` | text | Representative non-null values (up to 5) |
| 4 | `cardinality` | float | Unique value count / total row count |
| 5 | `null_ratio` | float | Fraction of NULL/NaN values |
| 6 | `value_entropy` | float | Shannon entropy of value distribution |
| 7 | `pattern_signals` | text | Matched regex patterns (email, phone, SSN, etc.) |
| 8 | `avg_value_length` | float | Mean string length of non-null values |
| 9 | `numeric_ratio` | float | Fraction of values parseable as numbers |
| 10 | `sibling_context` | text | Names of adjacent columns in the same table |
| 11 | `source_table` | text | Originating table name |
| 12 | `value_description` | text | LLM-generated natural language summary of value semantics |

## Procedure

1. Load the dataset (parquet or table reference) and sample columns
2. For each column, compute all 12 features from the `ColumnSample`
3. Build an embedding text representation via `to_embedding_text(mask)` where `mask` controls feature ablation
4. Return a `ColumnFeatures` dataclass per column with all 12 fields populated

## Input
- Dataset path or table reference
- Sample size (default: 100 rows)
- Feature mask (optional — dict of feature_name → bool for ablation experiments)

## Output
JSON array of feature records, one per column:
```json
{"column_name": "...", "column_type": "...", "cardinality": 0.85, "null_ratio": 0.02, ...}
```

## Notes
- Feature extraction is deterministic — same sample produces same features
- Pattern signals are computed by the 8 regex detectors (see `/detect-patterns`)
- `value_description` requires an LLM call; omit when running in batch mode
