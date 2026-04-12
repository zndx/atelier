# Generate Synthetic Tables

Create deterministic synthetic training tables with representative column data.

## Instructions

1. Load the controlled vocabulary:
   ```python
   from atelier.classify.taxonomy import load_mock_annotations
   cs = load_mock_annotations(hierarchical=True)
   ```

2. For each leaf category, generate synthetic tables:
   ```python
   from atelier.classify.synth import generate_synth_tables

   results = generate_synth_tables(
       cs,
       output_dir="build/data/synth/run_001",
       tables_per_category=2,
       columns_per_table=50,
       rows_per_table=100,
   )
   ```

3. Each synthetic table should contain:
   - Columns whose names, types, and values are characteristic of the target category
   - Realistic but deterministic data (seeded random generators)
   - Ground truth labels for every column
   - A mix of easy (strong signal) and hard (ambiguous) columns

4. Validate synthetic data:
   - Every annotation category has at least one representative column
   - Ground truth labels match the vocabulary
   - No category has more than 5x the columns of the smallest category

## M0 Status

Currently returns mock fixture data. M1 will implement deterministic procedural generators driven by Claude Agent SDK keystone agents to create richer synthetic datasets.

## Output

A list of synthetic table metadata dicts written to `build/data/synth/{run_id}/`.
