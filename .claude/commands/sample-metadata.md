# Sample Metadata

Sample column metadata from production tables via CAI Data Platform hive connections.

## Instructions

1. Load the AtelierConfig to get data connection settings:
   ```python
   from atelier.config import load_config
   cfg = load_config()
   ```

2. Use the sampler module to collect column metadata:
   ```python
   from atelier.classify.sampler import sample_table_metadata, discover_tables

   # Discover available tables
   tables = discover_tables(cfg, database="default")

   # Sample each table's columns
   for table_name in tables[:10]:  # limit for first pass
       sample = sample_table_metadata(cfg, table_name, sample_size=50)
       for col in sample.columns:
           print(f"  {col.name} ({col.column_type}): {col.values[:3]}")
   ```

3. For each column, extract the following metadata:
   - Column name and SQL type
   - Sample values (first 5 non-null)
   - Total row count and null count
   - Sibling column names (other columns in the same table)

4. Save sampled metadata to `build/data/samples/{run_id}/columns.json`

## Mock Data

When `cml.data_v1` is unavailable (devenv/CI), the sampler automatically falls back to mock fixtures in `src/atelier/classify/fixtures/mock_tables.json`. These contain 8 realistic tables with 50 columns and known ground truth labels.

## Output

A list of `ColumnSample` objects ready for feature extraction and classification.
