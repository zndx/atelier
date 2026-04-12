# Discover Tables

List and filter tables in a hive database via CAI Data Platform connections.

## Instructions

1. Connect to the configured hive data source:
   ```python
   from atelier.config import load_config
   from atelier.classify.sampler import discover_tables

   cfg = load_config()
   tables = discover_tables(cfg, database="default", limit=100)
   ```

2. Filter tables based on:
   - Exclude system/metadata tables (starting with `_`, `sys_`, `information_schema`)
   - Exclude tables already classified in the current run
   - Prioritize tables with more columns (higher classification yield)

3. Report discovery results:
   - Total tables found
   - Tables after filtering
   - Estimated column count

## Output

A filtered list of table names ready for metadata sampling.
