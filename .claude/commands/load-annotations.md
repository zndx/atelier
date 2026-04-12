# Load Annotations

Load the controlled vocabulary from the `default.annotations` hive table.

## Instructions

1. Load annotations from hive (or cached JSON):
   ```python
   from atelier.config import load_config
   from atelier.classify.taxonomy import (
       load_annotations_from_json,
       load_mock_annotations,
       save_annotations_json,
   )
   from atelier.classify.sampler import load_annotations_from_hive

   cfg = load_config()

   # Try hive first, fall back to cache/mock
   try:
       records = load_annotations_from_hive(cfg)
       from atelier.classify.taxonomy import _build_category_set_from_records
       cs = _build_category_set_from_records(records, hierarchical=True)
       save_annotations_json(cs, "build/data/annotations/annotations.json")
   except Exception:
       cs = load_mock_annotations(hierarchical=True)
   ```

2. Validate the vocabulary:
   - All leaf categories have non-empty labels and descriptions
   - Dot-notation hierarchy is consistent (parents exist)
   - No duplicate codes
   - Deprecated entries excluded

3. Build the FrameOfDiscernment:
   ```python
   from atelier.classify.belief import FrameOfDiscernment
   frame = FrameOfDiscernment(cs)
   ```

## Annotations Schema

| Column | Maps to | Purpose |
|--------|---------|---------|
| id | code | Hierarchical dot-notation identifier |
| ontology | label group | Sensitivity tier / parent grouping |
| annotation | label | The annotation tag name |
| definition | description | Human-readable definition |
| common_names | abbrev | Pipe-separated aliases |
| deprecated | filter | "yes" = exclude from active vocabulary |
| non_corp, emp_contractor, individual, corp | sensitivity | Data subject role ratings |

## Output

A `HierarchicalCategorySet` with leaf categories, parent nodes, and a `FrameOfDiscernment` ready for DST mass function computation.
