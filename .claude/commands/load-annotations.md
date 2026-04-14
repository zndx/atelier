# Load Annotations

Load the controlled vocabulary — universal BFO-grounded base plus optional domain extensions from the `default.annotations` hive table.

## Instructions

1. Load annotations (two-layer composition):
   ```python
   from atelier.config import load_config
   from atelier.classify.taxonomy import (
       load_annotations_from_hive,
       load_annotations_from_json,
       load_universal_vocabulary,
       compose_vocabularies,
       save_annotations_json,
   )

   cfg = load_config()

   # Universal base (always available, BFO-grounded)
   universal = load_universal_vocabulary(hierarchical=True)

   # Try domain extensions from hive, compose on top
   try:
       domain_cs = load_annotations_from_hive(cfg)
       cs = compose_vocabularies(universal, domain_cs)
       save_annotations_json(cs, "build/data/annotations/annotations.json")
   except Exception:
       cs = universal
   ```

2. Validate the vocabulary:
   - All leaf categories have non-empty labels and descriptions
   - Mnemonic hierarchy is consistent (parents exist)
   - No duplicate codes
   - Deprecated entries excluded

3. Build the FrameOfDiscernment:
   ```python
   from atelier.classify.belief import FrameOfDiscernment
   frame = FrameOfDiscernment(cs)
   ```

## Two-Layer Vocabulary Architecture

```
Domain Extensions (runtime, from hive)     Universal Base (in git, BFO-grounded)
  ACME.TRADE_SECRET is_a BUSINESS    →     ICE → SENSITIVE → PID → CONTACT → EMAIL
```

- **Universal layer**: BFO-grounded terms shipped in `fixtures/universal_vocabulary.json`
- **Domain layer**: Customer-specific terms from hive annotations table, attached via `parent_code`

## Annotations Schema (Hive)

| Column | Maps to | Purpose |
|--------|---------|---------|
| id | code | Mnemonic dot-path identifier (e.g., ICE.SENSITIVE.PID.CONTACT.EMAIL) |
| ontology | label group | Sensitivity tier / parent grouping |
| annotation | label | The annotation tag name |
| definition | description | Human-readable definition |
| common_names | abbrev | Pipe-separated aliases |
| deprecated | filter | "yes" = exclude from active vocabulary |

## Output

A `HierarchicalCategorySet` with leaf categories, parent nodes, and a `FrameOfDiscernment` ready for DST mass function computation.
