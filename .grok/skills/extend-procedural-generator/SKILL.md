---
name: extend-procedural-generator
description: >-
  Add an exact-IRI value generator for ontology terms the procedural
  generator does not shape. Use when TASK.md asks for
  out/extend_generators.py, including /extend-procedural-generator.
allowed-tools: read_file, write, list_dir
---

# Extend the procedural generator

`ontology.csv` lists SDG terms that still need a stored-value generator. Each term's IRI is in the `iri` column. Bind a generator to that IRI exactly.

## Read

1. Read the extend-procedural-generator skill, then `ontology.csv`, `TASK.md`, and `installed.py`.
2. `installed.py` holds IRIs that are already accepted. Do not repeat them.

## Write

3. Write only `out/extend_generators.py`. Raw Python, not Markdown. The file is exactly:

```python
import random

GENERATORS = {
    "https://signals.zndx.org/sdg#ROLE_PARENTING_ROLE": lambda rng: f"PR-{rng.randint(100, 999)}",
}
```

4. Each entry is one IRI from the `iri` column and a `lambda rng: ...`. No helpers, classes, regexes, or other statements.
5. Copy the IRI exactly. One entry per requested term.
6. The lambda calls `rng.randint` or `rng.choice` and returns a short value a database would store: an identifier, a date, an amount, or a status token. It does not return the label or the description. A BFO or CCO genus is a real term; write a value a column of that term would store.
7. Leave `ontology.csv`, `TASK.md`, `installed.py`, and `request.json` unchanged.
