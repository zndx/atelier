---
name: extend-procedural-generator
description: >-
  Add a relational generator for a target column's published SKOS IRI.
  Use when TASK.md asks for out/extend_generators.py, including
  /extend-procedural-generator.
allowed-tools: read_file, write, list_dir
---

# Extend the procedural generator

`ontology.csv` lists published SDG terms whose columns need a synthetic table. Each term's IRI is in the `iri` column. Bind a generator to that IRI exactly. The generator returns a new table that would lead a reader to assign that IRI to exactly one of its columns.

## Read

1. Read the extend-procedural-generator skill, then `ontology.csv`, `TASK.md`, and `installed.py`.
2. `installed.py` holds IRIs that are already accepted. Do not repeat them.

## Write

3. Write only `out/extend_generators.py`. Raw Python, not Markdown. The file is exactly:

```python
import random

GENERATORS = {
    "https://signals.zndx.org/sdg#ROLE_PARENTING_ROLE": lambda rng: {
        "table": f"care_placement_{rng.randint(10, 99)}",
        "columns": [
            {
                "name": "caregiver_role",
                "values": [rng.choice(["primary", "kinship", "foster"])],
                "iri": "https://signals.zndx.org/sdg#ROLE_PARENTING_ROLE",
            },
            {
                "name": "started_on",
                "values": [f"201{rng.randint(0, 9)}-04-01"],
            },
        ],
    },
}
```

4. Each entry is one IRI from the `iri` column and a `lambda rng: ...`. No helpers, classes, regexes, or other statements.
5. Copy the IRI exactly. One entry per requested term. The lambda calls `rng`.
6. The return value is one table. `columns` has the labeled column and at least one sibling. Exactly one column sets `iri` to that same IRI. Names are snake_case. Each `values` list holds short strings a database would store. Those strings are not the label and not the description.
7. Leave `ontology.csv`, `TASK.md`, `installed.py`, and `request.json` unchanged.
