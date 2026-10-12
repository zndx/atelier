---
name: synthetic-relational-corpus
description: >-
  Write a synthetic relational corpus as CSV tables from ontology.csv and
  TASK.md in the working directory. Use when asked to generate tables for
  ontology codes, including /synthetic-relational-corpus.
allowed-tools: read_file, write, list_dir
---

# Synthetic relational corpus

The working directory is the whole task. `ontology.csv` is the only catalogue. `TASK.md` names the requested codes and how many data rows each table needs.

## Read

1. Read `ontology.csv`. Columns are `code`, `label`, `abbrev`, `parent_code`, `description`.
2. Read `TASK.md`. Generate one table for each requested code.

## Write

3. For each requested code, invent one operational table that would store that concept, plus one sibling column that belongs on the same record.
4. The table name matches `[a-z][a-z0-9_]{0,63}`. Write `out/tables/<table>.csv` in UTF-8. The first row is the header. Then write the requested number of data rows. Column names are `snake_case`.
5. One column carries the requested code. A sibling column is listed in the manifest only when its code is also a `code` in `ontology.csv`.
6. Write `out/manifest.json` as raw JSON, not Markdown. It is a list of objects with the keys `table`, `column`, and `code`. Copy `code` from `ontology.csv`. Every requested code appears once.

```json
[{"table": "account", "column": "account_id", "code": "EXAMPLE.ACCOUNT"}]
```

```csv
account_id,opened_on
A-100,2020-01-04
A-204,2021-11-19
```

That pair shows the shape. `EXAMPLE.ACCOUNT` is not a code in this run. Use codes from `ontology.csv`, and do not copy these sample cells.

## Cell values

Write values a database would store: identifiers, `YYYY-MM-DD` dates, amounts, short names, or status tokens. A cell is not the ontology description and not the column label repeated on every row.

## Files

Create files with the write tool. Leave `ontology.csv`, `TASK.md`, and `request.json` unchanged.
