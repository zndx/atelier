# Ægir ↔ Atelier: first direct Atlas integration (Phase 0 + DDL-as-input)

Date: 2026-06-30
Atlas instance: Ægir's forked Apache Atlas v3.0.0-SNAPSHOT (AGE/Postgres backend)
on `http://localhost:21000`, auth `admin/admin`.

## What prompted this

We spent most of the CAI work inside the deployed pod and never evidenced a
real Atlas integration — only an anemic health check. Meanwhile Ægir matured
a deterministic ontology → SQL DDL spine and started projecting it into a live
Atlas. This is the first concrete step at bringing the two projects together:
**Ægir's DDL-generation artifacts become inputs to classification here, and
Ægir's Atlas becomes a real resource (discovery + vocabulary + scoreboard).**

## Live Atlas state (verified, read-only)

- `admin/status` → ACTIVE. 243 typedefs: 123 ENTITY, 15 CLASSIFICATION,
  1 BUSINESS_METADATA (`OntologyProvenance`).
- Ægir's `rdbms_*` projection is live: **890 `rdbms_table`, 5848
  `rdbms_column`, 418 `rdbms_foreign_key`**, plus the **`Aegir Ontology`
  glossary (540 terms)** across 7 families and 29 categories.
- Classification typedefs (`cta_*`, `cpa_*`, `domain_*`) are **registered but
  NOT applied** to any entity (sampled columns carry empty
  `classificationNames`; `OntologyProvenance` business metadata is null on the
  columns checked). **Implication:** the "shared scoreboard" of Phase 3 is not
  pre-populated by Ægir — there is no tag collision today, and Atelier can
  write its predictions into a distinct namespace cleanly.
- View-column qualified names are malformed relative to the
  `db.table.col@cluster` convention (`corpus.view_x@aegir.col` — the `@aegir`
  precedes the column). Footprint columns are well-formed
  (`footprint.t_x.col@aegir`). We read **footprint**, not corpus views.

## Connectivity (Phase 0) — proven with Atelier's own client

`GovernanceClient.from_atelier_config(cfg)` connects to the live instance with
**zero code changes** — only three env vars
(`ATELIER_ATLAS_URL/USER/PASSWORD`). `cfg.has_atlas` flips True, the
`CDPUrlResolver` maps `:21000` → `/api/atlas/v2`, `ping()` returns
`{ok, classification_count: 15, entity_type_count: 123}`, a known footprint
column round-trips to a GUID, and the 540-term glossary reads back.

## What was built

1. **`src/atelier/classify/aegir_release.py`** — loads Ægir's Atelier-facing
   release (`/raid/checkpoints/aegir-artifacts/atelier_release_v0_3/`) into
   `list[TableSample]`. The release is a **blind** dataset: obfuscated column
   names (`x`, `y`), signal in `sample_values`; the answer key is a physically
   separate `reference.parquet`. Verified: **9229 tables / 16516 columns**
   load blind with **no `reference_code` leakage**; the held-back key joins by
   id and by name (146 distinct codes, == `template_coverage`);
   `with_reference=True` populates the validation channel.
   - `load_aegir_release_samples(...)`, `load_aegir_reference(...)`,
     `load_aegir_reference_by_name(...)` (scores blind predictions, which come
     back keyed by name), `load_release_stats(...)`.

2. **`src/atelier/governance/atlas_source.py`** — reads the live Atlas as two
   Atelier inputs:
   - `read_tables(...)` / `read_table_schema(...)` → value-less `TableSample`
     schema skeletons from `rdbms_table.columns` (real names like `subject`,
     `effective_date`, `priority`; surrogate `id` dropped). Atlas holds no
     sample values, so these feed the name/pattern channels and are meant to
     be *joined* with release values for the value-driven channels.
   - `read_glossary(...)` → `GlossaryTerm` vocabulary. Term `name` ==
     Ægir `template_id` (the join to reference codes); surface forms parsed
     from the `longDescription` text convention. Verified: **540/540 terms**
     parse a non-empty surface-form list; categories match the 7 families.

3. **`src/atelier/governance/atlas.py`** — added
   `AtlasClient.get_glossary_terms(guid, limit, offset)` (the rich
   `/glossary/{guid}/terms` endpoint; `basic_search` returns only name+qn).

4. **Tests** (hermetic, no live Atlas / `/raid`): `tests/classify/
   test_aegir_release.py` (blind invariant, reference joins, caps) and
   `tests/governance/test_atlas_source.py` (surface-form/category parsing,
   surrogate drop, corpus-view filter, glossary build). 9 passed.

## Data-path summary (the three keyed surfaces)

| Surface | Keyed by | Carries | Use |
|---|---|---|---|
| `corpus_columns.parquet` | `table_id`,`column_id` | obfuscated name, **values** | blind input (value channels) |
| `reference.parquet` | `column_id` → `template_id`,`source_table` | true `SDG.ICE.*` code | held-back scoring key |
| Atlas `rdbms_column` | `qualifiedName` | **real** name, FK graph | schema/discovery + writeback target |
| Atlas glossary term | `name` == `template_id` | surface forms, axiom, category | classification vocabulary |

`reference.template_id`/`source_table` is the bridge: it ties a release
column to both its glossary term (vocabulary) and its Atlas
`footprint.<source_table>@aegir` entity (writeback target).

## Next milestones (not done this turn)

- **Blind run + score**: feed `load_aegir_release_samples(...)` through
  `run_classification_pipeline(samples=...)` and score against
  `load_aegir_reference_by_name(...)`. Needs the embedding/LLM stack
  (devenv + API key); natural as a `@slow @tier-1` BDD efficacy scenario.
  NB: v0.3 covers 146/540 templates — parity numbers speak to ~¼ of the
  ontology, per analytical-conclusion-strength discipline.
- **Vocabulary adoption**: map the 540 glossary terms (term name +
  surface forms) into Atelier's term/code structure so classification targets
  the SDG vocabulary Ægir owns, rather than a local taxonomy.
- **Writeback / shared scoreboard**: parameterize `governance/sync.py` off the
  hardcoded `hive_table`/`hive_column` onto `rdbms_table`/`rdbms_column`
  (QN format already matches: `QualifiedName(database="footprint",
  cluster="aegir")`), and apply Atelier predictions as tags in a distinct
  classification namespace (`atelier_*`) so Ægir's reference and Atelier's
  prediction sit side by side on the same entity.
- **Config wiring**: surface a release-dir setting and an Ægir-cluster default
  (`cluster_name="aegir"`) via HOCON rather than call-site args.
