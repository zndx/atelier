<!-- Copyright (c) 2026 Cloudera, Inc.  All rights reserved. -->

# Comprehensive CCO Coverage

Atelier's classification vocabulary is grounded in the
[Common Core Ontologies](https://github.com/CommonCoreOntology/CommonCoreOntologies)
(CCO v2.1) via the CCO-mediated BFO alignment in
[`ontology/README.md`](../../../src/atelier/classify/ontology/README.md). To date
the applied pipeline has imported only **3 of CCO's 11 modules**
(Information Entity, Agent, Extended Relation) — a deliberate scoping while
we iterated toward the PII/metadata target. The program goal is now
**comprehensive coverage across all 11 CCO modules**, driven on two fronts:

- **Upstream — Aegir** (ontology owner): import the remaining modules,
  ground classification leaves in canonical CCO classes, generate synth
  corpora per domain, train models. See
  [project_ontology_migration_to_aegir](../../../). 
- **Applied — Atelier / signals**: a CCO-rooted classification taxonomy,
  public-data fixtures (GitTables/SOTAB) that *exercise* every reachable
  module, and a coverage evaluation that reports module-level completeness.

## The classification-target nuance

Every column Atelier classifies is an **Information Content Entity**
(`cco:InformationContentEntity` ≡ `atelier:ICE` ≡ `cco:ont00000958`). The
other ten CCO modules describe what an ICE is *about* — its referent. A
column of country names is a `DesignativeICE` (`ont00000686`) that
*designates* a **Geospatial** entity; a column of masses is a
`DescriptiveICE` (`ont00000853`) that *describes* a **Quality**. So:

> Comprehensive CCO coverage = the vocabulary's ICE leaves collectively
> denote referents spanning all 11 CCO modules, across the ICE trichotomy
> (Designative / Descriptive / Prescriptive).

The taxonomy stays ICE-rooted (the target is always an ICE); the CCO module
is the **referent domain** carried as a second axis on each leaf.

## Coverage matrix (11 CCO modules)

Reachability = which public annotation task surfaces the domain. **CTA**
(Column Type Annotation, GitTables + SOTAB) labels a column's entity/value
type; **CPA** (Column Property Annotation, SOTAB) labels the *relation* a
column expresses — the only vehicle for the relational module.

| # | CCO module | CCO anchor (readable) | Reachable via | Applied status (fixture) | Owner of grounding |
|---|---|---|---|---|---|
| 1 | **Information Entity** | `cco:InformationContentEntity` (`ont00000958`) | CTA ★★★ | ✓ strong — id, name, title, description, genre | Aegir (imported) |
| 2 | **Agent** | `cco:Agent` / Person, Organization | CTA ★★★ | ✓ author, publisher | Aegir (imported) |
| 3 | **Time** | `cco:TemporalRegion` | CTA ★★★ | ✓ date, duration, year | Aegir (todo) |
| 4 | **Quality** | `cco:Quality` (BFO quality) | CTA ★★★ | ✓ length, width (measure) | Aegir (todo) |
| 5 | **Units of Measure** | `cco:MeasurementUnit` | CTA ★★☆ (implicit) | ◐ via measure values | Aegir (todo) |
| 6 | **Geospatial** | `cco:GeospatialRegion` | CTA ★★★ (SOTAB-rich) | ◐ only `state` | Aegir (todo) |
| 7 | **Currency Unit** | `cco:CurrencyUnit` | CTA ★★☆ | ◐ price | Aegir (todo) |
| 8 | **Event** | `cco:Act` / process (occurrent) | CTA ★☆☆ | ✗ | Aegir (todo) |
| 9 | **Artifact** | `cco:Artifact` | CTA ★☆☆ (thin in web tables) | ✗ | Aegir (todo) |
| 10 | **Facility** | `cco:Facility` | CTA ★☆☆ (thin) | ✗ | Aegir (todo) |
| 11 | **Extended Relation** | object properties (is_input_of, affects…) | **CPA** ✓ | ✗ (needs CPA slice) | Aegir (imported) |

Readable `cco:` labels follow the vocab-README convention; canonical
`ont…` IRIs resolve from the module TTLs (Aegir's grounding job). Only the
ICE IRIs are pinned above because they are the classification root.

### Reachability is conditioned on table shape, not fundamental

The ✗/◐ on **Units of Measure** and **Extended Relation** above hold only
for **wide / relational** tables, where units hide inside Quality *values*
(`"5.0 m"`) and relations hide in the *schema* (the column's property).
They are **not** fundamental CCO gaps. **Entity-Attribute-Value (EAV)**
tables — relational support in progress — relocate both axes into column
*content* and thereby surface both modules:

- the EAV **attribute/property** column holds relation/property names as
  values (`mass`, `has_currency`) → **Extended Relation** is classifiable
  column content, not a schema-level CPA annotation;
- the EAV **unit** column holds units as values (`kg`, `m`, `USD`) →
  **Units of Measure** is a column type, with no value-level unit detector.

So full 11/11 coverage follows from admitting EAV-pattern tables, which is
strictly more general than the CPA-only path (EAV reaches *both* residual
modules at once). CPA remains a complementary route to Extended Relation
for wide tables.

## What the public data reaches today

- **Strong from CTA** (GitTables + SOTAB): Information Entity, Agent, Time,
  Quality, Geospatial — 5 modules with abundant labeled columns.
- **Moderate from CTA**: Units of Measure (units ride on quality values),
  Currency Unit (Currency/price types).
- **Recovered by strided scanning**: Event, Artifact, Facility — sparse in
  the corpus *prefix* but present once the scan strides across all ~562k
  tables; the `test-gittables` fixture now includes all three.
- **EAV-gated**: Units of Measure and Extended Relation — surfaced by
  EAV-pattern tables (in progress; see above), with CPA
  (`/raid/datasets/sotab/sotab_cpa_*`) a complementary wide-table route to
  Extended Relation.

**Net: 9 of 11 modules already reached** by the strided GitTables CTA
fixture (Information Entity, Agent, Time, Quality, Geospatial, Currency,
Event, Artifact, Facility). The residual 2 (Units of Measure, Extended
Relation) are EAV-gated, not data-scarce — admitting EAV tables closes the
gap to 11/11.

## Division of labor

| Concern | Aegir (upstream) | Atelier / signals (applied) |
|---|---|---|
| Import all 11 CCO module TTLs | ✓ | consumes |
| Ground leaves in canonical CCO classes | ✓ | consumes vocab |
| Per-domain synth corpora + model training | ✓ | consumes models |
| CCO-rooted classification taxonomy | — | ✓ |
| Public fixtures spanning every reachable module | — | ✓ (extend `test-gittables` + SOTAB + CPA) |
| Module-level coverage evaluation | — | ✓ |

## Phased plan (applied side)

1. **Re-root the taxonomy by CCO module** — DONE. Every leaf carries its
   referent `cco_module` + ICE trichotomy class; the coverage matrix falls
   out of the taxonomy (populated modules vs gaps).
2. **Broaden the fixture across modules** — DONE. Strided scanning across
   the full corpus took the fixture to **9 of 11 modules** (30 leaf types).
3. **Admit EAV-pattern tables** (in progress) → Units of Measure + Extended
   Relation as classifiable column content (10–11/11). CPA
   (SOTAB-CPA-derived held-out set) is a complementary wide-table route to
   Extended Relation.
4. **Module-level coverage scenario** — a BDD/eval assertion that reports
   per-CCO-module coverage and fails if a targeted module regresses to zero
   exercised leaves.

See [`sotab-coverage.md`](sotab-coverage.md) for the SOTAB CTA label set and
[`test-gittables`](../../../src/atelier/classify/fixtures/test-gittables/PROVENANCE.md)
for the current fixture.
