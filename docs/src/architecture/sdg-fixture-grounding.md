<!-- Copyright (c) 2026 Cloudera, Inc.  All rights reserved. -->

# Grounding the test fixture in SDG — a requirements-driven recommendation for Aegir

**Status:** recommendation. Built and extended in Atelier; handed to Aegir,
which owns SDG, to adopt or refine.

## Why

The `test-gittables` fixture currently carries an invented `GT.*` namespace —
the fifth overlapping governance vocabulary alongside `sdg:`, `SIGDG:`,
`atelier-vocab.ttl`, and signals' `GDC.*`. Inventing a sixth dilutes the
refinement effort; the fix is to **iterate on the one shared surface**, the
Signals Data Governance ontology (`sdg:`, Aegir's
`src/aegir/ontology/sdg-vocab.ttl`, BFO 2020 + CCO grounded). See
[`feedback_iterate_shared_sdg_surface`] and [cco-coverage](cco-coverage.md).

## SDG already fits — the property map

SDG is property-centric and reuses CCO's ICE classes (`sdg:identifies`,
`sdg:hasValueType`, `sdg:hasIdentifier` all `rdfs:domain cco:DesignativeICE`).
Every axis the fixture encodes is already an SDG property — we were
reinventing them as `GT.*`:

| Fixture axis (what we built) | SDG property (already exists) |
|---|---|
| CTA type label on a column | `sdg:classifies` ("CTA / CPA label … it classifies") |
| the entity/value **type** (CTA) | `sdg:hasValueType` |
| the **relation** a column expresses (CPA) | `sdg:describesProperty` |
| a designative column → the entity it picks out | `sdg:identifies` |
| the **unit** of a measurement (our `unit: UNRESOLVED`) | `sdg:hasUnit` |
| sensitivity / governance tier | `sdg:atTier` (NIST PII / ISO 19944) |
| column statistics (cardinality, null rate, distribution) | `sdg:hasCardinality`, `sdg:hasNullRate`, `sdg:hasDistribution` |

So the CTA-vs-CPA distinction we encoded as `GT.CUR.CURRENCY` vs
`GT.REL.CURRENCY` is, in SDG, the same ICE `sdg:hasValueType currency` vs
`sdg:describesProperty currency` — one surface, two properties.

## The grounding model (replaces `GT.*`)

A fixture leaf is not a new code; it is an **SDG-grounded term**:

- the column is a `cco:InformationContentEntity` (Designative or
  Descriptive — the ICE trichotomy we already carry as `ice_class`);
- its **referent CCO module** is the existing `cco_module` axis;
- a CTA leaf is a **value type** (`sdg:hasValueType <term>`), a CPA leaf is a
  **relation** (`sdg:describesProperty <term>`);
- the concrete `<term>` is `sdg:`-namespaced, with the **DBpedia IRI as
  `dcterms:source` / our `definition_source`** and `cco:acronym` from the
  mnemonic — the CCO annotations we already emit;
- the **unit** is `sdg:hasUnit`, left unfilled where unresolved — i.e. our
  `unit: UNRESOLVED` absence *is* an unfilled `sdg:hasUnit` / `cco:has_token_unit`.

Net: the fixture stops minting `GT.*` and instead instantiates SDG value
types + relations, grounded in CCO. The taxonomy becomes a *test-scoped
subset of SDG*, and building it is an SDG coverage audit.

## Requirements for Aegir (adopt or refine)

The fixture needs concrete `sdg:` terms for the types/relations it exercises.
These are the requirements — proposed as SDG extensions for Aegir to adopt or
reshape:

1. **Value-type terms** for the GitTables CTA types covering 9 CCO modules
   (e.g. length, weight, currency, country, author, date, …) as
   `sdg:hasValueType` ranges, each grounded in its CCO module + DBpedia IRI.
2. **Relation terms** for the SOTAB CPA relations (author, publisher,
   publicationDate, currency, price, …) as `sdg:describesProperty` ranges
   (Extended Relation data face).
3. **`sdg:hasUnit` resolution path** — the unresolved-unit case (the only
   un-covered CCO module, Units of Measure) needs an EAV/unit producer to
   fill `sdg:hasUnit`; until then it is a positively-represented absence, not
   a silent gap. This is the highest-priority refinement.
4. **Namespace reconciliation** — pick `sdg:` as canonical and retire/redirect
   `SIGDG:`, `atelier-vocab.ttl`, and `GDC.*` so the surface stops forking.

## Plan

1. Rewire the fixture builder to emit SDG-grounded terms (CCO-module referent
   + `sdg:hasValueType` / `sdg:describesProperty` + DBpedia source), retiring
   `GT.*`. Unmapped types are recorded as SDG-coverage gaps.
2. Emit a machine-readable `sdg_requirements.json` from the build — the exact
   value-type/relation terms the fixture needs — as the artifact Aegir
   consumes.
3. Aegir adopts/refines into `sdg-vocab.ttl` + catalogs; Atelier re-consumes.
