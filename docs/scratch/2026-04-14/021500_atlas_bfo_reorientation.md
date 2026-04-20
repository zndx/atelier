# Atlas/BFO Reorientation: Findings & Plan

## Context

The atelier classification pipeline has drifted from Apache Atlas terminology.
Mock fixtures and dot-notation codes risk leaking proprietary ontology structure.
This note captures research findings and scoping decisions for a targeted
realignment — not a full rewrite, but surgical terminology changes that bring
us into Atlas alignment while preserving working internal mechanics.

## Terminology Mapping (Decided)

| Current Atelier | Apache Atlas | Decision |
|---|---|---|
| `CategorySet` | Glossary | **Keep** — Glossary is too colloquial; CategorySet is descriptive |
| `HierarchicalCategorySet` | Glossary + Categories | **Keep** — dot-code hierarchy maps well to Category nesting |
| `ReferenceCategory` | Term | **Rename candidate** — Atlas Terms carry labels, CURIEs, and belong to Categories |
| `ReferenceCategory.code` | (no Atlas equivalent) | **Rethink** — dot-notation leaks structure; see hierarchy options below |
| `ReferenceCategory.abbrev` | Term short code | **Rename to `term_code`** — aligns with Atlas Term + SIGDG CURIE convention |
| `ReferenceCategory.label` | Term display label | **Keep** — already correct |
| Entity (UI stat) | Entity (hive_table, hive_column) | **Already aligned** — things that get Terms assigned |
| Classification | Classification (Atlas type) | **Already aligned** — the governed type, not the process |
| `predicted_code` | (assignment of Term to Entity) | **Rename candidate** — `predicted_term` or `predicted_term_code` |

## Apache Atlas Concept Map

```
Glossary (our CategorySet)
  +-- Category (organizational nesting, arbitrarily deep)
  |     +-- child Categories
  |     +-- Terms organized here
  +-- Term (atomic vocabulary word)
        +-- abbreviation, display label
        +-- synonyms, antonyms, isA, classifies
        +-- assignedEntities (the column/table assignments)
        +-- classifications (propagate to assigned entities)

Classification (formal governed type)
  +-- superTypes / subTypes (inheritance tree)
  +-- propagates through lineage edges
  +-- drives Ranger access policies
  +-- can carry custom attributes
```

## BFO Grounding (Summary)

BFO (Basic Formal Ontology, ISO/IEC 21838-2) provides the upper ontology.
The relevant anchor for data governance:

```
Entity (BFO root)
  +-- Continuant
        +-- Generically Dependent Continuant
              +-- Information Content Entity (IAO)  <-- our Terms live here
```

All classification labels, sensitivity ratings, and governance metadata are
**Information Content Entities** (ICEs) — copyable content that depends on
some carrier (a database row, a policy document) but whose identity is the
pattern, not the medium.

"BFO-grounded" means every term in our vocabulary has an unbroken `is_a`
chain back to ICE. We don't need full OWL formalization — genus-differentia
definitions and single inheritance capture 90% of the value.

Key mid-level ontologies to reference:
- **IAO** (Information Artifact Ontology) — defines ICE
- **CCO** (Common Core Ontologies) — DoD/IC baseline; agents, events, info
- **IAO-Intel** — closest analogue: classifies information artifacts

## Hierarchy Representation Options

The dot-notation codes (1.1.1.1) encode tree position directly in the
identifier. This leaks structure and is brittle to restructuring. Options:

### Option A: Atlas Classification superTypes (pragmatic winner)

Create Atlas classification types with mnemonic paths:

```
SENS                              superTypes: []
SENS.PID                          superTypes: [SENS]
SENS.PID.CI                       superTypes: [SENS.PID]
SENS.PID.CI.EMAIL                 superTypes: [SENS.PID.CI]
```

Atlas supports dots in type names. Tagging an entity with `SENS.PID.CI.EMAIL`
makes it discoverable under all ancestors via inheritance. Native search,
native propagation, no external systems.

**Pros:** Atlas-native, hierarchy via inheritance, no GUIDs
**Cons:** Type names leak hierarchy, ~300 type definitions

### Option B: SKOS notation (canonical standard)

SKOS `skos:notation` was designed exactly for this: carry classification
codes alongside explicit hierarchy expressed via `skos:broader`/`skos:narrower`.

```turtle
ex:EMAIL a skos:Concept ;
    skos:notation "1.1.1.1"^^ex:SIGDGNotation ;
    skos:prefLabel "Email Address"@en ;
    skos:broader ex:ContactInformation .
```

Notation is independent of hierarchy. Concepts can carry multiple notations.
XKOS extension adds `xkos:depth` for statistical classification levels.

**Pros:** W3C standard, decouples code from hierarchy, interchange format
**Cons:** RDF-native, needs mapping layer to Atlas

### Option C: Opaque CURIEs (OBO Foundry pattern)

Mint opaque identifiers like `SIGDG:0100` with hierarchy only in `is_a`:

```
SIGDG:0100  "Email Address"   is_a  SIGDG:0042 "Contact Information"
```

Identifiers reveal nothing about tree position. Stable through restructuring.

**Pros:** Maximum decoupling, stable, no structure leak
**Cons:** Requires ID registry, adds indirection, no natural Atlas fit

### Option D: Hybrid (recommended)

Combine Atlas superTypes (Option A) with SKOS-informed attributes:

1. Atlas classification types use **mnemonic dot-paths** as names
   (`SENS.PID.CI.EMAIL`) with `superTypes` expressing hierarchy
2. Carry original dot-notation as a classification **attribute**
   (`notation: "1.1.1.1"`) following SKOS `skos:notation` pattern
3. Maintain a SKOS JSON-LD file as canonical interchange format
4. Internal code uses mnemonic paths (not dot-codes) for tree navigation

This gives native Atlas search/inheritance, preserves original codes as
queryable metadata, uses mnemonics instead of opaque numbers, and avoids
GUID tree navigation entirely.

### SHACL Assessment

SHACL is a constraint/validation language for RDF, not a knowledge
organization system. It can validate hierarchy well-formedness but has no
"notation" concept. Not applicable here. The user may have been thinking
of SKOS, which is the correct standard.

## Leak Analysis

### What leaks now (committed to git)

1. **`mock_annotations.json`** — 24 entries with real hierarchy codes
   (1.1.1.1, 1.1.2.3, etc.), sensitivity ratings (non_corp, emp_contractor,
   individual, corp with scale 1-3), and term structure. Generic labels
   (Email Address, SSN) but hierarchy reveals classification framework.

2. **`mock_tables.json`** — reference-code fields use real hierarchy codes.
   Combined with mock_annotations, someone can reconstruct the taxonomy tree.

3. **`real_data_loader.py` `_HEADER_MAP`** — exposes exact column names from
   proprietary annotations.csv ("Specifics, Examples and/or Additional Context",
   "NON_CORP", "EMP, CONTRACTOR").

4. **`synth.py` docstring** — references `signals/scripts/generate_meta_tagging_train.py`.

### What doesn't leak

- Real data files (~/local/tmp/meta-tagging/) — external, gitignored
- Real annotations (296 records in hive) — never cached in git
- CAI-specific credentials — env vars only

### Remediation

- Replace dot-notation codes in fixtures with mnemonic paths or CURIE-style IDs
- Replace sensitivity column names with generic terms
- Move `_HEADER_MAP` to a config file outside git (or make it dynamic)
- Audit synth.py references to signals internals

## Scope for Targeted Refactoring

**In scope (next session):**

1. Add `_normalize_record()` handling for Atlas-compatible key formats
   (already done in CAI fixes — landing separately)
2. Consider renaming `ReferenceCategory` → `Term` (or `VocabularyTerm`)
3. Consider renaming `code` → `notation` (SKOS-aligned) in the data model
4. Replace dot-notation codes in mock fixtures with mnemonic paths
5. Add SKOS `notation` as an attribute alongside mnemonic identifiers
6. Document the Atlas concept mapping in architecture docs

**Out of scope (future):**

- Full Atlas REST client integration (port from signals)
- OWL/RDF formalization
- Classification propagation via lineage
- Ranger policy integration
- BFO OWL import (we adopt the principles, not the formalism)

## References

- [Apache Atlas Glossary](https://atlas.apache.org/2.0.0/Glossary.html)
- [Apache Atlas Classification Propagation](https://atlas.apache.org/2.0.0/ClassificationPropagation.html)
- [SKOS Reference (W3C)](https://www.w3.org/TR/skos-reference/)
- [BFO ISO/IEC 21838-2](https://basic-formal-ontology.org/)
- [IAO (Information Artifact Ontology)](https://github.com/information-artifact-ontology/IAO)
- [CCO (Common Core Ontologies)](https://github.com/CommonCoreOntology/CommonCoreOntologies)
- [XKOS (Extended SKOS)](https://rdf-vocabulary.ddialliance.org/xkos/xkos.html)
