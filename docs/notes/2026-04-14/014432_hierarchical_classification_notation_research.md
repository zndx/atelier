<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Hierarchical Classification Notation for Apache Atlas

Research into representing dot-notation hierarchy codes (e.g., 1.1.1.1, 1.1.2.3)
in an Apache Atlas-compatible way without requiring full GUID-based tree navigation.

**Context**: The governance ontology has 302 classification codes across 7 depth
levels, ranging from top-level categories (0.0, 1.1) down to leaf-level data
elements (1.1.1.1.1.1.9). Each code carries an annotation mnemonic (e.g., C_PID,
PAN, DOB) and a human-readable name.

---

## 1. SHACL (Shapes Constraint Language)

**What it is**: W3C standard (2017) for validating RDF graph structure. Defines
"shapes" (constraints) that RDF nodes must satisfy.

**Hierarchy support**: SHACL can validate class hierarchies via `rdfs:subClassOf`
traversal and constrain property values via `sh:node`, `sh:class`, etc. Shapes
targeting a class are inherited by all subclasses.

**Notation for classification paths**: **None**. SHACL has no built-in concept of
"notation" or "classification code." It is a *constraint language*, not a
*knowledge organization system*. It validates that graphs conform to expected
shapes; it does not model taxonomies or classification hierarchies itself.

**Verdict**: SHACL is the wrong tool for this problem. It could *validate* that
a classification hierarchy is well-formed (e.g., "every concept with depth > 1
must have exactly one skos:broader link"), but it cannot *represent* the
hierarchy or carry the dot-notation codes. The user may have been thinking of
SKOS (see below), which is the W3C standard that actually addresses this.

| Criterion | Assessment |
|-----------|-----------|
| Atlas compatibility | None (different domain entirely) |
| Preserves dot-notation | No notation concept exists |
| Complexity of adoption | High (requires RDF infrastructure) |
| Avoids leaking structure | N/A |

**Sources**: [W3C SHACL Spec](https://www.w3.org/TR/shacl/),
[Ontotext SHACL Fundamentals](https://www.ontotext.com/knowledgehub/fundamentals/what-is-shacl/)

---

## 2. SKOS (Simple Knowledge Organization System)

**What it is**: W3C Recommendation (2009) for representing controlled
vocabularies, taxonomies, thesauri, and classification schemes as RDF. This is
the canonical standard for exactly this problem domain.

**Hierarchy support**: `skos:broader` / `skos:narrower` express direct
parent-child relationships. `skos:broaderTransitive` / `skos:narrowerTransitive`
handle indirect ancestry. `skos:hasTopConcept` links schemes to roots.

**The `skos:notation` property**: This is the key property. It is an
`owl:DatatypeProperty` designed specifically for carrying classification codes:

```turtle
ex:PaymentCardData a skos:Concept ;
    skos:notation "1.1.1.1.1.1"^^ex:SIGDGNotation ;
    skos:prefLabel "Payment Card Data"@en ;
    skos:broader ex:PaymentData ;
    skos:inScheme ex:DataGovernanceScheme .
```

Key design points:
- Notation is a **typed literal** with a user-defined datatype URI (e.g.,
  `ex:SIGDGNotation`), making it unambiguous which coding system the "1.1.1.1.1.1"
  belongs to.
- Notation is **independent of hierarchy** -- the dot-notation is a label, not a
  structural assertion. Hierarchy is always expressed via `skos:broader`/`skos:narrower`.
- A concept can have **multiple notations** (e.g., the dot-code AND the mnemonic:
  `skos:notation "C_PCD"^^ex:SIGDGMnemonic`).
- Notations need not encode hierarchy at all -- they are opaque identifiers that
  happen to have a recognizable pattern.

**XKOS extension**: For statistical classifications specifically,
[XKOS](https://rdf-vocabulary.ddialliance.org/xkos.html) extends SKOS with
`xkos:ClassificationLevel` (with `xkos:depth` property) and
`xkos:numberOfLevels`. Real-world example from NACE:

```turtle
<http://id.insee.fr/codes/nafr2/classe/27.12>
    skos:notation "27.12" ;
    skos:prefLabel "Fabrication of electrical distribution..."@fr .
```

**Verdict**: SKOS is the strongest conceptual model for this problem. The
`skos:notation` property was designed for exactly this use case -- carrying
classification codes alongside explicit hierarchy relationships. The question is
how to bridge SKOS semantics into Apache Atlas's type system.

| Criterion | Assessment |
|-----------|-----------|
| Atlas compatibility | Indirect -- SKOS is RDF; Atlas is its own type system. Requires a mapping layer. |
| Preserves dot-notation | Yes, via `skos:notation` typed literal |
| Complexity of adoption | Medium (SKOS model is simple; bridging to Atlas requires design work) |
| Avoids leaking structure | Yes -- notation is an opaque label, hierarchy is in `broader`/`narrower` |

**Sources**: [W3C SKOS Reference](https://www.w3.org/TR/skos-reference/),
[SKOS Classification Publishing Guide](https://www.w3.org/wiki/SkosDev/ClassificationPubGuide),
[XKOS Best Practices](https://linked-statistics.github.io/xkos/xkos-best-practices.html),
[SKOS Primer](https://www.w3.org/TR/skos-primer/)

---

## 3. Atlas Classification superTypes (Dot-Notation in Type Names)

**What it is**: Atlas classifications support inheritance via `superTypes` /
`subTypes` arrays in `AtlasClassificationDef`. A classification type can declare
a parent, inheriting its `entityTypes` restrictions and attributes.

**Dot-notation in type names**: Initially, Atlas prohibited dots in type names
(ATLAS-1408). This was later reversed specifically for classification types
(ATLAS-1434: "updated typename validation to allow '.' for classifications").
So classification types CAN contain dots in their names.

This means you could create a classification hierarchy like:

```json
[
  {"name": "SENS",                       "superTypes": []},
  {"name": "SENS.PID",                   "superTypes": ["SENS"]},
  {"name": "SENS.PID.PD",                "superTypes": ["SENS.PID"]},
  {"name": "SENS.PID.PD.FD",             "superTypes": ["SENS.PID.PD"]},
  {"name": "SENS.PID.PD.FD.PAY",         "superTypes": ["SENS.PID.PD.FD"]},
  {"name": "SENS.PID.PD.FD.PAY.PCD",     "superTypes": ["SENS.PID.PD.FD.PAY"]},
  {"name": "SENS.PID.PD.FD.PAY.PCD.PAN", "superTypes": ["SENS.PID.PD.FD.PAY.PCD"]}
]
```

**Advantages**:
- Native Atlas -- no external system needed.
- Type inheritance means tagging an entity with `SENS.PID.PD.FD.PAY.PCD.PAN`
  automatically makes it findable under all ancestor classifications.
- Classification propagation works out of the box.
- The dot-notation in the name is human-readable and self-documenting.

**Risks**:
- 302 classification types is a lot of type definitions. Atlas type systems
  are not designed for thousands of types, but 302 is within reason.
- 7 levels of superType depth is unusual but should work.
- Type names leak the hierarchy structure directly -- anyone with Atlas API
  access can see the full tree topology in the type names.
- Renaming or restructuring requires type migration (types are harder to change
  than entity attributes).
- Using mnemonics (PAN, DOB) vs. numeric codes (1.1.1.1.1.1.1) in the name
  is a one-way decision.

**Important nuance**: The dot in the type name is cosmetic -- it encodes
hierarchy in the NAME, not in the structure. The actual hierarchy is in
`superTypes`. You could use underscores or any other separator. The dot just
happens to match the source ontology's notation.

| Criterion | Assessment |
|-----------|-----------|
| Atlas compatibility | Native -- this IS Atlas's mechanism |
| Preserves dot-notation | Yes, directly in type names (or via attributes) |
| Complexity of adoption | Low (straightforward API calls to create type hierarchy) |
| Avoids leaking structure | **No** -- type names are visible to all Atlas users |

**Sources**: [Atlas Type System](https://atlas.apache.org/2.0.0/TypeSystem.html),
[AtlasClassificationDef API](https://atlas.apache.org/api/v2/json_AtlasClassificationDef.html),
[Atlas release-log (ATLAS-1434)](https://github.com/apache/atlas/blob/master/release-log.txt),
[ClearPeaks Atlas Custom Types](https://www.clearpeaks.com/data-governance-with-apache-atlas-custom-types-in-atlas-part-3-of-3/)

---

## 4. OBO/CURIE Approach

**What it is**: OBO Foundry ontologies use opaque numeric identifiers in CURIE
(Compact URI) format: `PREFIX:LOCALID`, e.g., `OBI:0000011`, `UBERON:0000948`.
The identifier is intentionally meaningless -- hierarchy is expressed solely
through `is_a` (subClassOf) relationships, never encoded in the ID.

**How it would apply**: Mint a prefix (e.g., `SIGDG`) and assign opaque numeric
IDs:

```
SIGDG:0001  -- Personally Identifiable Data  (was 1.1)
SIGDG:0002  -- Personal Data                 (was 1.1.1)
SIGDG:0003  -- Financial Data                (was 1.1.1.1)
SIGDG:0100  -- Payment Card Number           (was 1.1.1.1.1.1.1)
```

Hierarchy metadata lives in a separate structure (RDF `is_a` triples, or Atlas
`superTypes`, or a lookup table). The dot-notation becomes a secondary label or
`skos:notation`, not the primary identifier.

**Advantages**:
- Maximum decoupling between identifier and structure.
- Safe to share externally -- SIGDG:0100 reveals nothing about where it sits
  in the tree.
- Follows a well-established pattern from the biomedical ontology community.
- IDs are stable through restructuring -- moving a concept in the hierarchy
  doesn't change its ID.

**Disadvantages**:
- Requires maintaining a registry (ID-to-concept mapping).
- Opaque IDs are harder for humans to work with day-to-day.
- Two layers of indirection: CURIE -> concept -> hierarchy position.
- Does not directly fit Atlas's classification type system (would need to be
  stored as attributes on a generic classification, not as type names).

| Criterion | Assessment |
|-----------|-----------|
| Atlas compatibility | Indirect -- CURIEs would be attribute values, not type names |
| Preserves dot-notation | No (intentionally replaces it with opaque IDs) |
| Complexity of adoption | High (requires ID registry, mapping layer, cultural shift) |
| Avoids leaking structure | **Yes** -- this is the primary advantage |

**Sources**: [OBO Foundry ID Policy](http://obofoundry.org/id-policy.html),
[OBO Foundry Wikipedia](https://en.wikipedia.org/wiki/OBO_Foundry),
[OAK CURIEs and URIs Guide](https://incatools.github.io/ontology-access-kit/guide/curies-and-uris.html)

---

## 5. Dublin Core / ISO 25964

**ISO 25964** is the international standard for thesauri (Part 1: structure,
Part 2: interoperability). It models concepts with BT/NT (broader/narrower term)
relationships and supports notation schemes for classified arrangements.

**Key points**:
- Concept groups may have "a scheme of notation distinct from that used for
  concepts," providing classified arrangement that complements the generic
  hierarchy.
- The standard defines a data model and XML schema for interchange.
- Dublin Core attributes (title, creator, date, etc.) describe the thesaurus
  as a whole.
- ISO 25964 was designed to be compatible with SKOS -- the
  [SKOS-Thes documentation](https://www.dublincore.org/specifications/skos-thes/ns/)
  provides a mapping between ISO 25964 and SKOS concepts.

**Verdict**: ISO 25964 validates the approach of having separate notation schemes
alongside hierarchical relationships, but it does not add anything beyond what
SKOS already provides. It is a specification-level standard (for building
thesaurus software) rather than an implementation-level vocabulary (for tagging
data). The SKOS mapping is the actionable path.

| Criterion | Assessment |
|-----------|-----------|
| Atlas compatibility | None directly (interchange standard, not a runtime system) |
| Preserves dot-notation | Yes (notation schemes are a first-class concept) |
| Complexity of adoption | Medium-high (standard is behind ISO paywall; SKOS mapping is the practical route) |
| Avoids leaking structure | Depends on notation design |

**Sources**: [ISO 25964 Wikipedia](https://en.wikipedia.org/wiki/ISO_25964),
[NISO ISO 25964](https://www.niso.org/standards-committees/iso-25964),
[SKOS-Thes Namespace](https://www.dublincore.org/specifications/skos-thes/ns/)

---

## 6. Atlas qualifiedName Convention

**What it is**: Atlas uses dot-separated `qualifiedName` for glossary categories:

```
Loans.Customer@HortoniaBank
```

Format: `CategoryName.ParentCategoryQualifiedName@GlossaryName`

The qualifiedName auto-updates when hierarchy changes (reparenting a category
updates its qualifiedName and all descendants).

**How it could apply**: Rather than encoding hierarchy in classification TYPE
names, use Atlas Glossary categories to model the ontology:

```
PaymentCardData.PaymentData.FinancialData.PersonalData.PID.Sensitive@DataGovernance
```

Then associate glossary terms with classifications via Atlas's built-in
glossary-classification linkage.

**Advantages**:
- Uses Atlas's own hierarchical naming convention.
- Glossary categories are designed for exactly this kind of taxonomic structure.
- `qualifiedName` updates automatically on restructuring.
- Separates the taxonomy (glossary) from the tagging mechanism (classifications).

**Disadvantages**:
- Glossary qualifiedNames get very long at depth 7.
- Glossary categories are not the same as classifications -- you would need
  both a glossary structure AND classification types, with a mapping between them.
- The dot-separated qualifiedName is Atlas-internal; it is not the same as
  carrying the ontology's own dot-notation (1.1.1.1).
- Terms cannot contain dots (only the category hierarchy uses dots).

| Criterion | Assessment |
|-----------|-----------|
| Atlas compatibility | Native (built-in glossary feature) |
| Preserves dot-notation | Partially (Atlas's own dot convention, not the ontology's) |
| Complexity of adoption | Medium (requires both glossary and classification setup) |
| Avoids leaking structure | No (qualifiedName exposes full hierarchy path) |

**Sources**: [Atlas Glossary](https://atlas.apache.org/1.1.0/Glossary.html),
[Atlas Glossary 2.0](https://atlas.apache.org/2.0.0/Glossary.html)

---

## Comparative Summary

| Option | Atlas-Native | Dot-Notation | Complexity | Structure Privacy |
|--------|:---:|:---:|:---:|:---:|
| SHACL | -- | -- | High | N/A |
| SKOS + skos:notation | Indirect | Yes | Medium | Yes |
| Atlas superTypes (dot names) | Yes | Yes | Low | No |
| OBO/CURIE | Indirect | No (opaque) | High | Yes |
| ISO 25964 | -- | Yes | Medium-High | Depends |
| Atlas qualifiedName | Yes | Partial | Medium | No |

---

## Recommended Approach: Hybrid (Atlas superTypes + SKOS-informed attributes)

The strongest option combines Atlas-native classification inheritance with
SKOS-informed metadata:

1. **Atlas classification types with mnemonic superType chains** (Option 3):
   Create the 302 classification types using short mnemonic names with
   `superTypes` expressing hierarchy. Use underscores or short dot-paths, not
   the full numeric code, as the type name:

   ```
   SENS > PID > PD > FD > PAY > PCD > PAN
   ```

   Or with dot-delimited segments for readability:
   ```
   SENS.PID.PD.FD.PAY.PCD.PAN
   ```

2. **Carry the ontology's dot-notation as a classification attribute**
   (SKOS `skos:notation` concept): Add an `ontology_code` attribute to
   the root classification type that stores the original dot-notation:

   ```json
   {
     "name": "SENS.PID.PD.FD.PAY.PCD.PAN",
     "superTypes": ["SENS.PID.PD.FD.PAY.PCD"],
     "attributeDefs": [
       {"name": "ontology_code", "typeName": "string", "defaultValue": "1.1.1.1.1.1.1"},
       {"name": "ontology_mnemonic", "typeName": "string", "defaultValue": "PAN"},
       {"name": "ontology_label", "typeName": "string", "defaultValue": "Payment Card Number"}
     ]
   }
   ```

3. **Maintain a SKOS-format canonical representation** outside Atlas for
   interchange, documentation, and tooling. This can be a Turtle/JSON-LD file
   that serves as the source-of-truth for the ontology, with Atlas being one
   downstream consumer.

This approach gives you:
- Native Atlas search and inheritance (tag something PAN, find it under SENS)
- The original dot-notation preserved as metadata (queryable via attributes)
- Structure privacy is partially addressed (mnemonics are less revealing than
  the full numeric hierarchy, though the superType chain is still visible)
- A standards-based interchange format (SKOS) for non-Atlas consumers
- No GUID tree navigation needed
