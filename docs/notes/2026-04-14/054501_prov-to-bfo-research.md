<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# PROV-to-BFO Mapping Repository Research

**Repository:** https://github.com/BFO-Mappings/PROV-to-BFO
**Paper:** Prudhomme et al., "A semantic approach to mapping the Provenance Ontology to Basic Formal Ontology," *Sci Data* 12, 282 (2025). https://doi.org/10.1038/s41597-025-04580-1
**License:** CC0-1.0
**DOI:** https://doi.org/10.5281/zenodo.11338700

## Repository Structure

```
prov-bfo-directmappings.ttl     # BFO mappings (production)
prov-ro-directmappings.ttl      # Relation Ontology mappings (production)
prov-cco-directmappings.ttl     # Common Core Ontologies mappings (production)
catalog-v001.xml
example-usage/
  example-ontology-full-imports.ttl
src/
  prov-mappings-edit.ttl        # Editor's development file (imports all three + test instances)
  Makefile                      # ROBOT-based build/test pipeline
  sparql/                       # SPARQL queries for validation
    unmapped-terms.rq           # Totality check: are any PROV terms unmapped?
    unsubsumed-object-properties.rq
    SSSOM-mappings.rq           # Extract SSSOM-format CSV
    candidate-superproperties.rq
    candidate-superproperties-complex.rq
    count-prov-terms.rq
    count-example-instances.rq
    get-imported-terms.rq
    construct/
      prov-triples.rq           # Materialize PROV-only triples for deductive diff
  imports/
    BFO/                        # bfo-core.ttl, CCO merged, RO
    PROV/                       # prov-ontologies.ttl
    SSN/                        # SOSA-PROV alignment (bonus)
    RO-imports-extracted.ttl    # Subset of RO terms used in mappings
    RO-imports-extracted.txt    # Term list for ROBOT extract
.github/workflows/
  test-mappings.yml             # CI: reason + verify on PRs
```

## Three-Tier Architecture

Production mapping files are separated into three independently importable modules:
1. **prov-bfo-directmappings.ttl** -- core BFO 2020 mappings (no external dependencies beyond BFO)
2. **prov-ro-directmappings.ttl** -- RO mappings (imports prov-bfo)
3. **prov-cco-directmappings.ttl** -- CCO mappings (imports prov-bfo)

Users import only what they need. The editor's file (`src/prov-mappings-edit.ttl`) merges all three plus full source ontologies and test instances for development.

## Mapping Idioms

### 1. Reified Annotation Axioms with SSSOM Labels

Every mapping is encoded as an OWL annotation axiom (`owl:Axiom`) wrapping the actual mapping triple. This allows attaching:
- `sssom:object_label` -- human-readable target description
- `rdfs:comment` -- philosophical/ontological justification

```turtle
[] rdf:type owl:Axiom ;
   owl:annotatedSource   prov:Activity ;
   owl:annotatedProperty owl:equivalentClass ;
   owl:annotatedTarget   obo:BFO_0000015 ;
   sssom:object_label    "process" ;
   rdfs:comment "A prov:Activity is equivalent to process because it transpires
     over time while not being a temporal region itself."@en .
```

### 2. Direct Equivalence (`owl:equivalentClass`)

Used when PROV and BFO concepts are coextensive:

| PROV Class | BFO/CCO Target | Label |
|---|---|---|
| `prov:Activity` | `obo:BFO_0000015` | process |
| `prov:InstantaneousEvent` | `obo:BFO_0000035` | process boundary |
| `prov:Location` | `obo:BFO_0000029` | site |
| `prov:Start` | `cco:ont00000197` | Process Beginning |
| `prov:End` | `cco:ont00000083` | Process Ending |

### 3. Subsumption (`rdfs:subClassOf`)

Used when PROV concepts are narrower than BFO concepts:

| PROV Class | BFO Target | Label |
|---|---|---|
| `prov:Role` | `obo:BFO_0000023` | role |
| `prov:Bundle` | `obo:BFO_0000031` | generically dependent continuant |
| `prov:Plan` | `obo:BFO_0000031` | generically dependent continuant |
| `prov:Dictionary` | `obo:BFO_0000031` | generically dependent continuant |
| `prov:KeyEntityPair` | `obo:BFO_0000031` | generically dependent continuant |

### 4. Complex Class Expressions

**prov:Entity** -- mapped to a union:
```turtle
prov:Entity rdfs:subClassOf
  [ owl:unionOf ( [ owl:intersectionOf ( obo:BFO_0000004      # independent continuant
                                          [ owl:complementOf obo:BFO_0000006 ] )  # MINUS spatial region
                   ]
                   obo:BFO_0000031    # generically dependent continuant
                   obo:BFO_0000020    # specifically dependent continuant
                 )
  ]
```
Justification: Entities are continuants (persist through time), but spatial regions are causally inert and cannot participate in processes.

**prov:Agent** (BFO layer) -- mapped to intersection:
```turtle
prov:Agent rdfs:subClassOf
  [ owl:intersectionOf ( obo:BFO_0000040           # material entity
                          [ owl:onProperty obo:BFO_0000056 ;     # participates in
                            owl:someValuesFrom prov:Activity ]
                          [ owl:onProperty obo:BFO_0000196 ;     # bears
                            owl:someValuesFrom [ owl:intersectionOf (
                              obo:BFO_0000023                     # role
                              [ owl:onProperty obo:BFO_0000054 ;  # realized in
                                owl:someValuesFrom prov:Activity ]
                            ) ] ]
                        )
  ]
```
A material entity that participates in an Activity AND bears a role realized in that Activity.

**prov:Influence** -- exclusive disjunction:
```turtle
prov:Influence rdfs:subClassOf
  [ owl:intersectionOf ( [ owl:unionOf ( obo:BFO_0000015    # process
                                          obo:BFO_0000035 ) ]  # process boundary
                          [ owl:complementOf [ owl:intersectionOf (
                            obo:BFO_0000015 obo:BFO_0000035 ) ] ]
                        )
  ]
```
Either a process OR a process boundary, but not both simultaneously.

### 5. Property Subsumption (`rdfs:subPropertyOf`)

**BFO layer** -- Most PROV Activity-to-Entity properties map to `has participant` / `participates in`:

| PROV Property | BFO Target | OBO ID |
|---|---|---|
| `prov:used` | has participant (at some time) | `BFO_0000057` |
| `prov:generated` | has participant (at some time) | `BFO_0000057` |
| `prov:wasStartedBy` | has participant (at some time) | `BFO_0000057` |
| `prov:wasEndedBy` | has participant (at some time) | `BFO_0000057` |
| `prov:wasAssociatedWith` | has participant (at some time) | `BFO_0000057` |
| `prov:invalidated` | has participant (at some time) | `BFO_0000057` |
| `prov:wasGeneratedBy` | participates in (at some time) | `BFO_0000056` |
| `prov:wasInvalidatedBy` | participates in (at some time) | `BFO_0000056` |
| `prov:qualifiedStart` | has temporal part | `BFO_0000121` |
| `prov:qualifiedEnd` | has temporal part | `BFO_0000121` |
| `prov:qualifiedUsage` | has temporal part | `BFO_0000121` |
| `prov:hadMember` | has continuant part | `BFO_0000178` |

**RO layer** -- Parallel property mappings for interoperability:

| PROV Property | RO Target | OBO ID |
|---|---|---|
| `prov:influenced` | causally related to | `RO_0002410` |
| `prov:influencer` | causally related to | `RO_0002410` |
| `prov:wasInfluencedBy` | causally related to | `RO_0002410` |
| `prov:wasAttributedTo` | causally influenced by | `RO_0002559` |
| `prov:wasDerivedFrom` | causally influenced by | `RO_0002559` |
| `prov:wasInformedBy` | causal relation between processes | `RO_0002501` |
| `prov:hadMember` | has member | `RO_0002351` |
| `prov:activity` | causal relation between processes | `RO_0002501` |

**CCO layer** -- More specific semantic relationships:

| PROV Property | CCO Target | CCO ID |
|---|---|---|
| `prov:wasAssociatedWith` | has agent | `ont00001833` |
| `prov:wasGeneratedBy` | is output of | `ont00001816` |
| `prov:generated` | has output | `ont00001986` |
| `prov:used` | affects | `ont00001834` |
| `prov:invalidated` | affects | `ont00001834` |
| `prov:wasInvalidatedBy` | is affected by | `ont00001886` |
| `prov:agent` | has agent | `ont00001833` |
| `prov:pairEntity` | is about | `ont00001808` |

### 6. Property Chain Axioms

**prov:hadPlan** uses a property chain through CCO:
```turtle
prov:hadPlan owl:propertyChainAxiom
  ( [ owl:inverseOf prov:qualifiedAssociation ]
    cco:ont00001920 ) ;   # prescribed by
```
"If an Association is the qualified association of some Activity, and that Activity is prescribed by some Plan, then that Association 'had plan' that Plan."

### 7. SWRL Rules

Eight SWRL rules in the BFO file handle `prov:atLocation` bidirectional mapping:

| Rule | Pattern | Conclusion |
|---|---|---|
| 1 | Activity + atLocation(x,y) | occurs_in(x,y) |
| 2 | InstantaneousEvent + atLocation(x,y) | occurs_in(x,y) |
| 3 | occurs_in(x,y) + Location(y) | atLocation(x,y) |
| 4 | Entity + atLocation(x,y) | located_in(x,y) |
| 5 | Agent + atLocation(x,y) | located_in(x,y) |
| 6 | located_in(x,y) + Location(y) | atLocation(x,y) |
| 7 | hadRole(x,y) + Process(x) | has_participant(x,y) |
| 8 | entity(x,y) + Process(x) | has_participant(x,y) |

SWRL is needed because `prov:atLocation` is polymorphic: for occurrents it maps to `occurs_in`, for continuants it maps to `located_in`. OWL alone cannot express this type-conditional property mapping.

Six additional SWRL rules in the CCO file handle:
- Dictionary involvement affects
- EntityInfluence entity -> has_input
- ServiceDescription carrier chain -> describesService
- Query service chain -> has_query_service
- Pingback service chain -> pingback
- Insertion insertedKeyEntityPair -> has_output

### 8. SKOS Relations for Non-Mappable Cases

When BFO/CCO semantics conflict with PROV semantics, `skos:relatedMatch` is used instead of subsumption:

```turtle
prov:ServiceDescription skos:relatedMatch cco:ont00000958 .  # Information Content Entity
```
Justification: "prov:ServiceDescription is a prov:SoftwareAgent, which is an independent continuant. By contrast, only generically dependent continuants such as information can be about or describe something in CCO."

Similarly for `prov:qualifiedGeneration` and `prov:qualifiedInvalidation` which are process boundaries (instantaneous) but would need to be processes for CCO `is output of` / `is affected by` to apply.

## Key Class Mappings Summary

### prov:Activity -> bfo:Process (equivalence)
The cleanest mapping. Activities transpire over time, are not temporal regions, and have participants. This is exactly BFO's definition of process.

### prov:Agent -> Material Entity with Role (complex)
**BFO layer**: subClassOf intersection -- material entity that participates in Activity AND bears a role realized in that Activity.
**RO layer**: Same structure but using `RO_0000087` (has role) instead of `BFO_0000196` (bears).
**CCO layer**: equivalentClass intersection -- `cco:Agent` that `is agent in` some Activity. This is stronger (equivalence vs. subsumption) because CCO has a dedicated Agent class.

CCO specializations:
- `prov:Person` = `cco:Person` AND `prov:Agent`
- `prov:Organization` = `cco:Organization` AND `prov:Agent`

### prov:Entity -> Continuant minus Spatial Region (complex)
Union of: (independent continuant MINUS spatial region) OR generically dependent continuant OR specifically dependent continuant. Spatial regions excluded because they are "causally inert."

### prov:Influence -> Process XOR Process Boundary (complex)
Exclusive disjunction: must be one or the other but not both simultaneously.

## CCO Modules Referenced

The CCO mapping imports the **merged CCO** (CommonCoreOntologiesMerged.ttl, v2.0-2024-11-06). Key CCO terms used:

| CCO ID | Label | Used For |
|---|---|---|
| `ont00001017` | Agent | prov:Agent equivalence |
| `ont00001262` | Person | prov:Person equivalence |
| `ont00001180` | Organization | prov:Organization equivalence |
| `ont00000958` | Information Content Entity | prov:Bundle, Plan, Dictionary, KeyEntityPair |
| `ont00000197` | Process Beginning | prov:Start equivalence |
| `ont00000083` | Process Ending | prov:End equivalence |
| `ont00001833` | has agent | prov:wasAssociatedWith, prov:agent |
| `ont00001787` | agent in | prov:qualifiedDelegation |
| `ont00001834` | affects | prov:invalidated, prov:used |
| `ont00001886` | is affected by | prov:wasInvalidatedBy |
| `ont00001816` | is output of | prov:wasGeneratedBy |
| `ont00001986` | has output | prov:generated |
| `ont00001921` | has input | SWRL EntityInfluence |
| `ont00001920` | prescribed by | property chain for prov:hadPlan |
| `ont00001777` | has process part | prov:qualifiedAssociation |
| `ont00001808` | is about | prov:pairEntity, prov:pairKey |
| `ont00001801` | is subject of | prov:has_anchor, has_provenance, asInBundle |

## Validation Infrastructure

### CI Pipeline (GitHub Actions)

`.github/workflows/test-mappings.yml` triggers on PRs to `main`:
1. **Consistency check**: `make -C src reason-edit` -- runs HermiT reasoner
2. **SPARQL verification**: `make -C src test-edit` -- runs all `.rq` queries

### Makefile Targets

| Target | Description |
|---|---|
| `reason-edit` | HermiT consistency check on editor's file |
| `test-edit` | ROBOT verify with all SPARQL queries |
| `unmapped` | Remove individuals, reason, then check for unmapped terms |
| `entailed-mappings` | Materialize all entailed mappings via HermiT |
| `deductive-diff` | Compare old vs new PROV entailments |
| `SSSOM` | Extract SSSOM-format CSV of direct mappings |
| `candidates` | Find candidate superproperties by domain/range matching |
| `count-prov-terms` | Count all PROV classes and object properties |
| `extract-imports` | ROBOT extract subset of RO needed for mappings |

### SPARQL Totality Query (`unmapped-terms.rq`)

The key validation query checks that **every** PROV class and object property has at least one mapping. It looks for terms that lack ALL of:
- Direct `rdfs:subClassOf` / `rdfs:subPropertyOf` / `owl:equivalentClass` / `owl:equivalentProperty`
- Transitive subsumption chains reaching a BFO/RO/CCO term
- Property chain axioms referencing BFO/RO/CCO terms
- SWRL rules mentioning the term
- Any of the above for the term's `owl:inverseOf` counterpart

### RO Terms Extracted

Only 7 RO properties are needed (extracted via `get-imported-terms.rq`):
- `RO_0000056` -- participates in
- `RO_0000057` -- has participant
- `RO_0000087` -- has role
- `RO_0002351` -- has member
- `RO_0002410` -- causally related to
- `RO_0002501` -- causal relation between processes
- `RO_0002559` -- causally influenced by

### ROBOT Pipeline

```
ROBOT v1.9.5 + HermiT v1.4.5.456
```

Key ROBOT commands used:
- `robot reason --reasoner HermiT` -- consistency checking
- `robot verify --queries *.rq` -- SPARQL-based test assertions
- `robot query` -- ad hoc SPARQL extraction
- `robot remove --select individuals` -- strip test instances before reasoning
- `robot extract --method subset --term-file` -- extract RO dependency subset
- `robot diff` -- deductive diff between old and new entailments
- `robot relax` -- materialize subClassOf from equivalentClass for SSSOM export
- `robot annotate` -- add version/provenance metadata

### Test Strategy

All canonical PROV-O examples (11 from core, plus PROV-AQ, Dictionary, Links, Dublin Core) are loaded as named individuals in the editor's file. HermiT checks that these instances remain consistent under the mapping axioms. If any mapping introduces a contradiction with the W3C examples, the reasoner flags it.

### SSSOM Integration

The `SSSOM-mappings.rq` query extracts flat subject-predicate-object triples in SSSOM CSV format with `mapping_justification = ManualMappingCuration`. Complex mappings (class expressions, SWRL) are noted as out of scope pending SSSOM specification evolution (see https://github.com/mapping-commons/SSSOM/issues/36).

## Design Principles

1. **Totality**: Every PROV-O class and object property must map somewhere.
2. **Consistency**: All W3C example instances must remain consistent under mappings.
3. **Modularity**: Three separate mapping files for BFO/RO/CCO, import only what you need.
4. **Justification**: Every mapping carries an `rdfs:comment` explaining the ontological reasoning.
5. **Inverse entailment**: Mapping one direction + `owl:inverseOf` lets the reasoner derive the other.
6. **SWRL for polymorphism**: When the same PROV property maps to different BFO relations depending on the type of its subject, SWRL rules disambiguate.
7. **SKOS for conflicts**: When BFO/CCO semantics genuinely conflict with PROV design choices (e.g., SoftwareAgent as material vs. information), `skos:relatedMatch` signals the issue without asserting false subsumption.
