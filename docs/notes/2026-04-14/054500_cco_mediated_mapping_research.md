<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# CCO-Mediated BFO Mapping Research — Phase 1 Study

Date: 2026-04-14

## Objective

Study the PROV-to-BFO alignment patterns, the existing signals GitTables taxonomy,
and CCO module structure to extract a reusable mapping template for Schema.org/DBpedia
→ CCO → BFO alignment.

## 1. PROV-to-BFO Mapping Patterns (7 Recurring Idioms)

Source: [BFO-Mappings/PROV-to-BFO](https://github.com/BFO-Mappings/PROV-to-BFO)

### Idiom 1: Reified SSSOM Annotations

Every mapping wrapped in `owl:Axiom` with `sssom:object_label` + `rdfs:comment` justification:

```turtle
[] rdf:type owl:Axiom ;
   owl:annotatedSource   prov:Activity ;
   owl:annotatedProperty owl:equivalentClass ;
   owl:annotatedTarget   obo:BFO_0000015 ;
   sssom:object_label    "process" ;
   rdfs:comment "A prov:Activity is equivalent to process because..."@en .
```

### Idiom 2: Direct Equivalence (`owl:equivalentClass`)

Used when concepts are coextensive. Examples:
- `prov:Activity` ≡ `BFO:process`
- `prov:InstantaneousEvent` ≡ `BFO:process_boundary`
- `prov:Location` ≡ `BFO:site`

### Idiom 3: Subsumption (`rdfs:subClassOf`)

Used when PROV is narrower than BFO:
- `prov:Role` ⊑ `BFO:role`
- `prov:Bundle` ⊑ `BFO:generically_dependent_continuant`

### Idiom 4: Complex Class Expressions

`prov:Entity` maps to a union excluding spatial regions:
```
(IndependentContinuant ∩ ¬SpatialRegion) ∪ GDC ∪ SDC
```

`prov:Agent` uses intersection with existential restrictions:
```
MaterialEntity ∩ (participates_in some Activity) ∩ (bears some (Role ∩ (realized_in some Activity)))
```

### Idiom 5: Property Chain Axioms

`prov:hadPlan` via CCO `prescribed_by`:
```turtle
prov:hadPlan owl:propertyChainAxiom
  ( [ owl:inverseOf prov:qualifiedAssociation ] cco:prescribed_by ) ;
```

### Idiom 6: SWRL Rules for Polymorphic Properties

When a property's meaning depends on the subject's type:
- Activity + atLocation → `occurs_in`
- Entity + atLocation → `located_in`

### Idiom 7: SKOS for Genuine Conflicts

When PROV design conflicts with BFO realism:
```turtle
prov:ServiceDescription skos:relatedMatch cco:InformationContentEntity .
```

### Three-Module Architecture

Independently importable:
1. `prov-bfo-directmappings.ttl` — core BFO (standalone)
2. `prov-ro-directmappings.ttl` — RO (imports prov-bfo)
3. `prov-cco-directmappings.ttl` — CCO (imports prov-bfo)

### Validation Pipeline

- HermiT consistency check on merged ontology + W3C test instances
- SPARQL totality query: every PROV class/property has at least one mapping
- ROBOT verify, reason, diff, relax commands
- CI via GitHub Actions

### RO Dependency (Minimal)

Only 7 RO properties: participates_in, has_participant, has_role, has_member,
causally_related_to, causal_relation_between_processes, causally_influenced_by.

---

## 2. Signals GitTables Taxonomy (122 DBpedia Types)

Source: `/home/rch/local/src/cldr/signals/config/sigint/gittables_taxonomy.py`

### Structure

5 levels of depth, 122 leaves + 36 internal nodes = 158 total.

```
L0: entity (BFO:Entity)
L1: continuant (BFO:Continuant), occurrent (BFO:Occurrent)
L2: continuant.gdc (BFO:GDC), continuant.quality (BFO:Quality),
    continuant.ic (BFO:IndependentContinuant),
    occurrent.process (BFO:Process), occurrent.temporal (BFO:TemporalRegion)
L3: 13 domain-specific nodes (IAO-inspired, not formally CCO)
L4: 8 further specializations
```

### Design Principles

1. Maps by "what the property's range value IS in BFO terms"
2. Uses IAO-inspired intermediates (MeasurementDatum, PlanSpecification)
3. No published DBpedia-to-BFO mapping existed — constructed manually
4. 47 documented confusable pairs for DST focal elements

### CCO Alignment Opportunities

Informal alignment is already present but not formalized:

| Signals Internal Node | Candidate CCO Class |
|-----------------------|--------------------|
| GDC.Identifier | CCO:DesignativeICE |
| GDC.DescriptiveContent | CCO:DescriptiveICE |
| GDC.PlanSpecification | CCO:PrescriptiveICE (via IAO:0000104) |
| GDC.MeasurementDatum | CCO:DescriptiveICE / IAO:MeasurementDatum |
| IC.Agent | CCO:Agent (Person, Organization) |
| IC.Artifact | CCO:Artifact |
| Process.Activity | CCO:Act / BFO:Process |
| Temporal.DatePoint | CCO:TemporalInstant |
| Temporal.Duration | CCO:TemporalInterval |
| Site.* | BFO:Site (CCO:GeospatialRegion for some) |

---

## 3. CCO Module Inventory

Source: [CommonCoreOntology/CommonCoreOntologies](https://github.com/CommonCoreOntology/CommonCoreOntologies)

11 modules, ~1,539 classes total. BSD-3. IEEE P3195.1 standardization in progress.
BFO + CCO are DoD/IC baseline since Jan 2024.

### High Relevance (import first)

| Module | Key Classes | Why |
|--------|------------|-----|
| **InformationEntityOntology** | ICE → DesignativeICE, DescriptiveICE, PrescriptiveICE | Core classification axis |
| **AgentOntology** | Person, Organization (Commercial/Gov/Edu), Roles | Agent classification |
| **ExtendedRelationOntology** | is_input_of, is_output_of, affects, has_process_part | Relationship typing |

### Medium Relevance (add for lifecycle)

| Module | Key Classes |
|--------|------------|
| **EventOntology** | Act, Planned Act, Act of Measuring, Change |
| **TimeOntology** | Unix Temporal Instant, Calendar Day/Month/Year |

### Low Relevance (skip initially)

QualityOntology, ArtifactOntology, UnitsOfMeasureOntology, GeospatialOntology,
FacilityOntology, CurrencyUnitOntology.

### The ICE Trichotomy

```
BFO:GenericallyDependentContinuant
  CCO:InformationContentEntity
    CCO:DesignativeICE      -- identifiers, names, codes
    CCO:DescriptiveICE      -- descriptions, measurements, predictions
    CCO:PrescriptiveICE     -- specifications, algorithms, plans
```

This maps directly onto the signals taxonomy's GDC subtree.

---

## 4. Gap Analysis: No Schema.org → BFO Mapping Exists

- **PROV-O → BFO/CCO**: Complete (Prudhomme et al. 2025)
- **DCAT → BFO**: Indirect via PROV-O
- **SSN/SOSA → BFO**: In progress (BFO-Mappings/SSN-to-BFO)
- **NFDIcore 2.0**: Bridges Schema.org/DCAT + BFO via IAO (closest precedent)
- **gistBFO**: Enterprise ontology (98 classes) aligned to BFO
- **Schema.org → BFO/CCO**: **No formal alignment published**
- **DBpedia → BFO**: **No formal alignment published**

### Implication

A principled Schema.org/DBpedia → CCO → BFO mapping would be a novel contribution.
NFDIcore 2.0 and the Prudhomme methodology provide the templates.

---

## 5. Mapping Template (Extracted from PROV-to-BFO)

### Per-term checklist

1. **Identify BFO category**: Use BFO Classifier decision diagram
2. **Choose mapping strength**:
   - Equivalence (owl:equivalentClass) when coextensive
   - Subsumption (rdfs:subClassOf) when source is narrower
   - SKOS relatedMatch when genuine conflict
3. **CCO mediation**: If direct BFO mapping loses nuance, route through CCO mid-level
4. **Annotate**: SSSOM object_label + rdfs:comment with philosophical justification
5. **Validate**:
   - HermiT consistency on representative instances
   - SPARQL totality (no orphan terms)
   - Conservativity (no new entailments in source ontology)

### File structure (following PROV-to-BFO pattern)

```
mappings/
├── schema-bfo-directmappings.ttl    # Core BFO alignments (standalone)
├── schema-cco-directmappings.ttl    # CCO specializations (imports schema-bfo)
├── dbpedia-bfo-directmappings.ttl   # DBpedia → BFO (standalone)
├── dbpedia-cco-directmappings.ttl   # DBpedia → CCO (imports dbpedia-bfo)
├── sparql/
│   └── unmapped-terms.rq            # Totality check
└── tests/
    └── instances.ttl                # Representative classification instances
```

---

## 6. Priority Subset for Prototype (Phase 2)

### From Schema.org (20 types, high value for metadata classification)

**Agents**: schema:Person, schema:Organization
**Creative works**: schema:CreativeWork, schema:Dataset, schema:SoftwareSourceCode
**Properties**: schema:name, schema:identifier, schema:description, schema:dateCreated,
  schema:dateModified, schema:url, schema:email, schema:telephone, schema:address
**Places**: schema:Place, schema:PostalAddress, schema:Country
**Events**: schema:Event
**Quantities**: schema:QuantitativeValue, schema:MonetaryAmount

### From DBpedia (15 types, covering signals taxonomy gaps)

**Agents**: dbo:Person, dbo:Organisation, dbo:Company
**Works**: dbo:Work, dbo:Software, dbo:WrittenWork
**Places**: dbo:Place, dbo:Country, dbo:City, dbo:PopulatedPlace
**Events**: dbo:Event, dbo:SportsEvent
**Temporal**: dbo:TimePeriod, dbo:Year
**Measures**: dbo:Currency

---

## Next Steps

1. Clone PROV-to-BFO repo, import into Protege alongside BFO 2024 + CCO modules
2. Draft `atelier-vocab.ttl` with the 35 priority types above
3. Validate conservativity against existing 16 PII leaves
4. Test inference utility via SPARQL on GitTables classification instances
