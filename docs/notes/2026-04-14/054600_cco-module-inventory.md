<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# CCO Module Inventory & Relevance for Metadata Classification / Data Governance

Date: 2026-04-14

## Background

The Common Core Ontologies (CCO) are a suite of **11 modular mid-level ontologies**
extending from the Basic Formal Ontology (BFO), an ISO/IEC 21838-2 standard top-level
ontology. CCO was adopted in January 2024 as the baseline standard for formal ontology
work in the US Department of Defense and Intelligence Community. Licensed BSD-3, publicly
available since 2017.

- Repository: https://github.com/CommonCoreOntology/CommonCoreOntologies
- IEEE standard: IEEE P3195.1
- Total classes across all modules: ~1,539
- FOIS 2024 paper: https://arxiv.org/abs/2404.17758

---

## Module Inventory

### 1. Information Entity Ontology (`InformationEntityOntology.ttl`)

**Scope:** Representation of generic types of information and relationships between
information and other entities.

**Relevance: HIGH -- Primary module for metadata classification.**

Key class hierarchy:
- **Information Content Entity (ICE)** -- the content/meaning, independent of bearer
  - **Designative ICE** -- symbols that uniquely denote entities
    - Designative Name (cultural/social designators, words/phrases)
    - Non-Name Identifier (per encoding systems or random)
    - Code Identifier (characters per encoding system with derivable metadata)
    - Arbitrary Identifier (no encoding system)
  - **Descriptive ICE** -- propositions that describe entities
    - Measurement Unit (standards for measuring physical quantities)
    - Reference System (standards organizing domain-specific data)
    - Measurement Information Content Entity (multiple types)
    - Predictive ICE (uncertain future events)
    - Measurement types per Stevens' classification:
      - Nominal (classifies by shared characteristics)
      - Ordinal (rank order)
      - Interval (equal units, e.g. Celsius)
      - Ratio (true zero, e.g. Kelvin)
  - **Prescriptive ICE** (formerly "Directive ICE") -- propositions/images that prescribe
    - Algorithm (finite instruction sequences)
    - Performance Specification
    - Artifact Model (prescribes common functions/qualities)
- **Information Bearing Entity** -- physical carriers (monitors, textbooks, databases)
- **Information Quality Entity** -- quality concretizing some ICE

**Why it matters:** InformationContentEntity is the BFO-grounded concept for any
classifiable piece of information -- metadata fields, annotations, labels, descriptions.
The Designative/Descriptive/Prescriptive trichotomy maps directly to metadata
classification concerns (identifiers, descriptions, specifications).

---

### 2. Agent Ontology (`AgentOntology.ttl`)

**Scope:** Representation of agents (persons, organizations) and their roles.

**Relevance: HIGH -- Agents are data stewards, data owners, classifiers.**

Key classes:
- **Person** -- Material Entity bearing Agent Capability
  - Citizen, Permanent Resident, Organization Member
  - Allied/Enemy/Neutral Person (conflict context)
- **Organization**
  - Incorporated Organization, Government Organization
  - Commercial Organization, Educational Organization
  - Service Provider, Armed Force, Geopolitical Organization
- **Roles**
  - Authority Role, Organization Member Role, Occupation Role
  - Citizen Role, Contractor Role, Operator Role
  - Commercial Role, Civilian Role
  - Interpersonal Relationship Role
  - Geopolitical Power Role
- **Skills**
  - Language Skill

**Why it matters:** Data governance requires modeling who creates, owns, stewards,
classifies, and accesses data. Person/Organization/Role triad maps to DCAT's
publisher/creator/contactPoint and Schema.org's author/publisher/contributor.

---

### 3. Event Ontology (`EventOntology.ttl`)

**Scope:** Representation of processual entities (acts performed by agents across domains).

**Relevance: MEDIUM -- Provenance, audit trails, data lifecycle events.**

Key classes:
- **Act** (Process with at least one Agent in causative role)
  - Planned Act / Unplanned Act
  - Act of Communication (representative, directive, expressive, by media)
  - Act of Observation, Act of Measuring, Act of Estimation
  - Act of Information Processing
  - Act of Planning, Act of Prediction
  - Act of Association, Act of Meeting
  - Act of Purchasing, Act of Funding, Act of Remuneration
  - Act of Government
  - Criminal Act (violation of rules/laws)
- **Change** (independent continuant endures, dependent entities increase/decrease)
  - Gain/Loss of Quality, Function, Role, Disposition
- **Natural Process**, **Mechanical Process**
- **Stasis** classes (unchanging states)

**Why it matters:** Data lifecycle events (creation, modification, access, deletion,
classification, review) are all Acts. The PROV-to-BFO mapping (see below) connects
W3C provenance directly into this hierarchy.

---

### 4. Quality Ontology (`QualityOntology.ttl`)

**Scope:** Attributes of entities -- qualities, realizable entities, process profiles.

**Relevance: LOW-MEDIUM -- Primarily physical qualities, limited metadata utility.**

Key classes (~150+ classes):
- Physical: Mass, Weight, Temperature, Hardness, Wetness, Texture
- Color: Color Hue, Saturation, Brightness + 20 specific colors
- Shape: 2D (Round, Square, etc.), 3D (Spherical, Cylindrical, etc.)
- Size: Length, Width, Height, Depth, Diameter, Thickness
- Spatial Orientation: Pitch, Roll, Yaw, Pointing
- Optical Properties: Opacity, Reflectivity, Emissivity, Absorptivity
- Dispositions: Strength, Magnetism, Vulnerability, Radioactive

**Why it matters:** Mostly about physical qualities of material entities. Relevant
if classifying sensor data, IoT metadata, or physical asset descriptions. Less
directly applicable to pure data governance / document metadata classification.

---

### 5. Time Ontology (`TimeOntology.ttl`)

**Scope:** Temporal regions and their relationships.

**Relevance: MEDIUM -- Timestamps, temporal validity, data freshness.**

Key classes:
- Instant-based: Unix Temporal Instant, Reference Time (Epoch), Time of Day,
  Julian Date, Modified Julian Date
- Day: Day, Calendar Day, Gregorian Day, Julian Day,
  Morning, Afternoon, Evening, Night
- Intervals: Second, Minute, Hour, Week, Month, Year, Decade
- Calendar variants: Calendar Week, Calendar Month, Calendar Year,
  Gregorian Year, Julian Year
- Multi-unit: Multi-Second through Multi-Year Temporal Intervals

**Why it matters:** Temporal metadata is fundamental to data governance (creation date,
modification date, retention period, temporal validity windows). CCO Time Ontology
provides BFO-grounded temporal primitives.

---

### 6. Extended Relation Ontology (`ExtendedRelationOntology.ttl`)

**Scope:** Relations that hold between entities at the mid-level.

**Relevance: HIGH -- The relational glue connecting all other modules.**

Key object properties:
- `is_successor_of` / `is_predecessor_of` (Independent Continuant -> IC)
- `has_process_part` (Process -> Process)
- `has_object` (Process -> Quality/Realizable/IC)
- `is_cause_of` (Process -> Process)
- `disrupts` / `inhibits` (Process -> Process)
- `is_input_of` / `is_output_of` (IC -> Process)
- `affects` (Process -> IC)
- `occurs_at` (Process -> Site)
- `process_starts` (Process -> Process)
- Aggregate relations for qualities, dispositions, roles

**Why it matters:** These relations define how information entities relate to agents,
events, and each other. Essential for expressing "who created what, when, where, why"
in a formally grounded way.

---

### 7. Artifact Ontology (`ArtifactOntology.ttl`)

**Scope:** Artifacts, specifications, and functions.

**Relevance: LOW-MEDIUM -- Information Processing Artifacts relevant for data systems.**

Key classes:
- Information Processing Artifact (designed to transform information using algorithms)
- Recording Device
- Material copies (Book, Certificate, Transcript, Title Document)
- Vehicles, weapons, communication instruments (domain-heavy)
- Telecommunication Network components

**Why it matters:** "Information Processing Artifact" is the BFO-grounded class for
computational systems, databases, and data processing pipelines. Relevant when modeling
data infrastructure in a governance context.

---

### 8. Geospatial Ontology (`GeospatialOntology.ttl`)

**Scope:** Sites, spatial regions, entities near Earth's surface.

**Relevance: LOW -- Only if classifying geospatial metadata.**

---

### 9. Facility Ontology (`FacilityOntology.ttl`)

**Scope:** Buildings and campuses designed for specific purposes.

**Relevance: LOW -- Only if governance includes physical facility data.**

---

### 10. Currency Unit Ontology (`CurrencyUnitOntology.ttl`)

**Scope:** Currencies issued by countries.

**Relevance: LOW -- Only if classifying financial/economic metadata.**

---

### 11. Units of Measure Ontology (`UnitsOfMeasureOntology.ttl`)

**Scope:** Standard measurement units.

**Relevance: LOW-MEDIUM -- Relevant for scientific/sensor data classification.**

---

## Relevance Summary Matrix

| Module | File | Relevance | Use Case |
|--------|------|-----------|----------|
| Information Entity | InformationEntityOntology.ttl | **HIGH** | Core metadata classification (ICE hierarchy) |
| Agent | AgentOntology.ttl | **HIGH** | Data stewards, owners, classifiers |
| Extended Relation | ExtendedRelationOntology.ttl | **HIGH** | Relational backbone (input/output/cause/affect) |
| Event | EventOntology.ttl | **MEDIUM** | Data lifecycle events, provenance |
| Time | TimeOntology.ttl | **MEDIUM** | Temporal metadata, retention, validity |
| Quality | QualityOntology.ttl | **LOW-MED** | Physical/sensor metadata only |
| Artifact | ArtifactOntology.ttl | **LOW-MED** | Data infrastructure modeling |
| Units of Measure | UnitsOfMeasureOntology.ttl | **LOW-MED** | Scientific data classification |
| Geospatial | GeospatialOntology.ttl | **LOW** | Geospatial metadata only |
| Facility | FacilityOntology.ttl | **LOW** | Physical facility data only |
| Currency Unit | CurrencyUnitOntology.ttl | **LOW** | Financial metadata only |

---

## Prior Schema.org / DBpedia -> BFO Mapping Work

### Direct Schema.org -> BFO/CCO Mappings

**No formal, published Schema.org-to-BFO/CCO alignment exists.** This is a gap.

The closest work is:

1. **NFDIcore 2.0** (https://arxiv.org/html/2410.01821v1)
   - A BFO-compliant ontology for German research data infrastructure (NFDI)
   - Integrates Schema.org classes while maintaining BFO compliance via IAO mediation
   - Key mappings:
     - `nfdicore:CreativeWork` = `iao:InformationContentEntity AND schema:CreativeWork`
     - `nfdicore:Service` = `iao:InformationContentEntity AND schema:Service`
   - Also maps to DCTERMS, DCAT
   - Strategy: use Schema.org for practical web integration, BFO for formal rigor
   - GitHub: https://github.com/ISE-FIZKarlsruhe/nfdicore

2. **gistBFO** (https://www.semanticarts.com/gistbfo-an-open-source-bfo-compatible-version-of-gist/)
   - Bridge ontology aligning Semantic Arts' "gist" enterprise ontology to BFO
   - 43 logical axioms (35 subclass + 8 subproperty assertions)
   - Key mappings:
     - gist:Specification -> bfo:GenericallyDependentContinuant
     - gist:Organization -> bfo:ObjectAggregate
     - gist:Event -> union of bfo:GDC and bfo:Process
     - gist:Category -> bfo:GDC
   - gist has 98 classes, 63 object properties -- enterprise-focused (Person, Organization, Agreement)
   - Does NOT directly reference Schema.org but covers similar enterprise territory
   - Creative Commons 4.0 license

### Direct DBpedia -> BFO Alignment

**No formal DBpedia-to-BFO alignment exists.** The research literature notes:
- Only ~17 known alignments from DOLCE to BFO (v1.0 only)
- DBpedia has its own ontology for Wikipedia infobox mapping
- The **BFO Classifier** tool (https://ceur-ws.org/Vol-3249/paper3-FOUST.pdf) provides
  a decision-diagram approach to align arbitrary domain entities to BFO, which could
  be applied to DBpedia classes

### PROV-O -> BFO/CCO Mapping (Completed, Published)

**This is the gold standard for web ontology -> BFO mapping.**

- Repository: https://github.com/BFO-Mappings/PROV-to-BFO
- Paper: Prudhomme et al., "A semantic approach to mapping the Provenance Ontology
  to Basic Formal Ontology," Scientific Data 12, 282 (2025)
- Covers 5 PROV specifications (PROV-O, Access/Query, Data Dictionary,
  Linking Across Provenance Bundles, Dublin Core)
- Three mapping files:
  - `prov-bfo-directmappings.ttl` (PROV -> BFO)
  - `prov-ro-directmappings.ttl` (PROV -> RO)
  - `prov-cco-directmappings.ttl` (PROV -> CCO)
- Uses rdfs:subClassOf, owl:equivalentClass, owl:equivalentProperty, SWRL rules

### DCAT -> BFO

**No direct mapping exists.** However:
- DCAT uses PROV-O for provenance (dcat:Dataset -> prov:Entity)
- Since PROV-O -> BFO/CCO mapping is complete, an indirect path exists:
  DCAT -> PROV-O -> BFO/CCO
- NFDIcore 2.0 maps to both DCAT and BFO, providing a practical bridge

### SSN/SOSA -> BFO

- Repository: https://github.com/BFO-Mappings/SSN-to-BFO (in progress)
- Semantic Sensor Network / Sensor, Observation, Sample, Actuator ontology

---

## Key Architectural Insight: The InformationContentEntity Hierarchy

For metadata classification, the most important BFO/CCO concept is:

```
BFO:Entity
  BFO:Continuant
    BFO:GenericallyDependentContinuant
      CCO:InformationContentEntity (ICE)
        CCO:DesignativeICE      -- identifiers, names, codes
        CCO:DescriptiveICE      -- descriptions, measurements, predictions
        CCO:PrescriptiveICE     -- specifications, algorithms, plans
```

Every piece of classifiable metadata is an InformationContentEntity. The three-way
split (Designative/Descriptive/Prescriptive) provides the top-level classification
axis for any metadata governance taxonomy.

---

## Recommendations

1. **Start with 3 modules**: InformationEntityOntology + AgentOntology + ExtendedRelationOntology
2. **Add Event + Time** when modeling data lifecycle / provenance
3. **Leverage PROV-to-BFO mapping** for provenance-aware classification
4. **Watch NFDIcore 2.0** as the most mature Schema.org-to-BFO bridge
5. **A Schema.org -> CCO alignment is a gap worth filling** -- no one has published one yet
