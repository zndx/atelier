<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# BFO (Basic Formal Ontology) Research

Research notes for applying BFO principles to a data sensitivity/governance classification system.

## 1. What BFO Is

BFO is a **top-level (upper) ontology** -- a small, domain-neutral framework of ~36 classes that provides the most general categories of reality. It contains zero domain-specific terms. Its purpose:

- Serve as a **common root** upon which domain ontologies are built
- Enable **interoperability** between independently developed ontologies (if ontology A and ontology B both extend BFO, they share structural compatibility)
- Standardized as **ISO/IEC 21838-2:2021** (the only top-level ontology with ISO recognition)
- Created by Barry Smith (University at Buffalo), adopted by **650+ ontology projects**

Key distinction: BFO is NOT a vocabulary or taxonomy. It is a *structural framework* -- it tells you what kinds of things CAN exist (objects, processes, qualities, roles) but says nothing about any specific domain.

## 2. Core Structure: The BFO Hierarchy

```
Entity
├── Continuant                              (persists through time, retains identity)
│   ├── Independent Continuant              (can exist on its own)
│   │   ├── Material Entity
│   │   │   ├── Object                      (e.g., a person, a server, a document)
│   │   │   ├── Object Aggregate            (e.g., a fleet of servers, a dataset collection)
│   │   │   └── Fiat Object Part            (e.g., the upper half of a hard drive)
│   │   └── Immaterial Entity
│   │       ├── Site                         (e.g., a data center, a network zone)
│   │       ├── Spatial Region
│   │       └── Continuant Fiat Boundary
│   ├── Specifically Dependent Continuant   (depends on ONE specific entity)
│   │   ├── Quality                          (e.g., the sensitivity level OF this dataset)
│   │   ├── Realizable Entity
│   │   │   ├── Role                         (e.g., "data steward" role of a person)
│   │   │   └── Disposition                  (e.g., a system's vulnerability to breach)
│   │   └── Function                         (e.g., the encryption function of a module)
│   └── Generically Dependent Continuant    (depends on SOME carrier, but can be copied)
│       └── Information Content Entity (ICE) (e.g., a policy document, a classification label)
│
└── Occurrent                               (happens in time, has temporal parts)
    ├── Process                              (e.g., a data classification review, an audit)
    ├── Process Boundary                     (e.g., the moment a classification decision is made)
    ├── Temporal Region
    │   ├── Zero-Dimensional Temporal Region (an instant)
    │   └── One-Dimensional Temporal Region  (a time interval)
    └── Spatiotemporal Region
```

### The Key Philosophical Split: Continuant vs. Occurrent

| Continuant | Occurrent |
|---|---|
| EXISTS at a time, persists | HAPPENS over time, unfolds |
| Has no temporal parts | Has temporal parts (beginning, middle, end) |
| "What IS" | "What HAPPENS" |
| A dataset, a policy, a person | A classification review, an access event, an audit |

### Critical BFO Category for Data Governance: Generically Dependent Continuant (GDC)

A GDC is an entity that:
- Depends on SOME carrier to exist (a physical document, a database row, a file on disk)
- Can be **copied** between carriers without losing identity
- The pattern/content is what matters, not the specific physical medium

The **Information Content Entity (ICE)** -- defined by the Information Artifact Ontology (IAO), a BFO extension -- is the primary subclass of GDC. This is where ALL information classification lives:
- A classification label ("CONFIDENTIAL") is an ICE
- A data governance policy is an ICE
- A sensitivity tag on a column is an ICE
- A regulatory category identifier is an ICE

## 3. How Domain Ontologies Extend BFO

The layering architecture is:

```
┌─────────────────────────────────────┐
│  Domain Ontology (YOUR vocabulary)  │  e.g., DataSensitivityOntology
│  - PII, PHI, financial_data, etc.  │
├─────────────────────────────────────┤
│  Mid-Level Ontology (bridge layer)  │  e.g., CCO, IAO
│  - InformationContentEntity         │
│  - Agent, Role, Artifact            │
├─────────────────────────────────────┤
│  BFO (top-level, ~36 classes)       │  ISO/IEC 21838-2
│  - Continuant, Occurrent            │
│  - Quality, Role, Disposition       │
└─────────────────────────────────────┘
```

**Grounding a domain term in BFO** means every term in your vocabulary has a path back to a BFO category via `is_a` (subclass) relationships:

```
SensitivityClassification
  is_a  InformationContentEntity      (from IAO)
    is_a  GenericallyDependentContinuant  (from BFO)
      is_a  Continuant                    (from BFO)
        is_a  Entity                      (from BFO)
```

**Practical steps to "BFO-ground" a domain ontology:**

1. **Top-down**: For each BFO leaf category, ask "do I have domain terms that fit here?"
2. **Bottom-up**: For each domain term, ask "what kind of thing IS this in BFO terms?"
3. Every domain class must be a subclass of exactly one BFO category
4. Relations between classes must be drawn from a standard relation ontology (RO)
5. Every term gets a textual definition following the "genus-differentia" pattern:
   `An X is a [BFO parent] that [differentiating characteristics]`

Example for data governance:
- "Sensitivity Label" is_a InformationContentEntity that *is about* the sensitivity level of a data asset
- "Classification Review" is_a Process that *has_participant* some DataSteward and *has_input* some DataAsset
- "Data Steward Role" is_a Role that *inheres_in* some Person and is *realized by* some ClassificationReview

## 4. Relationship to OBO Foundry and Beyond

### OBO Foundry (Open Biological and Biomedical Ontologies)
- A community of 250+ interoperable ontologies, ALL grounded in BFO
- Enforces strict principles: open license, unique ID space, textual definitions, single inheritance
- The *success story* that proved BFO works at scale -- but BFO itself is domain-neutral

### Key BFO-based Mid-Level Ontologies
| Ontology | Scope | Relevance to Data Gov |
|---|---|---|
| **IAO** (Information Artifact Ontology) | Information content entities, documents, data | HIGH -- defines ICE, the parent class for all information |
| **CCO** (Common Core Ontologies, 11 modules) | Agents, events, information, artifacts, time | HIGH -- DoD/IC baseline standard |
| **IOF** (Industrial Ontologies Foundry) | Manufacturing, supply chain, lifecycle | MEDIUM -- shows non-biomedical BFO adoption |
| **IAO-Intel** | Intelligence community information artifacts | HIGH -- classifies documents, images, emails for military intelligence |

### Non-Biomedical BFO Adoption
- **U.S. DoD + Intelligence Community**: BFO and CCO adopted as "baseline standards" (2024)
- **NIST / Industrial Ontologies Foundry**: Manufacturing, supply chain, maintenance
- **Information Security**: Cybersecurity classification ontology (BFO-based)
- **Financial**: Commercial exchange ontologies (CCO-based)

## 5. BFO for Data Governance / Information Classification

BFO is absolutely applicable to data governance. The relevant path:

**Your classification labels, sensitivity categories, and governance metadata are all Information Content Entities (ICEs)** -- they are generically dependent continuants that can be copied across systems and are *about* something (the data asset they classify).

### How a governance ontology maps to BFO

| Governance Concept | BFO Category | Rationale |
|---|---|---|
| Data asset (a database, a file) | Material Entity > Object | It's a physical/digital thing that persists |
| Sensitivity label ("CONFIDENTIAL") | GDC > InformationContentEntity | It's copyable information *about* something |
| Classification policy | GDC > InformationContentEntity | It's a document/rule set |
| Data steward (person) | Independent Continuant > Object | A person |
| Data steward (role) | Specifically Dependent Continuant > Role | The role inheres in the person |
| Classification review | Occurrent > Process | It happens over time |
| Regulatory requirement | GDC > InformationContentEntity | A normative specification |
| Access control rule | GDC > InformationContentEntity | Prescriptive information |
| "Column contains PII" assertion | GDC > InformationContentEntity | An assertion *about* a data element |

### IAO-Intel as a Precedent

IAO-Intel (2013, Barry Smith et al.) is the closest existing analogue:
- Extends IAO for the intelligence community
- Provides controlled vocabulary for classifying documents, images, emails
- Designed to unify multiple existing military dictionaries and metadata registries
- Shows BFO can anchor classification systems far from biomedicine

## 6. What "BFO-Grounded" Means in Practice

For a controlled vocabulary to be "BFO-grounded" means:

1. **Every term has a parent path to BFO**: No floating terms. Each concept traces back through `is_a` to a BFO class.

2. **Genus-differentia definitions**: Every term is defined as "An X is a [parent class] that [distinguishing features]." This prevents ambiguity.

3. **Single inheritance**: Each class has exactly one `is_a` parent (no multiple inheritance in the asserted hierarchy). Multiple classification is handled via roles and qualities.

4. **Standard relations**: Relationships between terms use a controlled set (RO -- Relation Ontology), not ad-hoc predicates. Examples: `is_about`, `has_participant`, `inheres_in`, `realized_by`, `part_of`.

5. **Unique, persistent identifiers**: Each term gets a CURIE (see below) that never changes, even if the label changes.

6. **Open-world assumption compatible**: The ontology describes what IS, not what ISN'T. Absence of a classification doesn't mean "unclassified."

**What you gain**: interoperability with ANY other BFO-grounded ontology, logical consistency checking (reasoners can detect contradictions), and a shared conceptual framework that survives organizational changes.

**What it costs**: initial intellectual overhead of mapping your terms to BFO categories, and discipline in maintaining genus-differentia definitions.

## 7. CURIE Codes (Compact URIs)

### Format

```
PREFIX:LOCAL_ID
```

- **PREFIX**: A short, registered namespace abbreviation (e.g., `OBI`, `IAO`, `GO`, `BFO`)
- **LOCAL_ID**: A unique identifier within that namespace (typically a zero-padded integer for OBO ontologies)

### Examples

| CURIE | Expands To | Meaning |
|---|---|---|
| `BFO:0000001` | `http://purl.obolibrary.org/obo/BFO_0000001` | Entity (BFO root) |
| `BFO:0000002` | `http://purl.obolibrary.org/obo/BFO_0000002` | Continuant |
| `BFO:0000003` | `http://purl.obolibrary.org/obo/BFO_0000003` | Occurrent |
| `IAO:0000030` | `http://purl.obolibrary.org/obo/IAO_0000030` | Information Content Entity |
| `OBI:0000011` | `http://purl.obolibrary.org/obo/OBI_0000011` | Planned Process |

### Key Properties

- **Opaque IDs**: The number is meaningless -- it's just a unique key. The human-readable label is separate metadata. This means labels can change without breaking references.
- **Prefix registries**: The Bioregistry (bioregistry.io) and OBO Foundry maintain canonical prefix-to-URI mappings.
- **Pattern**: For OBO ontologies, CURIEs match `^PREFIX:\d{7}$` and expand via: `http://purl.obolibrary.org/obo/PREFIX_LOCALID`
- **Why not human-readable IDs?**: Labels change ("Personally Identifiable Information" might become "Personal Data" under GDPR). Opaque IDs are stable forever.

### For a Custom Ontology

If you're building a data governance ontology, you would:
1. Register a prefix (e.g., `DGOV`) with a prefix registry or use your own namespace
2. Mint CURIEs like `DGOV:0000001` for "Sensitivity Classification"
3. Map them to URIs like `https://your-org.example/ontology/DGOV_0000001`
4. The label "Sensitivity Classification" is metadata ON the term, not the identifier itself

## Practical Implications for a Data Classification System

### What to adopt from BFO

1. **The ICE pattern**: All classification labels, policies, and metadata assertions are Information Content Entities. This gives you a principled answer to "what IS a classification label?"

2. **Role vs. Type distinction**: A "Data Steward" is a ROLE (can be gained/lost), not a TYPE of person. A "Sensitive Dataset" is a dataset that BEARS a sensitivity quality -- the sensitivity is not intrinsic to the data type.

3. **Process modeling**: Classification reviews, audits, access decisions are Processes with participants, inputs, and outputs. This lets you track provenance.

4. **Opaque identifiers (CURIEs)**: Use numeric IDs for terms. Labels are metadata. This future-proofs your vocabulary against renaming.

### What you probably do NOT need

1. **Full OWL formalization**: Unless you're running a DL reasoner, you don't need the full logical apparatus. A well-structured taxonomy with genus-differentia definitions captures 90% of the value.

2. **The entire BFO hierarchy**: You'll likely only use 5-8 BFO categories: InformationContentEntity, Quality, Role, Process, Object, ObjectAggregate, and maybe Disposition.

3. **OBO Foundry membership**: That's for open, community-maintained ontologies. An internal governance vocabulary can follow BFO principles without joining the Foundry.

## Sources

- [BFO Home](https://basic-formal-ontology.org/)
- [BFO on OBO Foundry](https://obofoundry.org/ontology/bfo.html)
- [BFO GitHub](https://github.com/BFO-ontology/BFO)
- [ISO/IEC 21838-2:2021](https://www.iso.org/standard/74572.html)
- [Stephen Diehl - BFO Overview](https://www.stephendiehl.com/posts/bfo/)
- [Common Core Ontologies](https://github.com/CommonCoreOntology/CommonCoreOntologies)
- [IAO GitHub](https://github.com/information-artifact-ontology/IAO)
- [IAO-Intel Paper (PDF)](http://ontology.buffalo.edu/smith/articles/STIDS-2013.pdf)
- [NCOR Wiki - BFO-Based Data Ontologies](https://ncorwiki.buffalo.edu/index.php/BFO-Based_Data_and_Information_Ontologies)
- [IOF Core Ontology (NIST)](https://www.nist.gov/publications/industrial-ontologies-foundry-iof-core-ontology)
- [BFO + CCO as DoD/IC Baseline Standards](https://www.buffalo.edu/cas/philosophy/news/latestnews/smith-top-level-ontologies.html)
- [OAK - CURIEs and URIs](https://incatools.github.io/ontology-access-kit/guide/curies-and-uris.html)
- [What's a CURIE (Biopragmatics)](https://cthoyt.com/2021/09/14/curies.html)
- [Building Ontologies with BFO (MIT Press)](https://mitpress.mit.edu/9780262527811/building-ontologies-with-basic-formal-ontology/)
