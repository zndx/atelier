# Atelier Vocabulary — CCO-Mediated BFO Alignment

Formal ontological grounding for Atelier's metadata classification vocabulary,
mapping Schema.org types, DBpedia types, and the 16 PII leaf categories to
[BFO 2020](http://purl.obolibrary.org/obo/bfo.owl) via the
[Common Core Ontologies](https://github.com/CommonCoreOntology/CommonCoreOntologies) (CCO v2.1).

## Methodology

Follows the mapping methodology from:

> Prudhomme et al. (2025). "A semantic approach to mapping the Provenance
> Ontology to Basic Formal Ontology." *Scientific Data* 12, 282.
> [doi:10.1038/s41597-025-04580-1](https://doi.org/10.1038/s41597-025-04580-1)

Four formal criteria govern every mapping axiom:

| Criterion | Description |
|-----------|-------------|
| **Equivalence** | `owl:equivalentClass` when concepts are coextensive |
| **Subsumption** | `rdfs:subClassOf` when the source is narrower |
| **Conservativity** | No new entailments in the source ontology |
| **Completeness** | Every mapped term has at least one alignment |

Every axiom carries SSSOM-style provenance (`sssom:object_label`) and a
philosophical justification (`rdfs:comment`).

## Coverage

| Source | Classes | Properties | Total |
|--------|---------|------------|-------|
| Schema.org | 11 | 9 | 20 |
| DBpedia | 15 | — | 15 |
| Atelier vocabulary | 300 leaves + 25 internal | — | 325 |
| **Total (unique)** | | | **360** |

## CCO Modules Imported

Only 3 of CCO's 11 modules are required:

1. **InformationEntityOntology** — ICE trichotomy (Designative/Descriptive/Prescriptive)
2. **AgentOntology** — Person, Organization, Commercial/Government Organization
3. **ExtendedRelationOntology** — is_input_of, affects, has_process_part

## Key Mapping: The ICE Trichotomy

The core classification axis maps directly onto CCO's Information Content Entity hierarchy:

```
BFO:GenericallyDependentContinuant
  cco:InformationContentEntity  ≡  atelier:ICE
    cco:DesignativeICE          ←  names, identifiers, codes, addresses, URLs
    cco:DescriptiveICE          ←  descriptions, measurements, timestamps, status
    cco:PrescriptiveICE         ←  software, specifications, plans
```

### Atelier PII leaves by CCO grounding

| CCO Class | Atelier Leaves |
|-----------|---------------|
| DesignativeICE | EMAIL, PHONE, ADDRESS, FULLNAME, GOVID, PAN, BAN, IPADDR, DEVID, URL, RECID |
| DescriptiveICE | DOB, TXNAMT, TIMESTAMP, STATUS |
| (parent ICE only) | NONSENSITIVE |

## Usage Examples

### Querying BFO ancestry for a PII leaf

```sparql
PREFIX atelier: <https://atelier.zndx.org/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?ancestor WHERE {
  atelier:ICE.SENSITIVE.PID.CONTACT.EMAIL rdfs:subClassOf+ ?ancestor .
}
# Returns: CONTACT, PID, SENSITIVE, ICE, cco:DesignativeICE, cco:ICE, BFO:GDC
```

### Finding all designative PII

```sparql
PREFIX cco: <https://www.commoncoreontologies.org/>
PREFIX atelier: <https://atelier.zndx.org/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?leaf WHERE {
  ?leaf rdfs:subClassOf+ cco:ont00000686 .  # DesignativeICE
  ?leaf rdfs:subClassOf+ atelier:ICE.SENSITIVE .
  FILTER NOT EXISTS { ?child rdfs:subClassOf ?leaf }
}
# Returns: EMAIL, PHONE, ADDRESS, FULLNAME, GOVID, PAN, BAN, IPADDR, DEVID, URL
```

### Cross-ontology inference

```sparql
PREFIX schema: <https://schema.org/>
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX cco: <https://www.commoncoreontologies.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

# Find all classes grounded in cco:Person
SELECT ?cls WHERE {
  { ?cls rdfs:subClassOf cco:ont00001262 }
  UNION
  { ?cls owl:equivalentClass cco:ont00001262 }
}
# Returns: schema:Person, dbo:Person
```

## Validation

### SPARQL totality check

```bash
robot verify --input atelier-vocab.ttl \
  --queries sparql/unmapped-terms.rq
```

Expected: zero unmapped terms.

### HermiT consistency check

```bash
robot reason --reasoner HermiT \
  --input atelier-vocab.ttl \
  --output atelier-vocab-reasoned.ttl
```

### Prerequisites

- [ROBOT](http://robot.obolibrary.org/) v1.9.5+
- Java 11+ (for HermiT reasoner)
- BFO 2020 OWL (`http://purl.obolibrary.org/obo/bfo.owl`)
- CCO v2.1 modules from [CommonCoreOntologies](https://github.com/CommonCoreOntology/CommonCoreOntologies)

## File Structure

```
ontology/
├── atelier-vocab.ttl           # Main mapping file (standalone)
├── sparql/
│   └── unmapped-terms.rq       # Totality validation query
└── README.md                   # This file
```

## Integration with Atelier

The mapping file is a reference artifact. The runtime classification pipeline
uses `universal_vocabulary.json` (JSON) for performance. The TTL provides:

1. **Formal grounding** — machine-verifiable BFO/CCO alignment
2. **Cross-ontology bridge** — connects Atelier's PII vocabulary to Schema.org
   and DBpedia type systems via shared CCO superclasses
3. **Extension template** — new domain vocabularies attach via CCO mid-level
   classes, preserving conservativity

## Integration with Signals

The signals project's GitTables taxonomy (122 DBpedia types) aligns informally
with the same CCO classes:

| Signals Internal Node | CCO Grounding |
|-----------------------|---------------|
| GDC.Identifier | DesignativeICE |
| GDC.DescriptiveContent | DescriptiveICE |
| GDC.MeasurementDatum | DescriptiveICE |
| GDC.PlanSpecification | PrescriptiveICE |
| IC.Agent | Agent |
| Process.Activity | BFO:Process |
| Temporal.* | BFO:TemporalRegion |

Retrofitting the signals taxonomy to use explicit CCO axioms requires only
adding `rdfs:subClassOf` declarations — no structural changes.

## Integration with Gaius RASE

The Gaius project's BFO references (Kudu-backed bases, RASE evidence model)
are compatible:

- `bfo:MaterialEntity` → CCO:Agent (for material entities bearing agent capability)
- `bfo:TemporalRegion` → CCO TimeOntology (when lifecycle modeling is added)
- `bfo:Quality` → CCO QualityOntology (for measured qualities)

The same `atelier-vocab.ttl` can be imported into the RASE evidence model
without altering existing SQL-type mappings.

## References

- [BFO 2020](http://purl.obolibrary.org/obo/bfo.owl) — ISO/IEC 21838-2
- [CCO v2.1](https://github.com/CommonCoreOntology/CommonCoreOntologies) — BSD-3
- [PROV-to-BFO](https://github.com/BFO-Mappings/PROV-to-BFO) — Methodology source
- [NFDIcore 2.0](https://github.com/ISE-FIZKarlsruhe/nfdicore) — Schema.org/BFO via IAO
- [SSSOM](https://w3id.org/sssom/) — Simple Standard for Sharing Ontological Mappings
