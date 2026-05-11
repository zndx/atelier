# SOTAB v2 Coverage Strategy

> **Ownership note (2026-05-09).**  Going forward, **all ontology /
> vocabulary / synthetic-data work moves to Ægir**.  The label space
> conditions model pre-training directly, so it lives where the model
> lives.  Atelier becomes the consumer of trained artifacts (H-Net/RWKV
> checkpoints + SVMs trained on Ægir-curated datasets).  This document
> stays in Atelier's docs as the specification of *what* we want
> covered; the actual TTL extensions, generators, and SOTAB integration
> implementation belong in `~/local/src/zndx/aegir/`.

This document specifies how the BFO/CCO-grounded vocabulary should
cover the SOTAB v2 Schema.org CTA label space (82 labels), so the
hierarchical RWKV-7 model in Ægir can ladder predictions up from raw
benchmark labels to BFO/CCO concepts.

## Background

- **Atelier (current)**: ICE trichotomy (Designative / Descriptive /
  Prescriptive) grounded through Common Core Ontologies into BFO 2020.
  20 Schema.org subjects mapped today (11 classes, 9 properties) in
  `src/atelier/classify/ontology/atelier-vocab.ttl`.  This snapshot
  remains operational for the existing classification pipeline during
  the migration window.
- **Atelier (future)**: consumer of pre-trained model artifacts.
  Loads H-Net/RWKV checkpoints and SVMs trained in Ægir against
  ontology-grounded label spaces; uses them as evidence sources in DST
  fusion.  No longer owns vocabulary.
- **Ægir (current → future ontology home)**: hierarchical RWKV-7 model
  targeting CTA + CPA on wide tables, trained against
  `gt-signals-dbpedia` and SOTAB v2.  SOTAB infrastructure already
  wired: `scripts/download_sotab.py` fetches the four canonical bundles,
  `src/aegir/data/table_dataset.py` loads `sotab_v2_cta_*_set.csv`
  ground truth.  Inheriting `atelier-vocab.ttl` + synth generators
  is part of the M2 roadmap.
- **Synthetic data pipeline**: currently in atelier
  (`synth_generators.py`, 316+ generators).  Migration target: Ægir,
  since the generators feed pre-training corpora directly.  Atelier's
  classification pipeline can consume generator output via a thin
  client during transition.

## Authoritative SOTAB v2 label space

Verified against `/raid/datasets/sotab/sotab_v2_cta_*_set.csv` (union of
training, validation, test, and the three robustness test sets:
`corner_cases`, `missing_values`, `format_heterogeneity`):

```
82 distinct CTA labels covering 17 root entity types
```

Root entity types (from `webdatacommons.org/structureddata/sotab/v2/`,
Table 2): Book, CreativeWork, Event, Hotel, JobPosting, LocalBusiness,
Movie, Museum, MusicAlbum, MusicRecording, Person, Place, Product,
Recipe, Restaurant, SportsEvent, TVEpisode.

The 82 labels are a mix of:

- **Class names** — `Country`, `MonetaryAmount`, `Organization`, etc.
- **Entity-property pairs** — `Book/name`, `Hotel/description`,
  `JobPosting/description` (the slash separates entity type from the
  property whose value the column carries).
- **Measurement units** — `Distance`, `Duration`, `Energy`, `Mass`.
- **Enumeration types** — `BookFormatType`, `EventStatusType`,
  `GenderType`, `RestrictedDiet`.
- **Coded attribute types** — `CoordinateAT`, `IdentifierAT`,
  `MusicArtistAT` (the `AT` suffix denotes "atomic type", a SOTAB
  convention, not Schema.org).

Aegir's stale `_LABEL_DIMS["sotab"] = 91` comment in
`src/aegir/data/table_dataset.py` should be reduced to `82`; the extra 9
appear to be carry-over from an earlier label set draft.

## Coverage analysis

### Direct hits (14 of 82)

Already grounded in `atelier-vocab.ttl`:

| SOTAB label | Atelier mapping |
|---|---|
| `Country` | `schema:Country ⊑ BFO:Site` |
| `CreativeWork` | `schema:CreativeWork ⊑ cco:ICE` |
| `CreativeWork/name` | `schema:name` (rdf property; we have it) |
| `Event/description`, `Event/name` | `schema:Event` + `schema:description`/`schema:name` |
| `MonetaryAmount` | `schema:MonetaryAmount ⊑ cco:DescriptiveICE` |
| `Organization` | `schema:Organization ⊑ cco:Organization` |
| `Person/name` | `schema:Person ⊑ cco:Person` + `schema:name` |
| `Place/name` | `schema:Place ⊑ BFO:Site` + `schema:name` |
| `PostalAddress` | `schema:PostalAddress ⊑ cco:DesignativeICE` |
| `QuantitativeValue` | `schema:QuantitativeValue ⊑ cco:DescriptiveICE` |
| `email` | `schema:email ⊑ cco:DesignativeICE` |
| `telephone` | `schema:telephone ⊑ cco:DesignativeICE` |
| `URL` | `schema:url` (we use lowercase; SOTAB uses `URL`) |

### Subsumption-reachable (~20 of 82)

Subclasses of types already grounded — adding them is a single
`rdfs:subClassOf` edge under the existing CCO branch:

**Schema:CreativeWork descendants** (8): `Book/description`, `Book/name`,
`BookFormatType`, `Movie/description`, `Movie/name`, `Recipe/description`,
`Recipe/name`, `MusicAlbum/name`, `MusicRecording/name`, `TVEpisode/name`,
`CreativeWorkSeries`, `Photograph`, `Review`.

**Schema:Organization descendants** (5): `Hotel/description`, `Hotel/name`,
`LocalBusiness/name`, `Museum/name`, `Restaurant/name`, `SportsTeam`.

**Schema:Event descendants** (1): `SportsEvent/name`.

**Schema:PostalAddress sub-properties** (4): `addressLocality`,
`addressRegion`, `postalCode`, `streetAddress`.

**Schema:QuantitativeValue measurement subtypes** (4 + 1):
`Distance`, `Duration`, `Energy`, `Mass`, `weight` (column-as-property).

### Missing — requires new vocab work (~48 of 82)

Grouped by extension target:

| Group | SOTAB labels | Target CCO/BFO grounding |
|---|---|---|
| **Product family** (new branch) | `Product/description`, `Product/name`, `ProductModel`, `Brand` | `cco:Artifact` + `cco:ArtifactModel` (Prescriptive territory) |
| **Job posting family** | `JobPosting/description`, `JobPosting/name`, `OccupationalExperienceRequirements`, `EducationalOccupationalCredential`, `workHours`, `paymentAccepted` | `cco:DescriptiveICE` (descriptive content about employment) |
| **Product economics** | `price`, `priceRange`, `currency`, `DeliveryMethod`, `ItemAvailability`, `OfferItemCondition` | Mix: `cco:DescriptiveICE` (price/range), enumerations under `cco:DesignativeICE` (DeliveryMethod, ItemAvailability) |
| **Temporal granularities** | `Date`, `DateTime`, `Time`, `DayOfWeek` | `cco:DescriptiveICE` temporal subtree (refinements of existing `TIMESTAMP`) |
| **Generic data types** | `Number`, `Boolean`, `Language` | Mix: `cco:DescriptiveICE` (Number, Boolean), `cco:DesignativeICE` (Language code) |
| **Enumerations** | `CategoryCode`, `EventStatusType`, `EventAttendanceModeEnumeration`, `GenderType`, `RestrictedDiet`, `BookFormatType` | `cco:DesignativeICE` (coded value identifiers) |
| **Person/identity attributes** | `GenderType`, `MusicArtistAT` | `cco:DescriptiveICE` (gender), `cco:Person` (artist) |
| **Place attributes** | `CoordinateAT`, `LocationFeatureSpecification`, `openingHours` | `cco:DescriptiveICE` (coordinates, hours), `cco:DescriptiveICE` (features) |
| **Annotations / commentary** | `category`, `label`, `Rating`, `Review`, `ItemList` | `cco:DescriptiveICE` |
| **Measurement helpers** | `unitCode`, `unitText` | Properties of `schema:QuantitativeValue` (we have); add as named properties |
| **Communication channel** | `faxNumber` | `cco:DesignativeICE` (sibling of `telephone`) |
| **Attribute-typed identifiers** | `IdentifierAT` | `cco:DesignativeICE` (sibling of `schema:identifier`) |

## Three-tier extension strategy

### Tier-A — measurement zoo (lowest effort, highest leverage)

Add ~10 `schema:QuantitativeValue` subclasses under `cco:DescriptiveICE`:

```turtle
schema:Distance       rdfs:subClassOf cco:ont00000853 .  # Descriptive ICE
schema:Duration       rdfs:subClassOf cco:ont00000853 .
schema:Energy         rdfs:subClassOf cco:ont00000853 .
schema:Mass           rdfs:subClassOf cco:ont00000853 .
schema:Speed          rdfs:subClassOf cco:ont00000853 .
schema:Temperature    rdfs:subClassOf cco:ont00000853 .
```

Plus property-level: `unitCode`, `unitText`, `weight`.

Implementation: ~10 lines in `atelier-vocab.ttl`, ~5 generators in
`synth_generators.py` (already have NUMERIC.* generators that can be
re-keyed to schema URIs).

**SOTAB labels covered**: 10 (Distance, Duration, Energy, Mass, weight,
unitCode, unitText, Number, Boolean, plus one ancillary).

### Tier-B — subclass plumbing (CreativeWork + Organization + Event subtrees)

Single-edge additions for entity types already grounded at parent level:

```turtle
schema:Book           rdfs:subClassOf schema:CreativeWork .
schema:Movie          rdfs:subClassOf schema:CreativeWork .
schema:Recipe         rdfs:subClassOf schema:CreativeWork .
schema:MusicAlbum     rdfs:subClassOf schema:CreativeWork .
schema:MusicRecording rdfs:subClassOf schema:CreativeWork .
schema:TVEpisode      rdfs:subClassOf schema:CreativeWork .
schema:Photograph     rdfs:subClassOf schema:CreativeWork .
schema:Review         rdfs:subClassOf schema:CreativeWork .
schema:Hotel          rdfs:subClassOf schema:Organization .
schema:LocalBusiness  rdfs:subClassOf schema:Organization .
schema:Museum         rdfs:subClassOf schema:Organization .
schema:Restaurant     rdfs:subClassOf schema:Organization .
schema:SportsTeam     rdfs:subClassOf schema:Organization .
schema:SportsEvent    rdfs:subClassOf schema:Event .
```

Implementation: ~14 lines in `atelier-vocab.ttl`, ~14 SSSOM annotation
blocks, ~14 generators in `synth_generators.py`.

**SOTAB labels covered**: ~20 (entity-property pairs cascade through
parent's `name`/`description` mappings).

### Tier-C — Product branch + JobPosting + economics

Largest single addition; introduces `cco:Artifact` lineage:

```turtle
schema:Product        rdfs:subClassOf cco:Artifact .       # NEW branch
schema:ProductModel   rdfs:subClassOf cco:ArtifactModel .  # NEW
schema:Brand          rdfs:subClassOf cco:DesignativeICE . # NEW
schema:JobPosting     rdfs:subClassOf cco:DescriptiveICE . # NEW
```

Plus property-level mappings: `price`, `priceRange`, `currency`,
`paymentAccepted`, `DeliveryMethod`, `ItemAvailability`,
`OfferItemCondition`, `workHours`, `OccupationalExperienceRequirements`,
`EducationalOccupationalCredential`.

Plus temporal refinements: `Date`, `DateTime`, `Time`, `DayOfWeek` as
subproperties of existing `TIMESTAMP` lineage.

Plus enumerations: `CategoryCode`, `EventStatusType`,
`EventAttendanceModeEnumeration`, `GenderType`, `RestrictedDiet`,
`BookFormatType` (~6 enumeration classes).

Implementation: ~30 lines in `atelier-vocab.ttl`, comparable SSSOM
annotation overhead, ~25 new synth generators (Product family is its own
generator pack: SKU, brand, model, GTIN, etc.).

**SOTAB labels covered**: remaining ~48.

### Cumulative coverage after all three tiers

100% of the 82 SOTAB v2 Schema.org CTA labels mapped to BFO/CCO grounding,
with provenance trails (SSSOM `sssom:object_label` axioms) for every
mapping.

## Ownership flow (post-migration)

| Concern | Owner | Notes |
|---|---|---|
| Vocabulary IRIs + CCO/BFO grounding | **Ægir** (target) | `atelier-vocab.ttl` migrates to `aegir/src/aegir/ontology/` |
| Synth value generators | **Ægir** (target) | Generator output feeds pre-training corpora directly |
| SOTAB-label → vocab-IRI lookup | **Ægir** | `aegir-vocab` exposes label↔IRI map; consumed by training + inference |
| SOTAB v2 download + extraction | **Ægir** | `scripts/download_sotab.py` (already wired) |
| CTA/CPA dataset loaders | **Ægir** | `src/aegir/data/table_dataset.py` (already wired) |
| Model training + evaluation | **Ægir** | `train.py`, `src/aegir/models/heads.py::AegirForColumnAnnotation` |
| Per-class F1 + Pareto evaluation | **Ægir** | M2 roadmap entry (per-class F1 bars in leaderboard UI) |
| BFO-grounded prediction emission | **Ægir** | Leaderboard predicts SOTAB label AND emits its CCO/BFO ancestry |
| Trained checkpoint consumption | **Atelier** | New: load H-Net/RWKV + SVM artifacts as DST evidence sources |
| DST evidence fusion + classification pipeline | **Atelier** | Unchanged — trichotomy + belief/plausibility logic stays |
| Gateway + UI + governance integration | **Atelier** | Unchanged |

During the migration window (until ontology fully relocates), atelier
keeps its operational `atelier-vocab.ttl` snapshot.  The concrete
contract Ægir publishes to atelier becomes a **`vocab_label_map.json`**
(IRI + BFO ancestry per SOTAB label) plus the trained model checkpoints
themselves.

## Aegir touchpoints (informative, not prescriptive)

The work in Ægir, in roadmap terms, lands inside its M2 milestone
("external-baseline harness, ontology editor with Postgres write paths,
per-class F1 bars"):

- **`src/aegir/data/table_dataset.py`** — fix the stale `_LABEL_DIMS["sotab"] = 91` to `82`; add a `label_to_iri` resolver that consumes
  the shared `sotab_label_map.json`.
- **`scripts/sotab_diagnostic.py`** — extend representation-collapse
  diagnostics to surface per-tier (A/B/C) coverage of predictions, so we
  can see whether collapses correlate with vocab gaps.
- **Leaderboard gateway** (`src/aegir/gateway/app.py`) — `/api/ontology`
  endpoint already exists; extend its response to include the BFO ancestry
  of each predicted label.
- **`src/aegir/models/heads.py::AegirForColumnAnnotation`** — no model
  change needed for tier work; the head already operates on a `(num_labels,)`
  output, and 82 vs 91 is just a config delta.

The pretraining work documented in
`aegir/docs/notes/2026-04-19/234700_sotab_diagnostic_representation_collapse.md`
(model collapses to single embedding point on SOTAB-small) is **orthogonal
to this strategy** — it's a model issue, not a vocabulary issue.  Vocab
extension proceeds independently and should improve the post-collapse
ceiling once representations are healthy.

## Synthetic data pipeline implications

The synth framework (`synth_generators.py`, 316+ hand-coded generators
plus the three-layer registry) **migrates to Ægir** with the rest of the
ontology work.  Ægir-resident synth gives pre-training direct access to
generator output without crossing repo boundaries.  After Tier-A/B/C
extensions:

- Tier-A adds **measurement generators** — `DURATION` (ISO-8601 strings),
  `MASS` (with unit suffix), `DISTANCE`, `ENERGY` — these are mostly
  numeric with unit annotations.  Existing `NUMERIC.*` generators can be
  re-keyed.
- Tier-B adds **entity-name generators** — `BOOK_TITLE`, `MOVIE_TITLE`,
  `RECIPE_NAME`, `HOTEL_NAME`, etc.  Cascade through the registry's
  `template` priority (priority 2): once Ægir has ~50 real Book/name
  samples from SOTAB itself, the registry generates plausible book
  titles via perturbation.
- Tier-C adds **product attribute generators** — SKU, Brand, GTIN,
  ProductModel.  Domain-specific; benefit from hand-coded generators
  (priority 1) seeded with realistic patterns.

Atelier's classification pipeline, post-migration, can either (a) call
into Ægir's synth via a thin client during local dev, or (b) bundle a
generator snapshot at release time.  The decision depends on whether
Atelier's BDD/pytest scenarios remain self-contained or are content to
require Ægir as a sibling repo.

## Verification

Coverage is mechanically verifiable via SPARQL totality:

```sparql
PREFIX cco: <https://www.commoncoreontologies.org/>
PREFIX schema: <https://schema.org/>

# Every SOTAB label must have a path to cco:InformationContentEntity (or descendant).
SELECT ?label WHERE {
  VALUES ?label { schema:Distance schema:Duration schema:Mass ... }  # all 82
  FILTER NOT EXISTS {
    ?label rdfs:subClassOf+ cco:ont00000958 .  # cco:InformationContentEntity
  }
}
# Empty result == 100% coverage.
```

This goes in `src/atelier/classify/ontology/sparql/sotab_totality.rq`
once Tier-A lands.

## Status

- **Strategy doc**: this file (2026-05-09).
- **Ownership migration**: Ægir takes over ontology / vocab / synth.
- **Tier-A implementation**: Ægir M2 — vocab edits + SSSOM annotations +
  SPARQL totality query + measurement generators.
- **Tier-B implementation**: Ægir M2.
- **Tier-C implementation**: Ægir M3.

Atelier's contribution post-migration is **consumption-side**: load Ægir's
trained checkpoints as DST evidence sources, surface BFO ancestry via the
gateway/UI, integrate predictions into the existing belief/plausibility
fusion machinery.  The vocabulary itself, and the work to extend it,
lives next to the model that uses it.
