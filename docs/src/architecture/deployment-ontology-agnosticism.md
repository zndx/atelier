# Deployment: Unseen Ontology, Known Schema

> **Operating principle**: out here we *iterate* on public benchmarks; in
> CAI we *execute* to a customer-specified objective.  The customer
> brings an unseen ontology shaped like a known annotations schema; the
> system has to produce classifications + calibrated belief intervals
> against that ontology without prior calibration.

This document captures the deployment-time invariants that the
classification pipeline must honor, names the assumptions baked into
today's code that *would* fail against a sufficiently weird customer
ontology, and proposes a roadmap milestone (**M11 — Bring Your Own
Vocabulary**) that closes the remaining gaps and makes public-data
iteration a *test surface* rather than a *target* for the same
execution path.

## Mode split — iteration vs. execution

| Dimension | Iteration mode (local / public data) | Execution mode (CAI deployment) |
|---|---|---|
| Vocabulary source | `atelier-vocab.ttl` (300 ICE leaves) + curated mappings + benchmark-specific class lists (SOTAB 82 Schema.org types, GitTables 122 DBpedia types, SemTab DBpedia hierarchy) | Customer's `default.annotations` Hive table — opaque to us until run-time |
| Hierarchy depth | Known (5 levels for ICE, ~3 for DBpedia subset, varying for Schema.org) | Unknown — could be 1 (flat) or 8+ (deep regulatory taxonomy) |
| Hierarchy shape | Tree, single root | Tree assumed; multi-root forest, cycles, unbalanced subtrees all plausible |
| Validation labels | Curated reference (synth, meta-tagging UAT, GitTables CTA gold) | Often absent.  Sometimes a small spot-check set; sometimes none. |
| Accuracy bar | Track records over time on published benchmarks | Customer-stated objective; calibration + sample review when no agent-mediated reference exists |
| BFO / CCO grounding | Available — we mapped 360 terms ourselves | Opportunistic — only if the customer's ontology happens to carry a `bfo_anchor` / `cco_anchor` / `schema_org_class` / `dbpedia_class` column |
| Iteration latency | Tight (re-run with overlay tweaks; soak on devenv) | Wide (CAI session lifecycle; nautilus + overwatch loops the only mid-run feedback) |

The bridge between the two modes is structural: **every iteration
target gets transformed into annotations-schema shape before the
pipeline sees it**.  SOTAB v2's class list, GitTables' 122 DBpedia
types, the OOTB sample's 316 ICE leaves, the customer's Hive table —
all four end up as a `HierarchicalCategorySet` built from a
`list[ReferenceCategory]` with `parent_code` edges, fed into the same
DST + agent + nautilus + overwatch stack.  The execution path doesn't
know or care which mode it's running under.

## The annotations schema contract (what's stable)

`load_annotations_from_hive` reads `SELECT * FROM default.annotations`
and returns `list[dict]`.  `_normalize_annotations_row` and
`_build_category_set_from_records` (both in
`src/atelier/classify/taxonomy.py`) translate that into a
`HierarchicalCategorySet`.  The fields we already accept, in order of
preference per row:

| Field | Required | Purpose | Fallback when missing |
|---|---|---|---|
| `code` (or `id` / `path` / dot-path) | yes | Identity for tree navigation, DST focal element, Atlas type name | row dropped (we cannot classify into an unnamed term) |
| `label` (or `display_name` / `name`) | strongly preferred | Human surface in UI + LLM prompt | falls back to last component of `code` |
| `parent_code` | no | Explicit parent edge | derived from dot-path (`A.B.C` → parent = `A.B`) |
| `description` | no | LLM context, embedding text | empty |
| `common_names` (synonyms / aliases) | no | LLM expansion + embedding text | empty |
| `notation` | no | SKOS-style dot code (numeric or otherwise) | empty |
| `abbrev` | no | Mnemonic shortcode for leaves | empty |
| `taxonomy` | no | Namespace discriminator | `"annotations"` |
| `sensitivity` | no | Domain-specific classification metadata | absent |

The contract is **structural, not semantic** — we do not require any
particular set of root codes, depth, or ICE-trichotomy alignment.  A
customer ontology rooted at `LEGAL.PRIVILEGE.ATTORNEY_CLIENT` is
structurally indistinguishable from one rooted at
`ICE.SENSITIVE.PID.CONTACT.EMAIL` from the algorithms' point of view.

## What's already ontology-agnostic

Most of the algorithmic surface from v0.4.0-rc1 operates on the
hierarchy *as a graph*, not on ICE-specific anchors:

- **Parent-aware DST frame** (`mass_functions.py`) — votes at any node;
  fold-up uses `HierarchicalCategorySet.descendants(code)`.
- **Hierarchical maxsim mass** (Shafer §3, `maxsim_to_mass`) —
  distributes ColBERT late-interaction similarity through the graph
  regardless of root identity.
- **Cross-subtree cautious_code** (Smets §6) — least-commitment
  promotion finds the deepest common ancestor on whatever tree is
  loaded.
- **Belief-path tracing** (`belief.py::belief_path`) — walks
  `parent_code` chains; doesn't care about labels.
- **Indep-tier revisit gate** — fires on consensus disagreement, not on
  a code-pattern match.
- **Atlas type graph export** (`HierarchicalCategorySet.atlas_type_graph`)
  — turns any tree into Atlas Classification typedefs with `superTypes`
  chains.
- **Validation** (`validate_taxonomy`) — collision + duplicate detector
  catches structural problems before classify starts (cycles emerge as
  parent_code self-reference; multi-root surfaces as multiple
  parent-less entries).
- **Cautious-Code Review** (`cautious_review.py`) — agent-mediated
  backoff is structural; the agent reasons about depth-vs-confidence
  on whatever tree it sees.

The DST math doesn't know it's classifying PII.  That's a feature — it
means the work we did on v0.4.0-rc1 transfers to deployment with
zero algorithm changes.

## What anticipates badly today

Five gaps that an unseen customer ontology will surface on first
encounter:

### 1. Schema flexibility — column-name variants

`_normalize_annotations_row` matches a fixed set of column-name
candidates.  Customers regularly bring annotation tables with names
like `Class_Name`, `parent_class`, `category_definition`,
`sensitivity_tier`, `pii_category` — none of which exactly match our
preferred field names.  Today this falls through to silent drops or
empty fields.

**Fix**: extend the column-name normalization to be configurable and
fuzzy.  Add a `vocab_schema_map` overlay setting that lets the
operator declare `{ "code": "Class_Name", "parent_code": "Parent" }`
at run-time.  Default behavior stays automatic via fuzzy matching on
common synonyms.

### 2. Hierarchy-shape resilience

Today's calibration assumes our 5-level ICE depth.  Discount defaults
(maxsim `0.20`, llm `0.15`, SVM `0.22`), gap_threshold `0.18`, and
cautious_review bel_threshold `0.0` (cautious review is off by default)
are all tuned against that.  A customer's flat 50-class taxonomy doesn't
need cautious-code review (no parents to back off to), and an 8-deep
regulatory taxonomy demands tighter cautious thresholds (more depth ×
more places to be wrong).

**Fix**: depth-aware defaults.  Compute hierarchy statistics at
LOADING_VOCAB time (max depth, mean branching factor, leaf/internal
ratio); apply scaled defaults if the operator hasn't overridden them.
Surface the stats in the Status page so the operator sees what they
got.

### 3. Multi-root and cycle handling

Single-root tree is assumed in `descendants` / `ancestors` traversal.
Customer dumps can have multiple top-level concepts (a forest), or —
rarely but consequentially — a cycle introduced by data entry error.
Today: cycles cause an unbounded loop in `descendants` (the traversal
uses an iterative stack with no visited-set, so a cyclic edge re-enqueues
its children forever); multiple roots silently work *because* the
traversal is parent-anchored, but pre-classification tooling (Atlas
export, vocabulary stats UI) breaks.

**Fix**: explicit multi-root support in
`HierarchicalCategorySet`.  Cycle detection + clear error in
`validate_taxonomy` with the offending edge identified.  Both behind a
feature flag so pathological customer data fails fast rather than
hangs.

### 4. Opportunistic CCO/BFO grounding

Customer ontologies that overlap with Schema.org / DBpedia / BFO /
CCO carry that overlap as metadata columns (e.g.,
`schema_org_class`, `bfo_anchor`, `cco_class`).  Today we ignore
these.  Wiring them lets us:

- Auto-validate the customer's hierarchy against a known reference
  (warn on inconsistent BFO anchoring; e.g., a node mapped to
  `cco:DesignativeICE` whose children include a `cco:Agent`).
- Reuse our 360-term mapping for embedding-text enrichment (a
  customer term mapped to `schema:Person` borrows the full
  description from the Schema.org corpus).
- Bridge to Atlas BFO classifications when the customer's
  governance team is ahead of theirs (Cloudera Atlas now ships BFO
  alignment as of mid-2025).

**Fix**: optional `bfo_anchor` / `cco_class` / `schema_org_class` /
`dbpedia_class` columns in the annotations contract; when present,
the loader populates them on `ReferenceCategory` and the embedding +
LLM-prompt builders consume them.  When absent, no behavior change.

### 5. Accuracy reporting without an agent-mediated reference

The customer often has *no* per-column gold-standard labels.  Our
v0.4.0-rc1 evaluation pipeline assumes a `curated_reference` table
(or per-row `reference_code` field).  When the customer doesn't
provide one:

**What we have**: belief-gap distribution, mean K, cautious-code
depth distribution, cross-source agreement counts, reasoning-trace
attribution analyzer.  These are *calibration* metrics, not accuracy.

**What we need**: a deployment-mode evaluation report that's honest
about the absence of an agent-mediated reference.  Three-tier report:

- **Internal consistency** — DST K stats, belief-gap distribution,
  contraction rate.  Always available.  Tells the operator the
  pipeline converged.
- **Sample review workflow** — eject N highest-uncertainty columns
  and N highest-confidence columns to the UI for human spot-check.
  The operator's accept/reject decisions feed an ad-hoc curated
  reference that grows over time.  This is essentially what UAT
  reviewers were doing manually; we can formalize it.
- **Public-benchmark proxy** — when the customer's ontology overlaps
  with SOTAB / GitTables / SemTab through opportunistic CCO grounding
  (gap #4), accuracy on the public benchmark serves as a
  conditional-confidence floor.

## Public-data iteration as test surface

The principle: **every public benchmark we adopt becomes a
deployment-shape simulator**, not a one-off integration.  Concretely:

1. **SOTAB v2** — wire as a classify source by transforming
   the 82 Schema.org type list into a `HierarchicalCategorySet`-
   shaped annotations table.  The Schema.org type tree provides
   parent_code edges; our existing `atelier-vocab.ttl` mappings
   (`schema:Person → cco:Agent`, etc.) opportunistically populate
   the BFO/CCO anchors on the resulting `ReferenceCategory` rows.
   Pipeline runs against SOTAB tables exactly as it would against
   a customer Hive corpus.

2. **GitTables** — same treatment.  The 122 DBpedia types become
   a flat (or DBpedia-hierarchy-enriched) annotations table.  Our
   15 already-mapped DBpedia → CCO bridges populate anchors where
   they exist; the other 107 stay un-anchored (correct behavior
   for opportunistic grounding).

3. **SemTab annual** — register the system, produce the annotations
   table from each year's vocabulary release, evaluate against the
   `cscore` metric (which natively rewards our cautious_code).

4. **Customer schema simulators** — synthetic annotation tables
   that test specific deployment shapes: flat 50-class taxonomy
   (legal exemption codes), 8-deep regulatory tree (HIPAA
   subcategories), forest with 3 roots (multi-domain governance).
   These exercise the M11 shape-resilience work without needing
   real customer data.

Each iteration target ships as a `data_sources` row + a loader (one
function each) + an annotations table built from the benchmark's
class list.  None of them need pipeline-side knowledge.

## M11 — Bring Your Own Vocabulary (proposed)

A milestone that delivers ontology-agnostic execution with the
five gaps above closed:

1. **Configurable vocab schema mapping** — overlay setting + fuzzy
   default; surfaces in Status when applied.
2. **Depth-aware default calibration** — compute hierarchy stats
   at load; scale gap_threshold + cautious bel_threshold.
3. **Multi-root + cycle support** — explicit; behind feature flags
   that fail loudly when violated.
4. **Opportunistic anchor columns** — `bfo_anchor` / `cco_class`
   / `schema_org_class` / `dbpedia_class` consumed when present.
5. **Three-tier deployment evaluation report** — internal
   consistency / sample review workflow / public-benchmark proxy.
6. **SOTAB v2 + GitTables wired as test sources** — proof that
   the same execution path handles three published benchmarks
   plus the customer's Hive table without code changes per
   target.

The work is concrete and bounded — roughly two focused sessions
(taxonomy.py + pipeline.py extensions; one loader + one fixture
test per benchmark).  Stronger leverage than a feature-by-feature
roadmap because every fix lands on the existing structural
abstraction rather than introducing new mechanisms.

## Out of scope (deferred)

- **Cross-customer ontology learning**.  Two customers with
  similar regulatory domains might benefit from shared inferences;
  we explicitly do not transfer learning across deployments.
  Each customer's session is a closed world.
- **Customer-driven hierarchy editing**.  The annotations table
  is a contract the customer controls upstream of Atelier.  We
  don't ship UI for editing it.
- **Ontology auto-discovery**.  Inferring a hierarchy from
  unannotated data tables (clustering plus LLM proposes a tree)
  is a research direction in its own right; out of scope for
  M11.

## Open questions

- **What if the customer brings two annotations tables**?  A
  primary domain vocabulary (`hipaa.annotations`) and a generic
  PII overlay (`atelier.annotations`).  Today's pipeline takes
  one.  M11 should consider `compose_vocabularies` in the
  loader path, but it changes the meaning of "the customer's
  ontology" — needs a design conversation.
- **Embedding-model robustness across languages**.  A German /
  Japanese / Mandarin annotations table will produce shorter
  embedding-text and weaker cosine signal at MiniLM-L6 scale.
  Bigger embedding models (BGE-large, E5-mistral) help but
  inflate per-run cost.  Defer to a separate i18n milestone.
- **Atlas BFO sync**.  Cloudera's Atlas team is shipping BFO
  alignment.  Once stable, our `atelier-vocab.ttl` ↔ Atlas
  classification typedef mapping should round-trip without loss
  (we ship to Atlas; Atlas hands back BFO-anchored entities; we
  read them as opportunistic anchors per gap #4).  Wait for
  Atlas BFO general availability before wiring.

## Cross-references

- [Classification Pipeline](./classification.md) — the execution
  path being made ontology-agnostic.
- [DST Evidence Independence](./dst-evidence-independence.md) — the
  numerical-methods framing that already operates on arbitrary
  hierarchies.
- [Pareto Capability Evolution](./pareto-capability-evolution.md) —
  the longer-horizon search-space that builds on M11.
- `src/atelier/classify/ontology/README.md` — the BFO/CCO substrate
  that opportunistic anchoring lifts into.
- `src/atelier/classify/taxonomy.py` — `_normalize_annotations_row`,
  `_build_category_set_from_records` (the adapter layer).
- `src/atelier/classify/sampler.py` — `load_annotations_from_hive`,
  `load_annotations_from_json`, `load_annotations_from_filesystem`
  (the source-shape variants).
