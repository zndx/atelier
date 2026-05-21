<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Late-Interaction Multi-Vector Cosine via Qdrant

This note specifies the next layer of the cosine evidence source: a
multi-vector late-interaction (ColBERT-style) representation per
annotation, stored in Qdrant, with enrichment supplied by an Agent-SDK
curation loop and procedural deterministic verifiers.  It composes
with — does **not** replace — the reliability discounting, indep-tier
consensus gate, hierarchical mass aggregation, and cost-sensitive LLM
prompting documented in [`dst-evidence-independence.md`](dst-evidence-independence.md).

## Position in the architecture

The existing DST treatment shapes *how per-source masses fuse*.  This
work shapes the *cosine source's input representation*.  Both are
necessary; neither is sufficient on its own.

The motivating gap is structural rather than algorithmic.  Current
cosine compresses each annotation into a single embedding from
`label + mnemonic + description` and compares it to a single
column-side embedding from `column_name + concatenated_samples`.  On
adversarial corpora — anonymized column names (`comm_val`,
`period_val`, `addr_ref`), mixed sample distributions, vocab-token-as-
data columns — the single-vector representation collapses
discriminative signal before it reaches the fusion layer.  Reliability
shaping (Haenni-Hartmann 2006) can route mass to ignorance correctly
in this regime, but it cannot recover the discriminative signal that
was lost to the compression.

Late interaction via ColBERT restores the discriminative surface:
instead of one dense-vector comparison per (column, tag) pair, the
ColBERT encoder produces per-token contextual embeddings (128-d after
the linear projection) for both entity and annotation texts.  Qdrant's
native MaxSim comparator computes the token-level cross-alignment
score directly — no Python-side scoring loop, no per-role weight
tuning.

The entity side feeds `ColumnFeatures.to_embedding_text()` — the same
text SAGE/SHAP ablate over — through the ColBERT encoder.  The
annotation side feeds a composed text from the enrichment payload
(label, description, prototype values, name hints, value patterns,
parent path, mnemonic) through the same encoder.  Anti-examples are
excluded from the annotation text (they add noise in the embedding
space without improving MaxSim discrimination).

The motivating failure modes resolve through token-level alignment:

- **Anonymized columns** — column-name tokens contribute little
  MaxSim, but sample-value tokens still align to annotation prototype-
  value tokens.  Graceful degradation by token structure: weak tokens
  contribute near-zero MaxSim without polluting strong token matches.
- **Long-tail distinguishing values** — a single distinctive sample
  value's tokens claim their own MaxSim against annotation prototype
  tokens, no longer averaged out by a single dense vector.
- **Sibling discrimination** — token-level alignment discriminates
  between semantically adjacent annotations (e.g., "credit card
  number" vs "bank account number") through fine-grained token
  matching that dense single-vector cosine collapses.
- **Parent-pull** — parent-path tokens in the annotation text provide
  hierarchical context.  The hierarchical aggregation in
  `_late_interaction_positive_mass` continues to flow residual mass
  to internal-node focal elements when subtree-level signal is what's
  available.

This is morphologically close to what the upstream
[Ægir project](https://zndx.github.io/aegir/) provides through a learned
hierarchical foundation model (RWKV-7 time-mixing + H-Net dynamic
chunking, RLVR-trained against a deterministic four-component
verifier on SOTAB / GitTables / WikiTables).  The two are
complementary, not redundant: Ægir's representations are learned end-
to-end against external corpora; late-interaction here is engineered
from the user-selected taxonomy with LLM-augmented annotation
profiles.  Both can coexist as separate evidence sources, and the
late-interaction infrastructure remains useful even after Ægir
integration for taxonomies Ægir has not been adapted to.

## Architecture overview

```
┌─ Source taxonomy (default.annotations or any user-selected) ────┐
│  label, mnemonic, description, parent path                       │
└────────────────────┬─────────────────────────────────────────────┘
                     │
                     ▼  scripts/enrich_annotations.py
              ┌──────────────────────────────┐
              │ Agent SDK enrichment loop    │
              │  + deterministic verifiers   │
              └──────────────┬───────────────┘
                             │
                             ▼  ColBERT token vectors + payload
         ┌────────────────────────────────────────────┐
         │ Qdrant collection: annotations_<tax>_<ver> │
         │   - single "colbert" multi-vector field     │
         │     (per-token 128-d, MaxSim comparator)    │
         │   - structured JSON payload                 │
         │   - operator_edits audit log                │
         └────────────┬───────────────────────────────┘
                      │
                      │  registered in PGlite taxonomy_registry
                      │  (administrative pointer, never primary storage)
                      │
                      ▼  build/exports/<tax>-enriched-<ver>-<utc>.parquet|tsv
                  on-demand snapshots for operator inspection

  At classify time:
       ColumnFeatures.to_embedding_text()
                 │
                 ▼  ColBERT encoder (colbert-ir/colbertv2.0)
          entity token vectors (N × 128)
                 │
                 ▼  Qdrant query_points (using="colbert", MaxSim)
          top-K annotations ranked by MaxSim score
                 │
                 ▼  late_interaction_to_mass
          mass function (Haenni-Hartmann reliability shaping)
                 │
                 ▼  DST fusion (existing pipeline)
          belief, plausibility, conflict per tag
```

## Qdrant payload schema

The collection per (taxonomy_id, augmentation_version) is the source
of truth.  No parallel relational mirror.  One point per annotation.

### Vector field

Each annotation point carries a single multi-vector field:

| Name       | Type         | Source |
|------------|:------------:|--------|
| `colbert`  | multi-vector | ColBERT token-level embeddings of the composed annotation text |

The composed annotation text is produced by
`qdrant_writer.compose_annotation_text()` from the enrichment
payload: label, description, prototype values (up to 10), name hints
(up to 10), value pattern descriptions (up to 5), parent path
(ontology chain), and mnemonic.  Anti-examples are deliberately
excluded — they add noise in the embedding space without improving
MaxSim discrimination.

The ColBERT encoder (`colbert-ir/colbertv2.0`) produces per-token
128-dimensional vectors via BERT + a learned linear projection
(768 → 128).  Special tokens ([CLS], [SEP], [PAD]) are stripped;
only content tokens contribute to MaxSim.

The collection is configured with `MultiVectorConfig(comparator=MAX_SIM)`
so Qdrant computes token-level late-interaction scoring natively —
no Python-side scoring loop.

### Payload (JSON)

```jsonc
{
  // Source taxonomy fields, immutable passthrough
  "code":           "ICE.SENSITIVE.PID.CONTACT.EMAIL",  // or user-vocab equivalent
  "label":          "Email",
  "mnemonic":       "EMAIL",
  "description":    "RFC 5322 email addresses, including international forms.",
  "parent_code":    "ICE.SENSITIVE.PID.CONTACT",
  "parent_path":    ["Sensitive Data", "PII", "Contact", "Email"],

  // Enrichment fields, generated + verified
  "prototype_values":     ["jane.doe@example.com", "user@subdomain.example.org", ...],
  "value_patterns":       [
      {"kind": "regex",  "expr": "[^@\\s]+@[^@\\s]+\\.[^@\\s]+"},
      {"kind": "format", "expr": "local-part @ domain, RFC 5322"}
  ],
  "name_hints":           ["email", "e_mail", "email_addr", "contact_email", "msg_val"],
  "anti_examples":        [
      {"value": "+1-555-123-4567", "confusable_tag": "A_PHN", "reason": "phone-shaped"},
      {"value": "https://example.com/path", "confusable_tag": "SYSURL", "reason": "URL-shaped"}
  ],

  // Provenance + audit
  "augmentation_version":  "v1",                       // prompt template + verifier version
  "embedding_model":       "colbert-ir/colbertv2.0",
  "embedding_dim":         128,
  "generated_at":          "2026-05-16T20:00:00Z",
  "generated_by":          "agent-sdk:opus-4.7",       // model + harness identifier
  "verifier_results": {
      "prototype_values_match_patterns": true,
      "patterns_compile":                 true,
      "anti_example_targets_exist":       true,
      "parent_path_consistent":           true,
      "checks_passed":                    4,
      "checks_total":                     4
  },

  // Operator edits log — append-only, every edit recorded
  "operator_edits": [
      {
          "at":     "2026-05-17T09:14:00Z",
          "by":     "operator@example.com",
          "field":  "prototype_values",
          "op":     "remove",
          "value":  "test@test.test",
          "reason": "weak exemplar"
      }
  ],

  // Cross-reference
  "taxonomy_id":       "default",
  "taxonomy_version":  "2026-05-01"
}
```

### Cache key (content-addressed)

Rebuilds are idempotent under stable inputs.  The cache key for a
single annotation point is:

```
key = sha256(
    taxonomy_id ||
    taxonomy_version_hash ||
    augmentation_version ||
    embedding_model ||
    source_row_hash       // hash of label+mnemonic+description+parent_code
)
```

Skip-on-cache-hit during rebuilds; force-rebuild via CLI flag.  The
cache layer is responsible for invalidation on any input change.

### Collection naming

```
annotations_<taxonomy_id>_<augmentation_version>
```

Example: `annotations_default_v1`, `annotations_hivepoc_synth_v1`.
The PGlite registry row tracks which collection is `current` for a
given `taxonomy_id`; old collections remain queryable for A/B
comparison and rollback.

## Enrichment pipeline (high-level)

Detailed in `scripts/enrich_annotations.py` (P2) and the
`atelier.enrichment` package.  Vocabulary identity is *dynamic*:
operators select a `(connection, database, annotations_table)`
triple at runtime; the pipeline must not encode the count, names,
or structure of the currently-loaded set as intrinsic.  The single
universal is that **every node — leaf or internal — is a first-class
tagging target**, so both leaf and internal nodes receive enrichment.

The shape:

1. **Read** source taxonomy rows from the active annotations table
   selected by the operator at runtime.  No vocabulary identity
   is hardcoded.
2. **For each node** (leaf or internal), run the enrichment loop:
   - Build a generation prompt with parent-aware framing for
     internal nodes (children listed, "what does a column tagged at
     this generality look like without specializing to a child") or
     leaf-aware framing for leaves (sibling-discriminative patterns,
     concrete prototype values).
   - Call the **provider-co-located** generator (see below) to
     produce the six-field structured payload.
   - Run the deterministic verifier suite
     (`atelier.enrichment.verifiers`).  Failed checks become
     verifier feedback that is fed back into the next generation
     attempt up to `enrichment.max_attempts`.
   - Compute `parent_path` deterministically from the taxonomy
     structure (no LLM needed) and confirm the LLM's reasoning is
     consistent with it.
3. **Compute embeddings** for each named vector using the configured
   embedding model.
4. **Write** the multi-vector point + payload to Qdrant, keyed by the
   content-addressed cache key.  Idempotent: same `(vocabulary
   content hash, augmentation_version, embedding_model, source_row
   hash)` quadruple → same point ID → no redundant work on partial
   rebuilds.
5. **Update** the PGlite `taxonomy_registry` row to record the build
   (taxonomy_id, augmentation_version, collection name, `built_at`,
   status).  The registry is an *administrative pointer* — it
   records *that* a collection exists and *where*, never the
   primary content.

This pipeline satisfies the LLM-mediated reference artifact bar
(audited via memory): every output is procedurally reproducible from
its inputs and falsifiable by the verifier suite.

### Provider co-location with classify

The enrichment generator does NOT introduce a separate provider
knob.  It reads `cfg.classify_llm_backend` and uses the same
backend the classification path uses — operators manage one set of
credentials, one cost regime, one billing surface.  Within that
backend, the generator selects the strongest reasoning model
available, because per-node generation is single-shot and
benefits from extended deliberation on structural taxonomy
judgments (sibling discrimination, prototype induction, regex
synthesis).

Selection rule (highest priority first), implemented in
`atelier.enrichment.model_resolver.resolve_enrichment_model`:

1. `cfg.enrichment_model_override` (env: `ATELIER_ENRICHMENT_MODEL`)
   — explicit operator choice, used verbatim.
2. Per-backend apex constant when the platform owns the model
   identity (currently: `anthropic → claude-opus-4-7`).
3. Fall through to `cfg.classify_llm_model` for backends where the
   model identity is endpoint-owned (`openai_compatible`,
   `cerebras`) — the operator's served endpoint *is* the apex
   available to that deployment.
4. **Bedrock without `model_override` raises**
   `EnrichmentModelError` with an operator-facing remediation
   hint.  Bedrock model identities are AWS account + region +
   inference-profile specific; no portable default constant would
   be correct across deployments, and silently degrading to a
   weaker model would contradict the strongest-reasoning-model
   discipline.  This is a deployment-readiness gate consistent
   with the no-silent-DST-degradation principle.

The generator records `{backend}:{model}` in the point's
`generated_by` provenance field, so verifier pass-rate per node is
attributable to the exact provider+model combination — the unit of
replayable experiment.

### Parent-aware vs leaf-aware prompts

Both prompt variants produce the same six-field JSON schema, so
downstream code treats their outputs identically.  The framing
difference shapes content quality:

- **Leaf prompt** asks for values, patterns, and name hints
  describing what a column tagged *exactly* at this leaf would
  contain.  Patterns are narrow enough to discriminate against
  sibling leaves under the same parent.
- **Parent prompt** asks for what a column tagged *at this
  generality level — without further specificity to a child*
  looks like.  Children are listed so the model knows what
  specializations would NOT route here.  Anti-examples are
  hierarchically aware: the confusable_tag may be a sibling at the
  same level OR a sibling of an ancestor, because the late-
  interaction architecture's anti-example evidence applies
  regardless of where in the tree the confusable target lives.

## Late-interaction execution

### Column-side multi-vector representation

For each column being classified, build the multi-vector query:

| Query vector       | Source                                              |
|--------------------|-----------------------------------------------------|
| `col_name_view`    | `embed(column_name + " in " + table_name)`          |
| `col_sample_*`     | `embed(sample_value)` per deduped sample (top-N by frequency or distinctness, configurable) |
| `col_context_view` | `embed("table columns: " + concat(other column names in same table))` |
| `col_pattern_view` | `embed(extracted format hints from samples)`        |

`col_pattern_view` is computed from sample values via the existing
regex/validator detection in the pattern source — this is where the
original "regex as embedding-text enrichment" intent (referenced in
[`dst-evidence-independence.md`](dst-evidence-independence.md) and in
the upstream Ægir documentation) re-enters cleanly: regex outputs
contribute *structured features into one of the multi-vector query
slots*, not as an independent mass function competing with cosine.
The pattern source's standalone mass-function status is preserved
for narrow PII detection (email, IBAN, monetary, …) where its hits
are crisp; the `col_pattern_view` augmentation is additional, not a
replacement.

### MaxSim aggregation

For each candidate tag and each query vector, find the best match in
the annotation's multi-vectors of the *corresponding role*:

```
positive_score(col, tag) =
    sim(col_name_view,    label_view of tag)         * w_label
  + sim(col_name_view,    name_hints of tag)         * w_name
  + max(sim(col_sample_i, prototype_values of tag))  * w_proto_per_sample
  + max(sim(col_sample_i, value_patterns of tag))    * w_pattern_per_sample
  + sim(col_context_view, parent_path_view of tag)   * w_context
  + ...
```

Execution happens in-engine via Qdrant's multi-vector query API
with MaxSim comparator.  HNSW indexing brings the cost down to
logarithmic in the annotation count, which is the dominant cost
as vocabularies scale across deployments.

## Mass function construction

`mass_functions.late_interaction_to_mass(scores, frame)` produces a
`BeliefAssignment` over the candidate frame from the Qdrant MaxSim
scores.

The MaxSim score per tag is calibrated to evidence mass via the
same reliability-shaping pattern documented in
[`dst-evidence-independence.md`](dst-evidence-independence.md):
Haenni-Hartmann α-bounded reliability + margin-aware allocation.

- `α_abs` — sigmoid of top-1 MaxSim score.  "Is the best match
  strong enough to carry mass?"
- `α_marg` — `tanh((s₁ − s₂) / σ)`.  "Is the top-1 decisive?"

Allocation:

```
m(top-1)         = α · margin_weight + α · (1 − margin_weight) · softmax_top1
m(top-i, i > 1)  = α · (1 − margin_weight) · softmax_top_i
m(Θ)             = 1 − α
```

Hierarchical subtree aggregation (`_significant_subtree`) routes
residual mass to internal-node focal elements when subtree-level
signal dominates leaf-level signal.

## Storage philosophy

Single source of truth per layer, with administrative pointers in
PGlite.

| Layer            | Primary storage                            | Role                                                          |
|------------------|--------------------------------------------|---------------------------------------------------------------|
| Vectors + payload | **Qdrant** (`annotations_<tax>_<ver>`)    | Truth for enriched annotations; supports late-interaction execution |
| Run artifacts     | **`build/`** (existing pattern)            | Parquet, classifications, evaluation, sweep manifests, exports |
| Administrative    | **PGlite** (`taxonomy_registry`, run regs) | Where things live, at which version, in which status         |
| Future (planned)  | **Iceberg in S3**                          | Intermediates + `hx` history tables (taxonomy_history, enrichment_history, classification_runs_history, sweep_history); snapshot/time-travel for `hx` semantics native to Iceberg |

PGlite **never holds vectors, payloads, classifications, or
intermediates**.  Its job is to answer "where is the current
enriched annotation collection for taxonomy X?" and "which run
produced this dataset?"  Both registries are small, fast to query,
and survive backend migrations untouched.

When Iceberg-HX-in-S3 lands, the migration is a backend swap at the
registry layer — `pipeline_run_registry.artifacts_backend` flips
from `build_local` to `iceberg_s3`, `artifacts_path` switches to an
S3 URI, and pipeline logic remains unchanged.  Current `build/`
artifacts are forward-compatible with this transition.

### PGlite tables (P1.2 migration)

```sql
CREATE TABLE taxonomy_registry (
    taxonomy_id          TEXT PRIMARY KEY,
    source_table         TEXT NOT NULL,
    qdrant_collection    TEXT NOT NULL,
    qdrant_url           TEXT,
    augmentation_version TEXT NOT NULL,
    embedding_model      TEXT NOT NULL,
    embedding_dim        INTEGER NOT NULL,
    built_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status               TEXT NOT NULL DEFAULT 'building',
        -- 'building' | 'current' | 'stale' | 'archived'
    summary              TEXT
);

CREATE INDEX idx_taxonomy_registry_current
    ON taxonomy_registry(taxonomy_id, status);

-- Extends fsm_runs to record which enriched annotation collection
-- the run consumed.  NULL = legacy cosine; non-NULL = late-interaction.
ALTER TABLE fsm_runs ADD COLUMN IF NOT EXISTS
    taxonomy_collection TEXT REFERENCES taxonomy_registry(qdrant_collection);
```

## Operator inspection and edit surface

The active enriched-annotations collection in Qdrant (whatever the
operator's runtime vocabulary selection happens to produce) is
operator-facing through two surfaces:

**On-demand export** (`scripts/export_enriched_annotations.py`,
P2.4): writes the Qdrant payload for a given (taxonomy_id, version)
to `build/exports/<tax>-enriched-<ver>-<utc>.parquet` and a
human-readable `.tsv`.  Read-only snapshots, diffable across
versions, dropable when no longer needed.  Operators inspect via
their existing tooling (parquet viewers, spreadsheet apps,
`mlr`/`q`/`duckdb` for CLI).

**Structured edit CLI** (`scripts/edit_enriched_annotation.py`,
deferred — part of P2 follow-on): operators issue targeted edits
(add/remove prototype value, rewrite anti-example, etc.) which:

- Write back to the Qdrant point's payload + re-embed affected views
- Append an entry to the `operator_edits` audit log
- Bump a per-row revision counter (separate from
  `augmentation_version`, which is the system-level prompt/verifier
  version)

Edits are reversible — the audit log carries the prior value for
every change.  Per-customer overlays (deployment-specific
augmentations beyond the base) follow the same shape on a separate
edits stack.

## SHAP / SAGE shift under late interaction

The structured per-segment inputs (column_name, each sample,
context, pattern view) provide natural attribution surfaces that
the prior single-vector representation flattened.

**SHAP becomes per-decision interpretability infrastructure.** For a
column predicted EMAIL, SHAP attributes the score across the
structured inputs: "sample_3 contributed 0.42 via match against
`EMAIL.prototype_values[7]`; column_name contributed 0.08 via
`name_hints`; everything else < 0.05."  Operator-legible
explanation per prediction, computable in-pipeline at moderate
cost (one late-interaction pass per perturbation).  Wired into
`features.FEATURE_NAMES` as new ablatable feature slots:
`late_interaction_positive`, `late_interaction_negative`,
`late_interaction_view_<name>`.

**SAGE moves to offline-first.**  Late-interaction inputs are
richer (more "features" — per-view contributions, per-vector
contributions), and SAGE's permutation-based global compute scales
with that dimensionality.  Per-pipeline-run SAGE becomes
impractical and, more importantly, of low marginal value: SAGE's
value proposition is *corpus-level stability* rather than per-run
signal.  The shift:

- SAGE runs as a separate offline pipeline, scheduled or
  on-demand, against the current enriched annotations + corpus
  characterization.
- Artifact written to
  `build/sage/<corpus_id>-<taxonomy_version>-<utc>.parquet`.
- Downstream consumers (UI, view-prioritization, operator
  dashboards) reference the cached artifact; the pipeline hot path
  never recomputes inline.
- Optional integration: SAGE importance scores prioritize which
  annotation views the late-interaction engine computes first, with
  early-exit when high-importance views already discriminate
  confidently — a wall-time win on large taxonomies.

CLAUDE.md already notes SAGE is optional; this makes "optional"
precise: optional in the hot path, scheduled-only otherwise.

## Integration with existing fusion mechanisms

Every mechanism in [`dst-evidence-independence.md`](dst-evidence-independence.md)
composes cleanly with this work.  Specifically:

| Existing mechanism                                         | Composes by                                                                   |
|------------------------------------------------------------|-------------------------------------------------------------------------------|
| Reliability discounting (Shafer §11.3)                     | Late-interaction cosine carries its own discount slot in `config/base.conf`; default starts at `cosine` value (0.20) and is sweep-tunable. |
| Indep-tier consensus + revisit gate                        | Late-interaction cosine remains in the independent tier (its only LLM dependence is the enrichment, which is offline + verified). Indep-tier fusion picks it up unchanged. |
| Cosine reliability shaping (Haenni-Hartmann 2006)          | The α-bounded + margin-aware allocation pattern is reused for the positive channel; quality indicators extend to include verifier-pass-rate. |
| Hierarchical mass aggregation + cross-subtree visibility   | The positive-channel mass function emits hierarchical mass identically: walk up from top-1 leaf to the most-specific subtree capturing ≥ 50% of softmax probability, redirect residual to internal-node focal element. `cautious_promoted_code` walks the full hierarchy as before. |
| Cost-sensitive classification at LLM layer (Elkan 2001)    | Unchanged — operates upstream of fusion and is orthogonal to the cosine representation. |
| Pattern-target alias resolver                              | Unchanged for the standalone pattern source. The pattern source's hits additionally enrich the `col_pattern_view` query vector. |
| Per-column residual trajectory                             | Unchanged — operates on the iteration history of fused belief, which still flows through `BootstrapState`. The late-interaction cosine's per-view scores can be added to the snapshot for finer-grained trajectory analysis (deferred). |

## Configuration

New keys under `classify.cosine.late_interaction` in `config/base.conf`:

```hocon
classify {
  cosine {
    # Late-interaction multi-vector cosine is the production cosine
    # source.  Default ON.  The legacy single-vector cosine path
    # remains in the code only as a transitional emergency fallback;
    # when the late-interaction flag is on and the path cannot run
    # (no enriched collection, Qdrant unreachable, qdrant-client
    # missing), the pipeline logs WARNING + marks the run degraded
    # via `cosine_path` in the per-column result.
    late_interaction {
      enabled = true
      enabled = ${?ATELIER_CLASSIFY_COSINE_LATE_INTERACTION}

      model = "colbert-ir/colbertv2.0"
      model = ${?ATELIER_COLBERT_MODEL}

      qdrant_url = "http://127.0.0.1:6333"
      qdrant_url = ${?ATELIER_QDRANT_URL}
    }
  }
}
```

Existing `classify.cosine.*` keys are unchanged; the late-interaction
path is the production cosine source under this design.  The flag
exists for emergency rollback only — leaving the pipeline in legacy
single-vector cosine is a deployment-degraded state, not a normal
operating mode, and runs in that state are tagged with
`cosine_path: "legacy_degraded:<reason>"` in the per-column result
so the degradation is visible in operator-facing artifacts.

## Deferred work

- **Synthia / copula-aware column-side patterns**: when the
  SVM-on-synthetic work lands (separate track), the column-side
  multi-vector can include copula-derived inter-column dependency
  features as additional query vectors.  The query-vector slot is
  already structurally available; only the feature extractor needs
  to land.
- **Aegir CTA + CPA outputs as additional query vectors**: when
  Aegir integration lands, its predictions (and its CPA / cross-
  table grouping outputs) can enter the column-side multi-vector
  as supplementary query views.  Same structural slot.
- **Per-deployment edit overlays** with separate version stack from
  the base augmentation.  Schema for the overlay is sketched above;
  implementation deferred until operator workflow is validated.
- **Iceberg-HX-in-S3 backend** for the on-demand exports + run
  artifacts.  Designed-for; not yet built.

## References

- Khattab, O. & Zaharia, M. (2020). ColBERT: Efficient and Effective
  Passage Search via Contextualized Late Interaction over BERT.
  *SIGIR '20*, 39–48.  Introduces the late-interaction MaxSim formulation.
- Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C., & Zaharia,
  M. (2022).  ColBERTv2: Effective and Efficient Retrieval via
  Lightweight Late Interaction.  *NAACL 2022*.  Refines the MaxSim
  scoring + residual compression.
- Qdrant multi-vector named-vectors API:
  <https://qdrant.tech/course/multi-vector-search/module-1/late-interaction-basics/>
- Shafer, G. (1976).  *A Mathematical Theory of Evidence.*  §11.3
  reliability discount.  (Reused per the existing DST treatment.)
- Smets, P. (1990).  The Combination of Evidence in the Transferable
  Belief Model.  *IEEE TPAMI* 12(5), 447–458.  Negative-channel
  framing.
- Haenni, R. & Hartmann, S. (2006).  Modeling Partially Reliable
  Information Sources.  *Information Fusion* 7(4), 361–379.  α-bounded
  reliability shaping reused here.
- Companion architecture note:
  [`dst-evidence-independence.md`](dst-evidence-independence.md)  —
  reliability discounting, indep-tier consensus, hierarchical
  aggregation, cost-sensitive LLM prompting.
- Upstream foundation-model work:
  <https://zndx.github.io/aegir/> (hierarchical byte-level sequence
  model + RLVR-trained ontology policy for CTA/CPA/cross-table
  grouping; complementary independent evidence source on a longer
  timeline).
