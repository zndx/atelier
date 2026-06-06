# MaxSim Channel — ColBERT Late-Interaction via Qdrant

> **Naming.** This DST evidence channel is named **`maxsim`** — after the
> scoring operation Qdrant performs (a sum of per-query-token max cosines
> over the ColBERT multi-vector field), not the single-vector cosine it
> replaced. The per-token metric is cosine and the encoder is ColBERT, but
> the channel's identity — the key in `source_masses`, `INDEPENDENT_TIER`,
> the `classify.maxsim.*` config namespace, and the `classify.discounts.maxsim`
> discount — is `maxsim`. The legacy single-vector `cosine` channel is
> retired (no fallback). Historical sprint notes may still say "cosine".

> **What this is *not*.** ColBERT here is **text** late-interaction
> (BERT token embeddings over the entity/annotation text). This is
> **not** ColPali / ColVision / ColQwen or any vision-document
> retrieval model — there is no image or page-rendering pathway
> anywhere in the channel.

This note specifies the maxsim evidence source: a
multi-vector late-interaction (ColBERT-style) representation per
annotation, stored in Qdrant, with enrichment supplied by an Agent-SDK
curation loop and procedural deterministic verifiers.  It composes
with — does **not** replace — the reliability discounting, indep-tier
consensus gate, hierarchical mass aggregation, and cost-sensitive LLM
prompting documented in [`dst-evidence-independence.md`](dst-evidence-independence.md).

## Position in the architecture

The existing DST treatment shapes *how per-source masses fuse*.  This
work shapes the *semantic source's input representation*.  Both are
necessary; neither is sufficient on its own.

The motivating gap is structural rather than algorithmic.  The
retired single-vector cosine source compressed each annotation into a
single embedding from
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
  `_maxsim_positive_mass` continues to flow residual mass
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
                 ▼  maxsim_to_mass
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
3. **Compute embeddings** for the composed annotation text with the
   configured ColBERT model, producing the per-token vectors for the
   single `colbert` multi-vector field.
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
  hierarchically aware: the `confusable_tag` field (a vestigial name
  retained for schema stability — see `anti_example_targets_exist`
  verifier) may point to a sibling at the same level OR a sibling of
  an ancestor, because the late-interaction architecture's anti-example
  evidence applies regardless of where in the tree the negative
  exemplar lives.

## Late-interaction execution

There are **no** per-role query slots and **no** Python-side weighted
sum.  The flow is single-text in, single multi-vector query, native
in-engine scoring:

1. **One entity text.**  The column being classified is rendered to a
   single string by `ColumnFeatures.to_embedding_text()` — the same
   text SAGE/SHAP ablate over (column name, samples, context, pattern
   hints already composed in).
2. **One ColBERT multi-vector query.**  That text is encoded by the
   ColBERT encoder into per-token 128-d vectors (an `N × 128`
   multi-vector — *one* query object, not several role vectors).
3. **Single `colbert` field.**  Qdrant's `query_points(using="colbert")`
   matches the query multi-vector against each annotation point's one
   `colbert` multi-vector field.
4. **Native MaxSim.**  Under `MultiVectorConfig(comparator=MAX_SIM)`,
   Qdrant computes the late-interaction score for each candidate as the
   **sum, over query tokens, of each query token's maximum cosine
   against the annotation's tokens**, normalized by the query-token
   count.  No Python scoring loop; no per-role weights.

```
maxsim(query, tag) = (1 / |Q|) · Σ_{q ∈ Q}  max_{d ∈ D_tag}  cos(q, d)
```

where `Q` is the entity query's token vectors and `D_tag` the
annotation point's token vectors.  HNSW indexing keeps cost
logarithmic in the annotation count, which dominates as vocabularies
scale across deployments.

The pattern source's regex/validator hits feed the entity text
upstream (and the annotation text's `value_patterns`); the pattern
source also retains its **standalone** mass-function status for narrow
PII detection (email, IBAN, monetary, …) where its hits are crisp.
There is no separate `col_pattern_view` query vector — the per-role
multi-slot design below was a sketch that the single-`colbert`-field
implementation superseded.

### Deferred / not yet implemented: per-role multi-slot query

> **Not built.**  An earlier design split the entity into several
> role-tagged query vectors (`col_name_view`, `col_sample_*`,
> `col_context_view`, `col_pattern_view`) scored against
> correspondingly-role-tagged annotation vectors with a tunable
> weighted sum (`w_label`, `w_name`, `w_proto_per_sample`, …).  The
> shipped channel does **not** do this — it uses one entity text, one
> `colbert` multi-vector field, and Qdrant-native MaxSim (above).  The
> per-role slots, the SAGE-prioritized per-view early-exit, and the
> copula / Ægir supplementary query vectors in
> [Deferred work](#deferred-work) all presuppose this multi-slot query
> and are likewise unbuilt.

## Mass function construction

`mass_functions.maxsim_to_mass(scores, frame)` produces a
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

The per-tag allocation above is the `union_focal_k = 0` path.  With
the shipped default `union_focal_k = 3` (see
[Configuration](#configuration)), `maxsim_to_mass` instead bypasses
per-tag allocation and assigns `union_focal_alpha` mass to the **union
of the top-K candidates** as a single focal element
(`_maxsim_union_focal_mass`) — an "the answer is in this set" shape
that matches the channel's measured top-1/top-3 signal.

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
-- the run consumed.  NULL = no enriched collection; non-NULL = late-interaction.
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

## SHAP / SAGE under late interaction

**SHAP surface: a `maxsim_attribution` dict.**  When late-interaction
runs cleanly, `maxsim_bridge` attaches a compact attribution object to
the per-column result alongside `maxsim_path`.  Its shape is the top-K
candidate tags by MaxSim score:

```jsonc
{
  "ranking_basis": "qdrant_maxsim",
  "top_k": [
    {"code": "ICE.SENSITIVE.PID.CONTACT.EMAIL", "is_leaf": true,  "maxsim_score": 0.81},
    {"code": "ICE.SENSITIVE.PID.CONTACT",        "is_leaf": false, "maxsim_score": 0.74},
    {"code": "ICE.SENSITIVE.PID.CONTACT.PHONE",  "is_leaf": true,  "maxsim_score": 0.69}
  ]
}
```

This is the entire surface — there is **no** per-view / per-sample
feature breakdown, and **no negative channel** (the single-`colbert`
field carries one positive late-interaction score per candidate).  It
is **not** wired into `features.FEATURE_NAMES`: there are no
`late_interaction_positive`, `late_interaction_negative`, or
`late_interaction_view_<name>` slots — `FEATURE_NAMES` ablates the
entity-text segments (`column_name`, `sample_values`, `pattern_signals`,
`ontology_priors`, …), and SAGE/SHAP attribute through that surface.

**SAGE remains offline-first.**  SAGE's value proposition is
*corpus-level stability* rather than per-run signal, so it runs as a
separate scheduled/on-demand pipeline against the current enriched
annotations + corpus characterization, with its artifact under
`build/sage/`; downstream consumers (UI, operator dashboards)
reference the cached artifact and the pipeline hot path never
recomputes inline.  CLAUDE.md already notes SAGE is optional; this
makes "optional" precise: optional in the hot path, scheduled-only
otherwise.  (A SAGE-prioritized *per-view early-exit* would require the
deferred multi-slot query above and is not implemented.)

## Integration with existing fusion mechanisms

Every mechanism in [`dst-evidence-independence.md`](dst-evidence-independence.md)
composes cleanly with this work.  Specifically:

| Existing mechanism                                         | Composes by                                                                   |
|------------------------------------------------------------|-------------------------------------------------------------------------------|
| Reliability discounting (Shafer §11.3)                     | The maxsim source carries its own discount key `classify.discounts.maxsim` (default 0.20) and is sweep-tunable. |
| Indep-tier consensus + revisit gate                        | The maxsim source is in the independent tier (its only LLM dependence is the enrichment, which is offline + verified). Indep-tier fusion picks it up unchanged. |
| MaxSim reliability shaping (Haenni-Hartmann 2006)          | The α-bounded + margin-aware allocation pattern shapes the MaxSim score into mass; quality indicators extend to include verifier-pass-rate. |
| Hierarchical mass aggregation + cross-subtree visibility   | The mass function emits hierarchical mass identically: walk up from top-1 leaf to the most-specific subtree capturing ≥ 50% of softmax probability, redirect residual to internal-node focal element. `cautious_promoted_code` walks the full hierarchy as before. |
| Cost-sensitive classification at LLM layer (Elkan 2001)    | Unchanged — operates upstream of fusion and is orthogonal to the maxsim representation. |
| Pattern-target alias resolver                              | Unchanged for the standalone pattern source. The pattern source's hits also feed the entity text the ColBERT encoder embeds. |
| Per-column residual trajectory                             | Unchanged — operates on the iteration history of fused belief, which still flows through `BootstrapState`. |

## Configuration

The channel lives under `classify.maxsim` in `config/base.conf`, with
its reliability discount under `classify.discounts`:

```hocon
classify {
  maxsim {
    # Default ON.  When enabled and the path cannot run (no enriched
    # collection registered, Qdrant unreachable, qdrant-client
    # missing) the bridge raises MaxSimUnavailable and the run fails
    # fast in the FSM — no silent fallback to single-vector cosine.
    enabled = true
    enabled = ${?ATELIER_CLASSIFY_MAXSIM_ENABLED}

    # ColBERT model for token-level late-interaction embeddings.  Both
    # entity and annotation sides use the same model; Qdrant's native
    # MaxSim handles the token-level cross-alignment.
    model = "colbert-ir/colbertv2.0"
    model = ${?ATELIER_COLBERT_MODEL}

    # Top-K union focal element — "the answer is in this candidate
    # set" focal shape matching the channel's actual signal (top-1
    # ~60%, top-3 ~76% on the 5ef4868c reference).  K=0 keeps the
    # per-tag mass path; K=3 is structurally optimal (wider K dilutes
    # the focal).  See `mass_functions._maxsim_union_focal_mass`.
    union_focal_k = 3
    union_focal_k = ${?ATELIER_MAXSIM_UNION_FOCAL_K}
    union_focal_alpha = 0.45
    union_focal_alpha = ${?ATELIER_MAXSIM_UNION_FOCAL_ALPHA}
  }

  discounts {
    maxsim = 0.20               # Shafer §11.3 reliability discount
    # …
  }
}
```

**`qdrant_url` is not a primary config key.** The Qdrant endpoint is
resolved from the active `taxonomy_registry` row (set at enrichment
time — see the [PGlite tables](#pglite-tables-p12-migration) above).
`classify.maxsim.qdrant_url` (env `ATELIER_QDRANT_URL`) exists only as
the fallback used when the registry row's `qdrant_url` is null.

> **Rejected legacy keys.** `classify.cosine.late_interaction.*` and
> `classify.discounts.cosine` are in `_LEGACY_MAXSIM_KEYS` and are
> **loudly rejected** by the config loader (`config.py`) — they do not
> configure this channel and must never be documented as live.  The
> single-vector `cosine` channel they once named is retired.

### Fail-fast contract (no silent fallback)

When `classify.maxsim.enabled = true`, late-interaction is the **only**
supported path for this evidence source.  Per the
no-silent-DST-degradation rule, the legacy single-vector cosine
fallback was **removed** (`pipeline.py`, with an explicit "Do NOT
reintroduce" comment) — silently mixing a non-equivalent signal into
DST fusion historically destroyed accuracy (measured **−13.6pp**)
without the operator knowing.

- **Enabled but cannot run** (no enriched collection registered,
  Qdrant unreachable, qdrant-client missing, scoring error) →
  `maxsim_bridge` raises `MaxSimUnavailable` → the pipeline re-raises
  and the **FSM advances to ERROR**.  There is no WARNING-and-degrade
  state and no single-vector fallback.
- **Explicitly disabled** (`enabled = false`) → the bridge returns no
  mass; the maxsim source is simply absent from fusion — by operator
  choice, not silent degradation.

Each per-column result records a `maxsim_path` field, one of:

| `maxsim_path`      | Meaning                                                       |
|--------------------|---------------------------------------------------------------|
| `late_interaction` | ColBERT MaxSim ran cleanly; mass contributed to fusion.       |
| `explicit_disable` | Operator set `enabled = false`; source absent by choice.      |
| `unused`           | The maxsim branch was not exercised for this column.          |

(There is **no** `legacy_degraded` value — the failure path raises
rather than degrading.)

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
