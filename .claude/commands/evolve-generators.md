# evolve-generators — author Callable[[random.Random], str] generators for taxonomy gaps

## Purpose

Phase B of the SHAP-priority-guided synthetic corpus expansion plan.
For every gap node in the taxonomy (codes without a hand-coded generator
or with low-diversity coverage), produce:

1. **A generator function** `Callable[[random.Random], str]` returning
   distributionally-realistic synthetic values for that category.
2. **A list of 15-20 column name variants** that real-world columns of
   this class would carry in production schemas.
3. **5-10 table-context templates**, each specifying plausible siblings
   and table names this category typically co-occurs with.

These artifacts feed the corpus generator (Phase C) which produces
~200 synthetic rows per node across diverse name/value/sibling/table
contexts.

## Why this matters

The factorized NHSVM head currently generalizes to 61.25% on the
held-out test (vs ~80% target).  The gap is data scarcity — only ~5
training examples per class.  Per-slice SHAP attribution says
`sample_values` dominates at 4× the next slice, so value-distribution
realism is the highest-leverage axis; secondary axes are
`column_name`, `column_type`, and contextual signals (`sibling_context`,
`source_table`).

## Inputs you have

- `build/svm_corpus_v2/coverage_audit.json` — the gap list with
  priority ordering: missing → inferred_only → template_only →
  low_diversity_handcoded.
- `build/data/svm_training/enrichment_payloads.json` — per-mnemonic
  enrichment metadata: `label`, `description`, `name_hints`,
  `prototype_values`, `value_patterns`, `anti_examples`.
- `src/atelier/classify/synth_generators.py` — ~70 hand-coded ICE.*
  reference generators to imitate the style of (single-family
  diversified strings via `rng.choice` / `rng.randint` / format mixing).
- `build/lib/generated/generators_v0.py` — 4 prior Opus-emitted
  generators for stylistic reference.

## Output contract — per gap code

For each gap code, write a JSON file at:
`build/svm_corpus_v2/proposals/<sanitized_code>.json`

(Sanitize by replacing `.` with `_`, so `1.1.1.9.3.1` →
`1_1_1_9_3_1.json`.)

JSON schema (strict):

```json
{
  "code": "1.1.1.9.3.1",
  "mnemonic": "EMAIL",
  "label": "Email Address",
  "rationale": "1-2 sentences explaining the shape your generator targets",
  "generators": [
    {
      "function_name": "generate_email_human",
      "code": "def generate_email_human(rng):\n    ...\n    return value\n",
      "variant_label": "human"
    },
    {
      "function_name": "generate_email_terse",
      "code": "def generate_email_terse(rng):\n    ...\n    return value\n",
      "variant_label": "terse"
    }
  ],
  "name_variants": [
    "email_address", "email", "e_mail", "contact_email", "primary_email",
    "user_email", "email_addr", "emailaddr", "email1", "personal_email",
    "work_email", "billing_email", "notify_email", "col_3a", "dat_e1"
  ],
  "table_context_templates": [
    {
      "table_names": ["users", "customers", "subscribers"],
      "siblings": ["user_id", "first_name", "last_name", "created_at"]
    },
    {
      "table_names": ["contacts", "address_book"],
      "siblings": ["contact_id", "phone", "address", "company"]
    }
  ]
}
```

## Constraints — strictly enforced

### Generator code constraints

- **Only stdlib imports allowed**: `random`, `string`, `re`. No file
  I/O, no network, no `eval`/`exec`/`compile`/`open`/`__import__`.
- **Always return a non-empty string** — never `None`, never `""`.
- **Use the `rng` parameter for all randomness**; no globals, no
  module-level state.
- **5-25 line body** per function.
- **Variety in output**: 50 successive calls must produce ≥30 distinct
  strings (≥0.6 distinct ratio).  Different lengths, prefixes,
  formats — not 100 near-identical strings.
- **Per code, 2-4 generator variants** with `variant_label`s like
  `human`, `terse`, `opaque`, `hash`, `formal`, `abbreviated`,
  `prefixed`, `suffixed`.  Each variant covers a different
  sub-distribution of how that category appears in the wild.

### Name variants constraints

- **15-20 names per code**.
- snake_case identifiers (lowercase, underscores, alphanumeric).
- No table prefixes.  Just the bare column name.
- **Mix: ~75% semantic, ~25% opaque** (e.g., `val_42`, `col_07`,
  `dat_3a`, `fld_5b`).  Per the SHAP analysis, the column_name slice
  contributes meaningfully (|attr|=0.23), so name vocabulary matters.
- Vary length and structure: single words, two-word combos, three-word
  combos with abbreviations.

### Table context templates

- **5-10 templates per code**.
- Each has `table_names` (3-5 plausible synthetic table names this
  column would appear in) and `siblings` (4-8 plausible column names
  that would co-occur as table neighbors).
- Mix domains: a `payment_amount` column should appear in `orders`,
  `transactions`, `invoices` — not just one.

## Execution

For each gap code, in priority order from coverage_audit.json's
`gap_list`:

1. **Read** the gap entry + look up enrichment payload by mnemonic.
2. **Author** the generators + name_variants + table_context_templates.
3. **Self-validate**: spot-check by running the generator 5-10 times
   in your head, ensuring distinct outputs and shape sanity.
4. **Write** the proposal JSON to
   `build/svm_corpus_v2/proposals/<sanitized_code>.json`.
5. **Move on** to the next code — do NOT re-validate previous codes,
   do NOT report aggregate status until the end.

The calling wrapper (`scripts/run_evolve_generators_sdk.py`) handles
AST validation, subprocess sandbox execution, output-shape gates, and
persistence to `build/lib/generated/generators_v1.py`.  You do not need
to do those steps — focus on producing high-quality proposals.

## Resume safety

If `build/svm_corpus_v2/proposals/<sanitized_code>.json` already
exists for a gap code, **skip it** — assume a prior agent run already
authored it.  Process only codes without an existing proposal file.

## Domain-adaptation mode (when refinement_targets.json is present)

If `build/svm_corpus_v2/refinement_targets.json` exists, read it.  Each
target entry has `{code, accuracy, neighbors: [{pred_code, count}]}` —
the `neighbors` are categories the prior generator's output was
indistinguishable from at the held-out test boundary.

For each refinement target whose proposal file has been cleared by the
refinement orchestrator:

- This is a **domain adaptation** task: re-author the generator so its
  output occupies a value-shape region clearly separable from the
  listed `neighbors`.  Focus on the SOLUTION (what distinguishes this
  category's true values) rather than the PROBLEM (what got confused).
- Read the existing entries for each neighbor code in
  `build/lib/generated/generators_v1.py` (or `synth_generators.py` if
  hand-coded ICE) to understand the adjacent value-shape regions.
- Choose a format family, prefix vocabulary, delimiter pattern, or
  length distribution that occupies a different region of value-shape
  space than the neighbors.
- Lead the proposal's `rationale` with: "DOMAIN-ADAPTED against
  neighbors <neighbor_codes>: <which value-shape region this
  category occupies that the neighbors do not>".

Refinement targets take priority over fresh gap codes.  Process them
first, then move to remaining gaps if budget allows.

## Stopping condition

Stop when:
- All gap codes (per coverage_audit.json's gap_list) have a proposal
  file, OR
- You have authored proposals for at least 30 codes in this session
  (incremental progress; the wrapper will re-invoke you if more
  remain), OR
- You encounter a structural blocker (missing input files, etc.).

Emit a final summary message listing the codes you processed and any
that you skipped (with reasons).

## Stylistic guidance from existing generators

Skim `src/atelier/classify/synth_generators.py` for examples:

- `gen_email` (lines 218-241) — combines first/last names, separators,
  digit suffixes, domain pools; multi-format output via `rng.choice`
  branches.
- `gen_phone` (lines 244-269) — multiple format families (US, EU,
  international) selected by `rng.choice` of format templates.
- `gen_ssn`, `gen_iban`, `gen_credit_card` — structured-ID generators
  with checksum-aware structure.

The pattern: 1) pick a format-family via `rng.choice`, 2) fill in
slots from value pools, 3) optionally add prefix/suffix/separator
variation.  Apply this pattern to whatever distribution your gap code
covers.

## Constraints on what NOT to produce

- Do NOT verbatim copy `prototype_values` from the enrichment payload
  into your generator output.  Use them as shape guidance only.
- Do NOT match the `anti_examples` from the payload — those are
  category-side negatives that should not be representative of the
  positive class.
- Do NOT produce single-format generators (e.g., always returning the
  same format string) — the diversity gate (n=50, ≥0.6 distinct ratio)
  will reject them.
- Do NOT depend on external data files, network calls, or any non-
  stdlib library.
