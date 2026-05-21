# Curate Agent-Mediated Reference

Produce or extend an agent-mediated reference artifact for Atelier accuracy
evaluation. This is the most epistemically load-bearing operation in the
project — the resulting `agent_mediated.json` is the referee against which
all pipeline improvements are measured.

## Argument: $ARGUMENTS

Parse the argument for:
- **scope**: `table <name>`, `tables <name1,name2,...>`, `resume`, `full`, or
  `calibrate` (taxonomy-adaptive bootstrap). Default: `resume` (pick up
  where the last session left off).
- **strategy override**: `--sample-first`, `--rubric-only`, `--quick` (skip
  deep reasoning on high-confidence columns).

## Principles (non-negotiable)

These come from `project_agent_mediated_principles.md` and
`feedback_llm_mediated_reference_artifacts.md`. Read both before starting.

1. **Independence**: Your reasoning is the evidence, not Atelier's
   predictions. Atelier's output is one signal — inspect it, weigh it,
   but never rubber-stamp it. You (Opus with extended thinking, cross-column
   reasoning, persistent memory) are structurally different from Atelier's
   runtime LLM (Sonnet under bounded latency, per-column, bounded context).
   That asymmetry is what makes the reference legitimate.

2. **Procedural reproduction**: Every decision records inputs, reasoning,
   confidence, and source-agreement summary. The audit trail is mandatory
   so future sessions can re-evaluate if the taxonomy changes.

3. **Harness > model**: Deterministic cross-checks (pattern detectors,
   value-format validators, sibling-column context) are morphologically
   superior to "thinking harder." Use them.

4. **Hierarchical integrity**: Every node — leaf or internal — is a
   first-class tagging target. A column whose values span multiple leaf
   concepts under one parent is correctly tagged at the parent, not
   force-fit to a leaf.

## Phase 1 — Environment Setup

```python
import json, sys, os
sys.path.insert(0, "src")

from pathlib import Path

# Load the working set (assembled by scripts/build_agent_mediated.py)
ws_path = Path("build/data/agent_mediated/working_set.json")
if not ws_path.exists():
    print("ERROR: No working_set.json. Run: python scripts/build_agent_mediated.py")
    sys.exit(1)

ws = json.loads(ws_path.read_text())
vocab = ws["vocabulary"]
columns = ws["columns"]

# Load existing decisions (resume-safe)
am_path = Path("build/data/agent_mediated/agent_mediated.json")
existing = json.loads(am_path.read_text()) if am_path.exists() else {}
audit_path = Path("build/data/agent_mediated/audit.json")
existing_audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
state_path = Path("build/data/agent_mediated/review_state.json")
review_state = json.loads(state_path.read_text()) if state_path.exists() else {}

decided = set(existing.keys())
remaining = {k: v for k, v in columns.items() if k not in decided}

print(f"Vocabulary: {len(vocab)} tags")
print(f"Total columns: {len(columns)}")
print(f"Already decided: {len(decided)}")
print(f"Remaining: {len(remaining)}")
print(f"Tables complete: {sum(1 for v in review_state.values() if v.get('status') == 'complete')}/{ws['metadata']['table_count']}")
```

Report the counts and ask which scope to proceed with if the argument
didn't specify.

## Phase 2 — Strategic Sampling (for scale)

When the target set exceeds ~1000 columns, do NOT iterate linearly. Instead:

### 2a. Stratified table selection

Group remaining tables by size (column count). Sample proportionally:
- Small tables (≤10 cols): review all columns — cheap and eliminates them.
- Medium tables (11–30 cols): review all — these are the typical case.
- Large tables (31+ cols): apply column-level sampling (Phase 2b).

### 2b. Column-level sampling within large tables

For tables with 31+ columns, use a three-tier strategy:

1. **Anchors** (always review): columns where Atelier's prediction has
   belief < 0.5, conflict > 0.6, or where the xlsx and Atelier disagree.
   These are the high-value decisions.

2. **Representatives** (sample): from the remaining columns, cluster by
   predicted annotation. Within each cluster, review 1-2 columns that
   have the strongest evidence (highest belief). If the representative
   is clearly correct, propagate to the cluster. If not, expand the
   cluster review.

3. **Confirmation batch** (spot-check): for columns where all signals
   agree (xlsx, Atelier, LLM-solution all match) and confidence is high,
   batch-confirm with a quick sanity check of the evidence_text. Do not
   write individual reasoning for these — instead record
   `"reasoning": "All signals agree (Atelier={tag}, xlsx={tag}, llm={tag}); evidence_text consistent with {tag} semantics."`.

### 2c. Value-level sampling

When reviewing a column, you don't need to inspect every sample value.
Use this heuristic:
- First 3 values: establish the pattern.
- If homogeneous: done — the column has one value shape.
- If heterogeneous: inspect all available values; heterogeneity is signal
  (may indicate the column spans multiple concepts → parent-node tagging).

## Phase 3 — Rubric-Based Classification

For each column under review, apply this rubric IN ORDER. The rubric
implements hierarchical back-pressure: it starts at the most specific
(leaf) level and only retreats to a parent when positive evidence
demands it.

### Step 1: Evidence assembly

Gather all available signals for this column:
- `evidence_text` from Atelier (column name, type, sample values, cardinality)
- `xlsx.atelier` prediction (if Cat A)
- `xlsx.llm_solution` prediction (if Cat A)
- `atelier_current` prediction (belief, conflict, evidence_sources)
- Sibling columns (other columns in the same table — use the working_set
  to enumerate; sibling context resolves ambiguity)
- `hive_sample` (if Cat C — raw column metadata without Atelier signal)

### Step 2: Leaf-first hypothesis

Identify the most specific (leaf-level) tag in the vocabulary that
could apply. Consider:
- Value format/pattern (emails, dates, phone numbers, SSNs, etc.)
- Column name semantics
- Value cardinality and distribution
- Table context (what domain does this table serve?)

### Step 3: Back-pressure test (resist parent retreat)

Before promoting to a parent node, the column must FAIL the leaf test
on ALL children of that parent. The back-pressure rules:

- **Leaf holds if**: sample values are consistent with the leaf's
  definition, even if the column name is ambiguous. Values > names.
- **Parent justified if**: sample values span multiple children of the
  parent (e.g., a "contact_info" column containing both emails and
  phone numbers), OR the leaf's definition requires specificity the
  values don't support (e.g., values are generic dates that could be
  birth dates, transaction dates, or event dates).
- **Parent NOT justified if**: you're retreating because the column
  name is vague but the values clearly indicate a specific leaf.
  That's the column name being uninformative, not the classification
  being broad. Tag at the leaf; record low confidence if the name
  creates doubt.
- **Internal-node ceiling**: never retreat above the second level of
  the hierarchy unless the column genuinely represents a concept that
  abstract (e.g., "miscellaneous_data" with truly heterogeneous values).

### Step 4: Cross-column consistency

Before finalizing, check: are sibling columns in this table getting
consistent treatment? Common patterns to enforce:
- If `first_name` and `last_name` are siblings, they should both be
  tagged under the name-related subtree (not one as "name" and the
  other as "generic text").
- If a table has `created_at` and `updated_at`, both are temporal
  (don't tag one as "Transaction Date" and the other as "Event Date"
  unless there's positive evidence for the distinction).
- ID columns: surrogate keys (sequential ints) → INOS; natural keys
  (SSN, email, etc.) → the specific PII type, not "Identifier."

### Step 5: Confidence assignment

- **high**: evidence is unambiguous; values + name + sibling context
  all point to the same tag; deterministic cross-check passes.
- **medium**: most evidence points one way, but one signal is
  ambiguous or missing (e.g., opaque column name but clear values).
- **low**: genuine ambiguity; multiple plausible tags; choosing based
  on best-fit rather than certainty.
- **unsure**: flag for operator review. Record what makes it hard
  and which tags are candidates.

### Step 6: Decision record

For each column, produce:

```json
{
  "tag": "ANNOTATION_MNEMONIC",
  "confidence": "high|medium|low|unsure",
  "reasoning": "1-3 sentences: what evidence led to this tag, what alternatives were considered.",
  "sources": {
    "atelier_current": "what Atelier predicted (or null)",
    "atelier_xlsx": "what the xlsx said (or null)",
    "llm_solution": "what the comparison LLM said (or null)"
  },
  "needs_operator_review": false
}
```

Set `needs_operator_review: true` when:
- Confidence is "unsure"
- The decision disagrees with ALL available signals
- The tag is business-domain-specific and you can't determine
  organizational policy from the data alone
- The column appears to contain data that could be classified under
  multiple equally-valid non-hierarchically-related subtrees

## Phase 4 — Taxonomy-Adaptive Calibration

When curating for a NEW taxonomy (first time Atelier is deployed into
a new environment), run this calibration phase BEFORE the full review:

### 4a. Taxonomy structure analysis

```python
# Analyze the vocabulary's hierarchy to understand natural depth
depth_histogram = {}
for mnem, entry in vocab.items():
    code = entry.get("code", "")
    depth = code.count(".")
    depth_histogram[depth] = depth_histogram.get(depth, 0) + 1

for depth in sorted(depth_histogram):
    print(f"  Depth {depth}: {depth_histogram[depth]} tags")
```

### 4b. Subtree-specific depth targets

Different subtrees have different natural granularity. For each
top-level subtree (depth-1 node):
- Count how many leaves it has
- Sample 5-10 columns that Atelier tagged under this subtree
- Assess: is the leaf granularity useful for these columns, or does
  the taxonomy over-specify for this value domain?
- Record the subtree's "natural tagging depth" — the level at which
  most columns should be tagged. This becomes the back-pressure
  anchor for that subtree.

### 4c. Bootstrap calibration set

Review 20-30 columns (stratified across subtrees) with maximum
rigor (full reasoning, all signals inspected). These become the
calibration anchors:
- Establish baseline tag distributions per subtree
- Identify which subtrees have near-synonym leaves (taxonomy quality issue)
- Identify which subtrees have useful leaf distinctions
- Record patterns (e.g., "in the Financial subtree, leaf-level
  distinctions between credit card number vs. CVV vs. expiration
  are useful and supported by value patterns; but in the Temporal
  subtree, leaf-level distinctions between birth date vs. transaction
  date require context beyond the column itself")

After calibration, proceed to full review with calibrated expectations.

## Phase 5 — Persistence

`scripts/apply_review.py` automatically archives the current artifacts
to `build/data/agent_mediated/archive/<iso-date>/` before writing
updates (once per calendar day — subsequent calls that day are no-ops).
Archive contents:

```
archive/2026-05-20/
  2026-05-20_agent_mediated.json
  2026-05-20_audit.json
  2026-05-20_review_state.json
```

Without `--force`, existing entries are SKIPPED (merge, not overwrite).
With `--force`, existing entries for the target table are replaced —
but the archive preserves the prior state. This makes every update
non-destructive: the previous day's snapshot is always recoverable.

Assemble the decisions JSON per table:

```python
import json, subprocess
from pathlib import Path

decisions_payload = {
    "table_name": table_name,
    "reviewed_at": "auto",  # apply_review.py stamps UTC
    "table_notes": "Optional: patterns observed in this table.",
    "decisions": {
        col_name: {
            "tag": tag,          # vocabulary mnemonic or null
            "confidence": conf,  # high/medium/low/unsure
            "reasoning": reason, # 1-3 sentences
            "sources": sources,  # {atelier_current, atelier_xlsx, llm_solution}
            "needs_operator_review": flag
        }
        for col_name, (tag, conf, reason, sources, flag) in table_decisions.items()
    }
}

# Write to a temp file and pipe to apply_review.py
dec_path = Path(f"build/data/agent_mediated/decisions/{table_name}.json")
dec_path.parent.mkdir(parents=True, exist_ok=True)
dec_path.write_text(json.dumps(decisions_payload, indent=2))

result = subprocess.run(
    ["uv", "run", "python", "scripts/apply_review.py",
     "--decisions-file", str(dec_path), "--allow-partial"],
    capture_output=True, text=True
)
print(result.stderr)  # progress output
```

Persist after EACH table (not at the end). This makes the process
resume-safe: if the session is interrupted, completed tables are saved.

## Phase 6 — Quality Assurance

After completing a review batch:

1. **Coverage check**: How many columns decided vs. total? How many
   tables complete?

2. **Distribution sanity**: Does the tag distribution look reasonable?
   A vocabulary with 295 tags should show at least 30-40 distinct tags
   in use across 920 columns. If one tag dominates (>20% of columns),
   investigate whether the rubric is over-generalizing.

3. **Confidence distribution**: If >50% of decisions are "high"
   confidence, that's healthy. If >30% are "unsure", the taxonomy
   may have structural issues the operator should know about.

4. **Disagreement summary**: List the columns where the curation
   decision disagreed with ALL available signals. These are either
   the most valuable corrections or potential errors — flag them
   for the operator.

5. **Memory persistence**: Save any taxonomy-level patterns discovered
   during review (e.g., "SYSSTATE and INOS are near-synonyms that
   confuse both Atelier and human reviewers") as project memories
   so future sessions benefit.

```python
# Quick distribution check
from collections import Counter
tag_counts = Counter(v for v in existing.values() if v)
print(f"Distinct tags used: {len(tag_counts)}/{len(vocab)}")
print(f"Top 10 tags:")
for tag, count in tag_counts.most_common(10):
    label = vocab.get(tag, {}).get("label", "?")
    print(f"  {tag} ({label}): {count}")

confidence_counts = Counter(
    v.get("confidence") for v in existing_audit.values()
)
print(f"\nConfidence distribution: {dict(confidence_counts)}")

needs_review = [k for k, v in existing_audit.items()
                if v.get("needs_operator_review")]
print(f"Flagged for operator review: {len(needs_review)}")
```

## Scaling to Millions of Tables

When Atelier is deployed against a data lake with millions of tables,
exhaustive per-column curation is impossible. The strategy shifts:

### Embedding-based representative sampling

1. Run Atelier's classify pipeline on a Monte Carlo sample (the
   pipeline's existing `monte_carlo.py` handles this).
2. From the classified results, cluster columns by their embedding
   vectors (the `embedding_text` field captures the column's semantic
   signature).
3. Select cluster centroids as representative columns for curation.
4. Curate the representatives with full rigor (Phase 3 rubric).
5. Propagate curated labels to cluster members with a confidence
   penalty (medium instead of high).

### Rubric-as-policy

Instead of curating every column, curate the RUBRIC:
- Define per-subtree tagging rules as deterministic decision trees
  (if value matches pattern X → tag Y; if column name contains Z
  → tag W).
- The rubric becomes the reproducible artifact (per the feedback
  memory's "procedural machinery" requirement).
- LLM curation handles the residual: columns that don't match any
  deterministic rule.

### Incremental refinement

- Start with the highest-confidence subset (all signals agree).
- Expand to disagreement cases.
- Expand to uncovered subtrees.
- Each batch improves the reference while maintaining resume-safety.

## Anti-patterns (DO NOT)

- Do NOT copy Atelier's predictions into the reference without
  independent reasoning. That measures self-consistency, not accuracy.
- Do NOT tag everything at the parent level because "it's safer."
  That destroys the reference's ability to measure leaf-level accuracy.
- Do NOT skip the audit trail. A reference without reasoning is a
  single-shot judgment with no procedural reproduction path.
- Do NOT review columns in isolation when sibling context is available.
  Cross-column consistency is a free deterministic cross-check.
- Do NOT persist decisions for an entire batch at the end. Persist
  per-table for resume-safety.
