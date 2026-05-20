# Bootstrap Environment

Unified agent-mediated bootstrap for a new Atelier deployment. Consolidates
annotation enrichment (Qdrant payloads for ColBERT late-interaction) and
reference curation (`agent_mediated.json` for accuracy evaluation) into a
single pass where annotation-level reasoning feeds both directions.

Both operations require deep reasoning about what each taxonomy node means,
what values look like, how siblings discriminate, and where leaf/parent
boundaries sit. Running them in one pass means the agent reasons once per
annotation — enrichment payloads and curation decisions share the same
understanding.

## Argument: $ARGUMENTS

Parse the argument for:
- **scope**: `full`, `resume` (default), `enrich-only`, `curate-only`,
  `report`, `table <name>`, `tables <name1,name2,...>`
- **source override**: `--source <connection/database>` (overrides
  auto-detection from the most recent real pipeline run)

| Scope | Phases | Use case |
|-------|--------|----------|
| `full` | 1-6 | New environment, start from scratch (resume-safe) |
| `resume` | 1-6 | Pick up where the last invocation stopped |
| `enrich-only` | 1-3 | Prepare enrichment before a curation session |
| `curate-only` | 4 | Enrichment already at 100%; curate columns only |
| `train-svm` | 5 | Train SVM after curation (requires Phases 1-4 complete) |
| `report` | 6 | Read-only summary of current bootstrap state |
| `table <name>` | 4 | Curate a specific table (with 100% enrichment pre-check) |

## Principles (non-negotiable)

Read `project_agent_mediated_principles.md` and
`feedback_llm_mediated_reference_artifacts.md` before starting.

1. **Independence**: Your reasoning is the evidence, not Atelier's
   predictions. Atelier's output is one signal — inspect it, weigh it,
   but never rubber-stamp it.

2. **Procedural reproduction**: Every decision records inputs, reasoning,
   confidence, and source-agreement summary. Audit trail is mandatory.

3. **Harness > model**: Deterministic cross-checks (pattern detectors,
   value-format validators, sibling-column context) are morphologically
   superior to "thinking harder." Use them.

4. **Hierarchical integrity**: Every node — leaf or internal — is a
   first-class tagging target. A column spanning multiple leaf concepts
   under one parent is correctly tagged at the parent, not force-fit
   to a leaf.

5. **Dynamic annotations**: Never hardcode annotation counts. Always
   say "100% coverage", not "296/296". Cardinality changes whenever the
   operator selects a different taxonomy.

---

## Phase 1 — Taxonomy Analysis + Source Detection

### 1a. Auto-detect target database

```python
import json, os, sys
sys.path.insert(0, "src")
from pathlib import Path

SYNTHETIC_SOURCES = {"ootb-sample", "synthetic", "meta-tagging"}

def detect_source():
    """Find the most recent real pipeline run's source_id."""
    results = Path("build/results")
    if not results.exists():
        return None, None
    dirs = sorted(results.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs:
        snap = d / "settings_snapshot.json"
        cls = d / "classifications.json"
        if not snap.exists():
            continue
        s = json.loads(snap.read_text())
        source_id = s.get("source_id", "")
        if source_id in SYNTHETIC_SOURCES or not source_id:
            continue
        # Verify the run actually classified something
        if cls.exists():
            data = json.loads(cls.read_text())
            n = len(data) if isinstance(data, list) else len(data.get("classifications", []))
            if n > 0:
                parts = source_id.split("/", 1)
                connection = parts[0]
                database = parts[1] if len(parts) > 1 else "default"
                return connection, database
    return None, None

connection, database = detect_source()
if connection:
    print(f"Detected source: {connection}/{database}")
else:
    print("No real pipeline run found. Specify --source <connection/database>")
```

If `--source` was provided in the argument, use that instead.

### 1b. Load taxonomy and analyze structure

```python
# Load vocabulary (from working_set if available, otherwise from enrichment source)
ws_path = Path("build/data/agent_mediated/working_set.json")
if ws_path.exists():
    ws = json.loads(ws_path.read_text())
    vocab = ws["vocabulary"]
    columns = ws.get("columns", {})
else:
    vocab = {}
    columns = {}
    print("No working_set.json — will need to assemble after enrichment")

# Taxonomy depth histogram
depth_histogram = {}
for mnem, entry in vocab.items():
    code = entry.get("code", "")
    depth = code.count(".")
    depth_histogram[depth] = depth_histogram.get(depth, 0) + 1

print(f"\nTaxonomy: {len(vocab)} annotations")
for depth in sorted(depth_histogram):
    print(f"  Depth {depth}: {depth_histogram[depth]} tags")
```

### 1c. Assess current state

```python
# Enrichment state
am_path = Path("build/data/agent_mediated/agent_mediated.json")
existing = json.loads(am_path.read_text()) if am_path.exists() else {}
state_path = Path("build/data/agent_mediated/review_state.json")
review_state = json.loads(state_path.read_text()) if state_path.exists() else {}

decided = set(existing.keys())
remaining = {k: v for k, v in columns.items() if k not in decided}

print(f"\nCuration: {len(decided)}/{len(columns)} columns decided")
if columns:
    tables_complete = sum(1 for v in review_state.values() if v.get("status") == "complete")
    tables_total = len({c["table_name"] for c in columns.values()})
    print(f"Tables: {tables_complete}/{tables_total} complete")

# Check enrichment coverage (requires Qdrant access)
print("\nChecking enrichment coverage...")
```

Report the full state and proceed to the appropriate phase based on scope.

---

## Phase 2 — Enrichment Verification & Repair

**Hard precondition**: 100% enrichment coverage before curation begins.
ColBERT cannot score columns against annotations not in Qdrant.

### 2a. Detect Qdrant access

```python
qdrant_reachable = False
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(url=os.environ.get("ATELIER_QDRANT_URL", "http://127.0.0.1:6333"), timeout=5)
    client.get_collections()
    qdrant_reachable = True
    print("Qdrant: reachable (Case A — direct access)")
except Exception as e:
    print(f"Qdrant: not reachable ({e})")
    print("Case B — will instruct operator to run enrichment on Application pod")
```

### 2b. Coverage check

**Case A (direct access)**:
```python
from atelier.enrichment.qdrant_writer import collection_name_for

taxonomy_id = "default"  # or from detected source
augmentation_version = "v1"
collection = collection_name_for(taxonomy_id, augmentation_version)

# Scroll collection to get all enriched codes
enriched_codes = set()
offset = None
while True:
    result = client.scroll(collection_name=collection, limit=100, offset=offset, with_payload=True)
    points, offset = result
    for p in points:
        code = (p.payload or {}).get("mnemonic") or (p.payload or {}).get("code")
        if code:
            enriched_codes.add(code)
    if offset is None:
        break

total_annotations = len(vocab)
enriched_count = len(enriched_codes)
missing = set(vocab.keys()) - enriched_codes
coverage_pct = (enriched_count / total_annotations * 100) if total_annotations else 0

print(f"Enrichment: {enriched_count}/{total_annotations} ({coverage_pct:.1f}%)")
if missing:
    print(f"Missing ({len(missing)}): {sorted(missing)[:10]}...")
```

**Case B (no direct access)**:
```
Instruct the operator:

  On the Application pod's Web Terminal, run:

    uv run python scripts/enrich_annotations.py \
      --from-connection {connection} \
      --database {database} \
      --taxonomy-id default \
      --augmentation-version v1

  Then check the checkpoint file:
    tail -5 build/enrichment/annotations_default_v1-*.jsonl

  Re-invoke /bootstrap-environment resume when enrichment completes.
```

### 2c. Repair missing annotations

If coverage < 100%:

**Case A**: Run enrichment for missing codes programmatically:
```python
from atelier.enrichment.loop import run_enrichment, EnrichmentLoopConfig
# ... configure and run for missing rows only
```

**Case B**: Provide the operator with the specific missing codes and
the enrichment command to re-run with `--no-skip-cache` targeting
those codes.

**Gate**: Do NOT proceed to Phase 3 until coverage is 100%. Report
the gap and stop.

---

## Phase 3 — Enrichment Quality Review

The 6-check verifier suite catches structural issues. This phase catches
**semantic quality** issues that require the same domain reasoning the
curation rubric exercises.

### 3a. Quality criteria (beyond the verifier)

For each enrichment payload, assess:

| Criterion | What to check | Red flag |
|-----------|---------------|----------|
| **Prototype realism** | Would these values appear in a real data warehouse column? | Synthetic-looking, overly clean, or format-only values |
| **Anti-example coverage** | Do anti_examples capture the genuinely confusable siblings? | Missing obvious confusables (e.g., EMAIL enrichment lacks phone-number anti-example) |
| **Name hint discrimination** | Would these column names distinguish this tag from siblings? | Generic names that apply to the parent, not the leaf |
| **Pattern tightness** | Do value_patterns discriminate, or are they vacuous? | `".+"` or `".*"` patterns that match everything |
| **Parent/leaf boundary** | For parents: broader than any child's prototypes? For leaves: specific enough to distinguish from siblings? | Parent prototypes identical to one child's; leaf prototypes as broad as parent's |
| **Sibling consistency** | Do sibling annotations have complementary (non-overlapping) prototypes and anti-examples? | Two siblings with near-identical prototype_values |

### 3b. Subtree-by-subtree sweep

Review enrichment payloads grouped by top-level subtree (depth-1 node).
Within each subtree:

1. Load all enrichment payloads for the subtree's nodes.
2. Check the parent/leaf boundary: does the parent's enrichment correctly
   encompass its children without being identical to any one child?
3. Check sibling discrimination: do leaves under the same parent have
   complementary prototype_values and cross-listed anti_examples?
4. Flag quality issues with structured notes.

### 3c. Feedback from prior curation (if resuming)

Load `build/enrichment/curation_feedback.json` (if it exists from a
prior Phase 4 run). Each entry identifies an annotation whose
enrichment payload was misleading during curation. Prioritize these
for quality review.

### 3d. Persist quality review

Write results to `build/enrichment/quality_review.json`:
```json
{
  "reviewed_at": "2026-05-20T...",
  "taxonomy_id": "default",
  "annotations_reviewed": 295,
  "quality_ok": 280,
  "flagged": 15,
  "issues": [
    {
      "code": "ANNOTATION_MNEMONIC",
      "issue_type": "weak_prototypes|missing_anti_example|vacuous_pattern|...",
      "detail": "Description of the quality issue",
      "regenerated": false
    }
  ]
}
```

---

## Phase 4 — Reference Curation (enrichment-grounded)

With 100% enrichment coverage guaranteed and quality reviewed, curate
column classifications. This phase is the curation rubric from
`/curate-agent-mediated` with two enhancements: enrichment grounding
and the feedback channel.

### 4a. Setup (same as curation Phase 1)

```python
ws = json.loads(Path("build/data/agent_mediated/working_set.json").read_text())
vocab = ws["vocabulary"]
columns = ws["columns"]
existing = json.loads(Path("build/data/agent_mediated/agent_mediated.json").read_text()) \
    if Path("build/data/agent_mediated/agent_mediated.json").exists() else {}
decided = set(existing.keys())
remaining = {k: v for k, v in columns.items() if k not in decided}
print(f"Remaining: {len(remaining)} columns")
```

### 4b. Strategic sampling (for scale)

When the target set exceeds ~1000 columns, apply stratified sampling:

- **Small tables (≤10 cols)**: review all — cheap and eliminates them.
- **Medium tables (11-30 cols)**: review all — typical case.
- **Large tables (31+ cols)**: three-tier column sampling:
  - **Anchors**: belief < 0.5, conflict > 0.6, or signal disagreement
  - **Representatives**: cluster by predicted annotation, review 1-2 per cluster
  - **Confirmation batch**: all signals agree + high confidence → batch-confirm

### 4c. Enrichment-grounded rubric

For each column under review, apply IN ORDER:

**Step 1: Evidence assembly (enhanced)**

Gather all signals PLUS enrichment payloads for candidate annotations:
- `evidence_text` from Atelier (column name, type, sample values, cardinality)
- `xlsx.atelier` / `xlsx.llm_solution` predictions (if Cat A)
- `atelier_current` prediction (belief, conflict, evidence_sources)
- Sibling columns in the same table
- **NEW**: enrichment payloads for the top candidate annotations
  (prototype_values, value_patterns, anti_examples, name_hints)

**Step 2: Leaf-first hypothesis (enrichment-grounded)**

Instead of building a mental model from scratch, consult the enrichment
payload for each candidate annotation:
- Do the column's sample values match the annotation's `value_patterns`?
- Are the column's values consistent with the annotation's `prototype_values`?
- Do the column's values appear in any annotation's `anti_examples`?
  (If so, that annotation is NOT the right tag.)
- Do the column's names match any annotation's `name_hints`?

**Step 3: Back-pressure test (resist parent retreat)**

Before promoting to a parent node, the column must FAIL the leaf test
on ALL children of that parent:

- **Leaf holds if**: sample values match the leaf annotation's
  `value_patterns` and are consistent with its `prototype_values`,
  even if the column name is ambiguous. Values > names.
- **Parent justified if**: sample values span multiple children's
  prototype ranges, OR no child's patterns match.
- **Parent NOT justified if**: retreating because the column name
  is vague but values clearly match a specific leaf's prototypes.
- **Internal-node ceiling**: never retreat above depth-2 unless the
  column genuinely represents a concept that abstract.

**Step 4: Cross-column consistency**

Check sibling columns in the same table for consistent treatment.

**Step 5: Confidence assignment**

- **high**: values match enrichment prototypes/patterns unambiguously
- **medium**: most evidence aligns, one signal ambiguous or missing
- **low**: genuine ambiguity; multiple plausible tags
- **unsure**: flag for operator review

**Step 6: Decision record + enrichment feedback**

For each column, produce two outputs:

The curation decision (persisted via `apply_review.py`):
```json
{
  "tag": "ANNOTATION_MNEMONIC",
  "confidence": "high|medium|low|unsure",
  "reasoning": "1-3 sentences",
  "sources": {
    "atelier_current": "...",
    "atelier_xlsx": "...",
    "llm_solution": "..."
  },
  "needs_operator_review": false
}
```

The enrichment feedback (accumulated in `build/enrichment/curation_feedback.json`):
- When the column's values DON'T match the winning annotation's
  prototypes → note that the enrichment prototypes may be too narrow
- When two annotations are genuinely confusable but neither lists the
  other as an anti_example → note the missing cross-reference
- When an annotation's name_hints fail to predict real column names
  observed during curation → note the gap

### 4d. Persistence

Use `scripts/apply_review.py` per table (resume-safe, with archive):
```python
import json, subprocess
from pathlib import Path

dec_path = Path(f"build/data/agent_mediated/decisions/{table_name}.json")
dec_path.parent.mkdir(parents=True, exist_ok=True)
dec_path.write_text(json.dumps(decisions_payload, indent=2))

result = subprocess.run(
    ["uv", "run", "python", "scripts/apply_review.py",
     "--decisions-file", str(dec_path), "--allow-partial"],
    capture_output=True, text=True
)
print(result.stderr)
```

Persist after EACH table. Do not batch.

---

## Phase 5 — SVM Training (agent-mediated)

With 100% enrichment coverage and the reference curated, train (or retrain)
the SVM evidence source using enrichment-derived generators. This follows
the same methodology as `/train-svm` but with two bootstrap-specific
advantages: enrichment payloads are already verified (Phase 3) and the
reference is freshly curated (Phase 4).

### 5a. Check prerequisites

```python
# Verify we have what SVM training needs
am = json.loads(Path("build/data/agent_mediated/agent_mediated.json").read_text()) \
    if Path("build/data/agent_mediated/agent_mediated.json").exists() else {}
if len(am) < 50:
    print(f"Only {len(am)} curated columns — need ≥50 for evaluation. Skipping SVM training.")
    # Skip to Phase 6
else:
    print(f"Reference: {len(am)} curated columns — sufficient for evaluation")
```

### 5b. Extract enrichment payloads

If Phase 3 already ran, payloads are in Qdrant and have been quality-reviewed.
Use `enrichment_loader.load_enrichment_payloads(cfg=cfg)` to load from Qdrant.
Persist to `build/data/svm_training/enrichment_payloads.json` (resume
checkpoint — skip if exists with matching vocab hash).

### 5c. Generate corpus from enrichment payloads

Bootstrap guarantees 100% enrichment. The registry layers ICE hand-coded
generators (matched via enrichment metadata) > template generators (from
prototype_values) > inferred generators (from category metadata):

```python
from atelier.classify.taxonomy import load_annotations_from_json
from atelier.classify.synth import generate_user_taxonomy_corpus

cache_path = Path("build/cache/annotations/default.json")
user_category_set = load_annotations_from_json(cache_path, hierarchical=True)

corpus_dir = Path("build/data/svm_training/corpus")
results, coverage = generate_user_taxonomy_corpus(
    user_category_set, payloads, corpus_dir,
    seed=42, variants_per_category=30,
)

from collections import Counter
print(f"Generator coverage: {Counter(coverage.values())}")
```

Agent reviews a sample of generated columns per subtree for semantic
quality (same review criteria as Phase 3 enrichment quality review,
but applied to generated values instead of prototypes).

### 5d. Train and evaluate

```python
from atelier.classify.ml_train import train_svm

candidate_path = Path("build/data/svm_training/candidate_svm")
train_svm(corpus_dir, candidate_path)
```

Score candidate against the Phase 4 reference. Compare vs. incumbent
SVM (if one exists). Gate criterion: candidate ≥ incumbent on exact
accuracy. On pass, promote to `build/models/svm.pkl` and invalidate
per-vocab SVM cache. On fail, report diagnosis but do not block the
bootstrap — the existing SVM (or no SVM) remains.

See `/train-svm` skill for the full evaluation procedure, confusion
matrix inspection, and promotion protocol.

### 5e. Persist

Training artifacts go to `build/data/svm_training/`:
- `enrichment_payloads.json` — extracted Qdrant payloads
- `corpus/` — synth CSVs + reference_labels.json
- `generator_review.json` — agent's quality assessment
- `candidate_svm.pkl` — trained candidate
- `evaluation.json` — A/B results + gate decision
- `promotion_log.json` — append-only promotion history

---

## Phase 6 — Coverage Report

```python
import json
from pathlib import Path
from collections import Counter

# Enrichment
qr_path = Path("build/enrichment/quality_review.json")
qr = json.loads(qr_path.read_text()) if qr_path.exists() else {}

# Curation
am = json.loads(Path("build/data/agent_mediated/agent_mediated.json").read_text()) \
    if Path("build/data/agent_mediated/agent_mediated.json").exists() else {}
audit = json.loads(Path("build/data/agent_mediated/audit.json").read_text()) \
    if Path("build/data/agent_mediated/audit.json").exists() else {}
state = json.loads(Path("build/data/agent_mediated/review_state.json").read_text()) \
    if Path("build/data/agent_mediated/review_state.json").exists() else {}
ws = json.loads(Path("build/data/agent_mediated/working_set.json").read_text()) \
    if Path("build/data/agent_mediated/working_set.json").exists() else {"columns": {}, "vocabulary": {}}

vocab = ws["vocabulary"]
total_cols = len(ws["columns"])
total_tables = len({c["table_name"] for c in ws["columns"].values()}) if ws["columns"] else 0

print("=" * 50)
print("BOOTSTRAP ENVIRONMENT STATUS")
print("=" * 50)

# Enrichment coverage
if qr:
    print(f"\nEnrichment:")
    print(f"  Reviewed: {qr.get('annotations_reviewed', '?')}")
    print(f"  Quality OK: {qr.get('quality_ok', '?')}")
    print(f"  Flagged: {qr.get('flagged', '?')}")

# Curation coverage
print(f"\nCuration:")
print(f"  Columns decided: {len(am)}/{total_cols}")
tables_complete = sum(1 for v in state.values() if v.get("status") == "complete")
print(f"  Tables complete: {tables_complete}/{total_tables}")

# Tag distribution
tag_counts = Counter(v for v in am.values() if v)
print(f"  Distinct tags used: {len(tag_counts)}/{len(vocab)}")
print(f"  Top 5 tags:")
for tag, count in tag_counts.most_common(5):
    label = vocab.get(tag, {}).get("label", "?")
    print(f"    {tag} ({label}): {count}")

# Confidence
conf_counts = Counter(v.get("confidence") for v in audit.values())
print(f"\n  Confidence: {dict(conf_counts)}")

# Operator review flags
needs_review = [k for k, v in audit.items() if v.get("needs_operator_review")]
print(f"  Flagged for operator review: {len(needs_review)}")

# Curation feedback
cf_path = Path("build/enrichment/curation_feedback.json")
if cf_path.exists():
    cf = json.loads(cf_path.read_text())
    print(f"\nEnrichment feedback from curation: {len(cf.get('entries', []))} notes")

# SVM training status
svm_eval_path = Path("build/data/svm_training/evaluation.json")
if svm_eval_path.exists():
    svm_eval = json.loads(svm_eval_path.read_text())
    cand = svm_eval.get("candidate", {})
    print(f"\nSVM Training:")
    print(f"  Training path: {cand.get('training_path', '?')}")
    print(f"  Training classes: {cand.get('training_classes', '?')}")
    print(f"  Exact accuracy: {cand.get('exact_accuracy', '?')}")
    print(f"  Hierarchical accuracy: {cand.get('hierarchical_accuracy', '?')}")
    print(f"  Gate: {svm_eval.get('gate_result', '?')}")
    inc = svm_eval.get("incumbent", {})
    if inc and inc.get("exact_accuracy") is not None:
        delta = (cand.get("exact_accuracy", 0) or 0) - (inc.get("exact_accuracy", 0) or 0)
        print(f"  vs incumbent: {delta:+.3f}")
else:
    print(f"\nSVM Training: not yet run")

# What's left
remaining = total_cols - len(am)
if remaining > 0:
    print(f"\n  Remaining: {remaining} columns to curate")
else:
    print(f"\n  COMPLETE: all columns curated, SVM {'trained' if svm_eval_path.exists() else 'pending'}")
```

---

## Consolidation Points (where reasoning feeds both directions)

| Point | Direction | Mechanism |
|-------|-----------|-----------|
| **A** | Enrichment → Curation | Tight prototypes/patterns → higher curation confidence; weak enrichment → lower confidence + flag |
| **B** | Curation → Enrichment | Column values consistently mismatch an annotation's prototypes → regeneration needed |
| **C** | Bidirectional | Taxonomy calibration (depth/breadth) informs both enrichment quality review and curation back-pressure |
| **D** | Curation → Enrichment | Genuinely confusable annotations discovered during curation → missing anti_examples |

---

## Anti-patterns (DO NOT)

- Do NOT copy Atelier's predictions into the reference without
  independent reasoning. That measures self-consistency, not accuracy.
- Do NOT proceed to curation (Phase 4) without 100% enrichment coverage.
  ColBERT scoring on partial enrichment produces degraded evidence.
- Do NOT tag everything at the parent level because "it's safer."
  That destroys the reference's ability to measure leaf-level accuracy.
- Do NOT skip the audit trail. A reference without reasoning is a
  single-shot judgment with no procedural reproduction path.
- Do NOT review columns in isolation when sibling context is available.
  Cross-column consistency is a free deterministic cross-check.
- Do NOT persist decisions for an entire batch at the end. Persist
  per-table for resume-safety.
- Do NOT hardcode annotation counts ("296/296"). Always use percentage
  or relative language ("100% coverage"). Cardinality is dynamic.
- Do NOT skip Phase 3 (enrichment quality review) even when coverage
  is at 100%. Coverage is necessary but not sufficient — the payloads
  must also be semantically correct.
