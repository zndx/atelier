# Train SVM

Agent-mediated SVM training for Atelier's classification pipeline. Generates
an enrichment-derived synthetic corpus and trains the SVM evidence source
with full taxonomy coverage.

The enrichment payloads (prototype_values, value_patterns, name_hints,
anti_examples) are generator specifications — the same annotation-level
reasoning that powers enrichment and curation drives corpus generation
here.  Enrichment data is always required — sourced from Qdrant (live,
source of truth) or a JSON export (for offline/CI).

## Argument: $ARGUMENTS

Parse the argument for:
- **scope**: (default, no arg) full training run, `evaluate`
  (score incumbent only), `resume`, `report`
- **options**: `--variants N` (variants per category, default 30),
  `--seed N` (RNG seed, default 42),
  `--payloads <path>` (JSON export of enrichment payloads — use when
  Qdrant is unavailable)

| Scope | Phases | Use case |
|-------|--------|----------|
| (default) | 1-6 | Full training: generate corpus, train, evaluate, promote |
| `evaluate` | 1, 5 | Score incumbent SVM against reference (no training) |
| `resume` | 1-6 | Pick up from checkpoint |
| `report` | read-only | Show training state + evaluation results |

Enrichment coverage must be 100%. If not, halt with an actionable
message directing the operator to complete enrichment first.

## Principles

1. **Synthetic only**: The corpus is generated from enrichment specs,
   never from real column data. Enrichment prototypes are generator
   specifications, not training data.

2. **Evaluation gated**: A trained model is not automatically better.
   The candidate must meet or exceed the incumbent's accuracy against
   the agent-mediated reference before promotion.

3. **DST independence preserved**: SVM features remain TF-IDF sparse
   lexical (structurally independent of dense ColBERT embeddings).
   The shared upstream (enrichment LLM) is unchanged. Discount stays
   at 0.22.

4. **Resume-safe**: Corpus persists to disk. Training can restart
   from saved corpus without regeneration.

5. **Dynamic annotations**: Never hardcode counts. Always "100%
   coverage", not "295/295".

6. **Enrichment required**: The SVM must only emit user-provided
   terminology. Without enrichment data, training cannot proceed.
   Qdrant is the source of truth; JSON export is the offline fallback.

---

## Phase 1 — Precondition Verification

```python
import json, os, sys
sys.path.insert(0, "src")
from pathlib import Path

# Load vocabulary
ws_path = Path("build/data/agent_mediated/working_set.json")
if not ws_path.exists():
    print("ERROR: No working_set.json. Run /bootstrap-environment first.")
    sys.exit(1)

ws = json.loads(ws_path.read_text())
vocab = ws["vocabulary"]
columns = ws.get("columns", {})

# Check reference
am_path = Path("build/data/agent_mediated/agent_mediated.json")
if not am_path.exists():
    print("ERROR: No agent_mediated.json. Run /bootstrap-environment curate first.")
    sys.exit(1)

reference = json.loads(am_path.read_text())
print(f"Vocabulary: {len(vocab)} annotations")
print(f"Reference: {len(reference)} columns decided")

# Check incumbent SVM
incumbent_path = Path("build/models/svm.pkl")
if incumbent_path.exists():
    from atelier.classify.svm_classifier import SVMClassifier
    incumbent = SVMClassifier.load(incumbent_path)
    print(f"Incumbent SVM: {len(incumbent.classes_)} classes")
else:
    print("No incumbent SVM found")

# Check enrichment coverage (Qdrant or JSON export)
print("\nChecking enrichment coverage...")
```

Enrichment coverage must be 100%. If `--payloads <path>` was given,
verify coverage against vocabulary from that file. Otherwise check
Qdrant. If coverage < 100%, halt with: "Enrichment incomplete.
Complete enrichment via /bootstrap-environment or export existing
payloads via: python scripts/export_enriched_annotations.py --format json"

---

## Phase 2 — Enrichment Payload Loading

```python
from atelier.classify.enrichment_loader import load_enrichment_payloads

# If --payloads was given, use JSON export; otherwise Qdrant
payloads_json = Path("...") if payloads_arg else None
payload_path = Path("build/data/svm_training/enrichment_payloads.json")
payload_path.parent.mkdir(parents=True, exist_ok=True)

if payload_path.exists() and not payloads_json:
    payloads = json.loads(payload_path.read_text())
    print(f"Loaded cached payloads: {len(payloads)} annotations")
else:
    payloads = load_enrichment_payloads(
        json_path=payloads_json,
        cfg=cfg,
    )
    payload_path.write_text(json.dumps(payloads, indent=2))
    print(f"Loaded {len(payloads)} enrichment payloads")

# Assess generator readiness
sufficient = sum(1 for p in payloads.values()
                 if len(p.get("prototype_values", [])) >= 3)
thin = len(payloads) - sufficient
print(f"Generator-ready (≥3 prototypes): {sufficient}")
print(f"Thin (will use ICE-matched or inferred generator): {thin}")
```

Agent flags annotations with fewer than 3 prototype values — these
fall to ICE-matched generators (via enrichment metadata keyword
matching) or inferred generators. If many annotations are thin, note
this as an enrichment quality concern that may limit corpus quality.

---

## Phase 3 — Corpus Generation (agent-mediated)

```python
from atelier.classify.taxonomy import load_annotations_from_json
from atelier.classify.synth import generate_user_taxonomy_corpus

# Build user-taxonomy category_set from working_set vocabulary
# (load_annotations_from_json expects the cached annotations format)
cache_path = Path("build/cache/annotations/default.json")
if cache_path.exists():
    user_category_set = load_annotations_from_json(cache_path, hierarchical=True)
else:
    # Fall back: construct from working_set vocabulary if cache unavailable
    print("No cached annotations file; constructing from working_set")
    # ... (construct from ws["vocabulary"])

corpus_dir = Path("build/data/svm_training/corpus")
corpus_dir.mkdir(parents=True, exist_ok=True)

results, coverage = generate_user_taxonomy_corpus(
    user_category_set,
    payloads,
    corpus_dir,
    seed=42,
    variants_per_category=30,
)

# Coverage report
from collections import Counter
source_counts = Counter(coverage.values())
print(f"\nGenerator coverage:")
for source, count in source_counts.most_common():
    print(f"  {source}: {count}")
missing = [code for code, src in coverage.items() if src == "missing"]
if missing:
    print(f"  WARNING: {len(missing)} codes without generators")
```

The registry layers ICE hand-coded generators (matched via enrichment
metadata) > template generators (from prototype_values) > inferred
generators (from category metadata keywords).

### Agent quality review

After corpus generation, review a sample of generated columns:

1. For 5-10 annotations (stratified across subtrees), load the generated
   CSV rows and compare against the enrichment payload's prototype_values.
2. Check: are generated values realistic for deployment context?
3. Check: do sibling annotations produce distinguishable columns?
4. Check: for annotations with only 3 prototypes, is the template
   generator just cycling the same 3 values with char-substitution?
   (If so, note as a quality concern — the enrichment needs richer
   prototypes for this annotation.)

Persist review to `build/data/svm_training/generator_review.json`:
```json
{
  "reviewed_at": "2026-05-20T...",
  "codes_reviewed": 10,
  "quality_ok": 8,
  "flagged": 2,
  "issues": [
    {
      "code": "MNEMONIC",
      "issue": "Only 3 prototypes — generator cycles same values",
      "recommendation": "Enrich with more prototype_values"
    }
  ],
  "coverage": {"hand-coded": 80, "template": 150, "inferred": 40, "missing": 5}
}
```

---

## Phase 4 — Model Training

```python
from atelier.classify.ml_train import train_svm

corpus_dir = Path("build/data/svm_training/corpus")
candidate_path = Path("build/data/svm_training/candidate_svm")

train_svm(corpus_dir, candidate_path)

# Check training stats
classes_path = Path(str(candidate_path) + ".pkl.classes.json")
if classes_path.exists():
    classes = json.loads(classes_path.read_text())
    print(f"Trained SVM: {len(classes)} classes")
else:
    print("WARNING: No classes.json produced")
```

If more than 20% of taxonomy codes were dropped as singletons during
training (SVMClassifier filters classes with <2 examples), flag this
as a corpus quality issue. The dropped codes need more synthetic
variants — either richer enrichment prototypes or higher
`variants_per_category`.

---

## Phase 5 — Evaluation Gate

Score the candidate SVM against the agent-mediated reference.

```python
from atelier.classify.svm_classifier import SVMClassifier, build_svm_text

candidate = SVMClassifier.load(Path("build/data/svm_training/candidate_svm.pkl"))
reference = json.loads(Path("build/data/agent_mediated/agent_mediated.json").read_text())

# Load the most recent pipeline run's column samples for evidence_text
results_dirs = sorted(
    Path("build/results").iterdir(),
    key=lambda p: p.stat().st_mtime, reverse=True,
)
classifications = None
for d in results_dirs:
    cls_path = d / "classifications.json"
    if cls_path.exists():
        classifications = json.loads(cls_path.read_text())
        break

if not classifications:
    print("No pipeline run with classifications found — cannot evaluate")
    # Fall back to synthetic self-evaluation
else:
    # Build column metadata lookup
    col_metadata = {}  # {table.col: {name, type, values}}
    # ... extract from classifications

    correct = 0
    total = 0
    hierarch_correct = 0
    per_class = {}  # {code: {tp, fp, fn}}

    for key, ref_tag in reference.items():
        if ref_tag is None:
            continue
        meta = col_metadata.get(key)
        if not meta:
            continue

        text = build_svm_text(
            meta["column_name"],
            column_type=meta.get("column_type", "string"),
            sample_values=meta.get("sample_values", [])[:5],
        )
        proba = candidate.predict_proba([text])[0]
        pred_tag = max(proba, key=proba.get) if proba else None

        total += 1
        if pred_tag == ref_tag:
            correct += 1
        # Hierarchical: correct if predicted code shares prefix up to depth-2
        if pred_tag and ref_tag:
            pred_parts = pred_tag.split(".")[:3]
            ref_parts = ref_tag.split(".")[:3]
            if pred_parts == ref_parts:
                hierarch_correct += 1

        # Track per-class stats for F1
        per_class.setdefault(ref_tag, {"tp": 0, "fp": 0, "fn": 0})
        if pred_tag == ref_tag:
            per_class[ref_tag]["tp"] += 1
        else:
            per_class[ref_tag]["fn"] += 1
            per_class.setdefault(pred_tag, {"tp": 0, "fp": 0, "fn": 0})
            if pred_tag:
                per_class[pred_tag]["fp"] += 1

    exact_acc = correct / total if total else 0
    hier_acc = hierarch_correct / total if total else 0

    # Macro F1
    f1s = []
    for stats in per_class.values():
        prec = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) else 0
        rec = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0

    print(f"\nCandidate evaluation ({total} columns):")
    print(f"  Exact accuracy: {exact_acc:.3f}")
    print(f"  Hierarchical accuracy: {hier_acc:.3f}")
    print(f"  Macro F1: {macro_f1:.3f}")
```

### A/B comparison

If an incumbent SVM exists, score it on the same column set:

```python
if incumbent_path.exists():
    incumbent = SVMClassifier.load(incumbent_path)
    # ... repeat evaluation loop for incumbent
    print(f"\nIncumbent: exact={inc_acc:.3f}, hier={inc_hier:.3f}")
    print(f"Delta: exact={exact_acc - inc_acc:+.3f}, hier={hier_acc - inc_hier:+.3f}")
```

### Gate decision

```python
evaluation = {
    "evaluated_at": _utc_iso(),
    "reference_size": len(reference),
    "evaluated_columns": total,
    "candidate": {
        "training_samples": "...",
        "training_classes": len(candidate.classes_),
        "exact_accuracy": exact_acc,
        "hierarchical_accuracy": hier_acc,
        "macro_f1": macro_f1,
    },
    "incumbent": {
        "exact_accuracy": inc_acc if incumbent_path.exists() else None,
    },
    "gate_result": "PASS" if exact_acc >= inc_acc else "FAIL",
    "gate_criteria": "candidate >= incumbent on exact accuracy",
}

eval_path = Path("build/data/svm_training/evaluation.json")
eval_path.write_text(json.dumps(evaluation, indent=2))
```

**Gate criterion**: candidate >= incumbent on exact accuracy. If no
incumbent exists, the candidate passes automatically (any model beats
no model).

Agent inspects the confusion matrix for systematic errors. A model
with good overall accuracy but 0% recall on critical codes (e.g., SSN,
credit card numbers) fails the spirit of the gate — flag for the
operator even if the numeric gate passes.

---

## Phase 6 — Promotion

If the evaluation gate passes:

```python
import shutil
from datetime import datetime, timezone

def _utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _utc_date():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

candidate_pkl = Path("build/data/svm_training/candidate_svm.pkl")
target_pkl = Path("build/models/svm.pkl")
target_pkl.parent.mkdir(parents=True, exist_ok=True)

# Archive incumbent
if target_pkl.exists():
    archive_dir = Path(f"build/data/svm_training/archive/{_utc_date()}")
    archive_dir.mkdir(parents=True, exist_ok=True)
    for f in target_pkl.parent.glob("svm.pkl*"):
        shutil.copy2(f, archive_dir / f.name)
    print(f"Archived incumbent -> {archive_dir}")

# Promote candidate
for f in candidate_pkl.parent.glob("candidate_svm.pkl*"):
    target = target_pkl.parent / f.name.replace("candidate_svm", "svm")
    shutil.copy2(f, target)
print(f"Promoted candidate -> {target_pkl}")

# Invalidate per-vocab SVM cache
cache_dir = Path("build/cache/svm")
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print("Invalidated per-vocab SVM cache")

# Append to promotion log
log_path = Path("build/data/svm_training/promotion_log.json")
log = json.loads(log_path.read_text()) if log_path.exists() else []
log.append({
    "promoted_at": _utc_iso(),
    "exact_accuracy": evaluation["candidate"]["exact_accuracy"],
    "hierarchical_accuracy": evaluation["candidate"]["hierarchical_accuracy"],
    "replaced_incumbent_accuracy": evaluation.get("incumbent", {}).get("exact_accuracy"),
})
log_path.write_text(json.dumps(log, indent=2))
```

If the gate fails: report diagnosis, do NOT promote. Recommend:
- If coverage gap -> richer enrichment prototypes for the missing codes
- If class imbalance -> increase `variants_per_category`
- If systematic confusion -> inspect confusable pairs, check enrichment
  anti_examples coverage
- If overall accuracy low -> check enrichment quality (prototype
  diversity, name_hint coverage)

---

## Anti-patterns (DO NOT)

- Do NOT train on real column data. The corpus must be synthetic.
  Enrichment prototypes are generator specifications, not training data.
- Do NOT skip the evaluation gate. Generator quality issues or class
  imbalance can produce an SVM worse than the incumbent.
- Do NOT use enrichment `anti_examples` as positive training data.
  Anti-examples specify what a code is NOT.
- Do NOT train without enrichment data. The SVM must only emit user-
  provided terminology — without enrichment, it cannot know what the
  user's codes mean.
- Do NOT hardcode annotation counts. Always "100% coverage".
- Do NOT promote without inspecting the confusion matrix for
  zero-recall on critical codes.
