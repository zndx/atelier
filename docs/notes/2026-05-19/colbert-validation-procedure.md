# ColBERT Late-Interaction Validation Procedure

Run from the CAI Application pod (access to PGlite + Qdrant).
Validates the new architecture before committing sweep compute.

## P0 fix applied (2026-05-19)

MaxSim scores are sums of per-token max-cosines (range ~1.7–6.4),
not single cosines in [0,1]. Two fixes:

1. **Bridge normalization**: `point.score / num_query_tokens` converts
   to mean per-token cosine ([0,1] range) before passing to mass fn.
2. **Clamp removal**: `min(1.0, score)` removed from
   `_late_interaction_positive_mass` — was flattening all scores to 1.0.

Re-run Steps 2–3 after pulling the fix to confirm discrimination.

## Context

The late-interaction cosine source was pivoted from MiniLM 7-field
pseudo-late-interaction to true ColBERT (colbert-ir/colbertv2.0)
with Qdrant-native MaxSim. The existing enrichment collection has
the old schema — it needs re-population with ColBERT vectors.

Key code changes (all on `feat/dst-late-interaction-cosine`):

- `colbert_encoder.py` — ColBERT model loader, per-token 128-d vectors
- `qdrant_writer.py` — single `colbert` multi-vector field + `compose_annotation_text()`
- `late_interaction_bridge.py` — rewritten: entity text → ColBERT → Qdrant MaxSim → mass
- `mass_functions.py` — `late_interaction_to_mass` accepts `(code, score)` tuples
- `enrichment/loop.py` — calls `compose_annotation_text` → ColBERT encode → upsert
- Deleted: `late_interaction.py`, `multi_vector_features.py`

## Prerequisites

```bash
# Pull the branch changes
git pull origin feat/dst-late-interaction-cosine

# Verify ColBERT encoder loads (downloads ~500MB on first run)
python3 -c "
import sys; sys.path.insert(0, 'src')
from atelier.classify.colbert_encoder import warmup
warmup()
"
# Expected: "ColBERT warmup OK: probe produced N tokens × 128 dims"
```

## Step 1 — Re-enrich a small slice with ColBERT vectors

The enrichment loop's `_embed_annotation()` now calls
`compose_annotation_text()` → ColBERT encoder instead of the old
per-field MiniLM embedding. We need a fresh collection with the new
schema.

Option A — re-run the enrichment loop against a small subset:

```python
import sys; sys.path.insert(0, 'src')

from atelier.classify.colbert_encoder import get_encoder
from atelier.enrichment.qdrant_writer import (
    compose_annotation_text, AnnotationVectors,
    ensure_collection, upsert_point, build_point,
    source_row_hash, taxonomy_version_hash,
    COLBERT_VECTOR_NAME,
)
from qdrant_client import QdrantClient

# Connect to local Qdrant
client = QdrantClient(url="http://127.0.0.1:6333")
encoder = get_encoder()

# Target collection — use a validation-specific name to avoid
# clobbering the production collection
COLLECTION = "annotations_default_v1_colbert_validation"
EMBEDDING_DIM = encoder.dim  # 128

ensure_collection(
    client,
    collection=COLLECTION,
    embedding_dim=EMBEDDING_DIM,
    recreate=True,  # fresh start for validation
)

# Read existing enriched payloads from the current collection.
# The payloads are schema-compatible — only the vectors change.
SOURCE_COLLECTION = "annotations_default_v1"

# Scroll all points from the existing collection (payloads only)
points, _offset = client.scroll(
    collection_name=SOURCE_COLLECTION,
    limit=30,  # small slice for validation
    with_payload=True,
    with_vectors=False,
)

print(f"Read {len(points)} annotations from {SOURCE_COLLECTION}")

# Re-embed each annotation through ColBERT and upsert
for pt in points:
    payload = pt.payload or {}
    code = payload.get("code", "?")

    text = compose_annotation_text(payload)
    if not text:
        text = payload.get("label") or code
        print(f"  WARN: empty composed text for {code}, using label fallback")

    token_vectors = encoder.encode_single(text)
    vectors = AnnotationVectors(colbert=token_vectors.tolist())

    # Reuse the existing point ID (content-addressed)
    from qdrant_client.http import models as qm
    client.upsert(
        collection_name=COLLECTION,
        points=[qm.PointStruct(
            id=pt.id,
            vector=vectors.to_qdrant_vectors(),
            payload=payload,
        )],
    )
    print(f"  {code}: {token_vectors.shape[0]} tokens × {token_vectors.shape[1]}d")

print(f"\nDone: {len(points)} annotations in {COLLECTION}")
```

Option B — if the existing collection is gone or has too few points,
run the full enrichment loop but limit to 20-30 annotations:

```bash
# The enrichment loop reads the source taxonomy and writes to Qdrant.
# Limit is controlled by the loop's batch size.
python3 -c "
import sys; sys.path.insert(0, 'src')
from atelier.enrichment.loop import run_enrichment_loop
# Check the function signature for a limit/max_rows parameter
import inspect; print(inspect.signature(run_enrichment_loop))
"
# Adjust invocation based on available parameters
```

### Expected outcome

- Collection `annotations_default_v1_colbert_validation` exists with
  20-30 points
- Each point has a `colbert` multi-vector field (N tokens × 128 dims)
- Payloads are unchanged from the source collection

### Verify

```python
info = client.get_collection(COLLECTION)
print(f"Points: {info.points_count}")
print(f"Vectors config: {info.config.params.vectors}")
# Should show: colbert → size=128, multivector_config with MAX_SIM
```


## Step 2 — Spot-check 5-10 columns against known agent-mediated reference

Pick columns with known labels from the agent-mediated reference file.
Run them through the bridge and check MaxSim scores + ranking.

```python
import sys, json; sys.path.insert(0, 'src')

from atelier.classify.colbert_encoder import get_encoder
from qdrant_client import QdrantClient
from atelier.enrichment.qdrant_writer import COLBERT_VECTOR_NAME

client = QdrantClient(url="http://127.0.0.1:6333")
encoder = get_encoder()
COLLECTION = "annotations_default_v1_colbert_validation"

# Load agent-mediated reference
gt_path = "build/data/agent_mediated/agent_mediated.json"
with open(gt_path) as f:
    gt = json.load(f)

# Pick 5-10 columns with diverse types — mix of easy and hard.
# Adjust these based on what's in the agent-mediated reference:
spot_checks = [
    # (table, column, expected_code) — fill from agent-mediated reference
    # Examples (adjust to match your reference file's shape):
]

# If reference is keyed by qualified_name:
for key, entry in list(gt.items())[:10]:
    code = entry if isinstance(entry, str) else entry.get("code", entry.get("label", "?"))
    spot_checks.append((key, code))

print(f"Spot-checking {len(spot_checks)} columns\n")

for item in spot_checks:
    if len(item) == 2:
        col_key, expected = item
        # Build a simple entity text (in production, ColumnFeatures
        # builds a richer text with samples, type, siblings, etc.)
        entity_text = col_key.replace(".", " | ")
    else:
        table, col, expected = item
        entity_text = f"{col} | in {table}"

    query_vectors = encoder.encode_single(entity_text)

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vectors.tolist(),
        using=COLBERT_VECTOR_NAME,
        limit=5,
        with_payload=True,
    )

    print(f"Column: {col_key}")
    print(f"  Expected: {expected}")
    print(f"  Entity text: {entity_text[:80]}...")
    if results.points:
        for i, pt in enumerate(results.points[:5]):
            code = (pt.payload or {}).get("code", "?")
            label = (pt.payload or {}).get("label", "?")
            marker = " <<<" if code == expected else ""
            print(f"  #{i+1}: {code} ({label}) score={pt.score:.4f}{marker}")

        top1 = (results.points[0].payload or {}).get("code")
        top1_score = results.points[0].score
        top2_score = results.points[1].score if len(results.points) > 1 else 0
        margin = top1_score - top2_score
        hit = "HIT" if top1 == expected else "MISS"
        print(f"  → {hit} | margin={margin:.4f}")
    else:
        print("  → NO RESULTS")
    print()
```

### What to look for

- **Top-1 accuracy**: does the correct annotation rank first?
  Don't expect 100% on column-name-only queries (no samples or
  type info). Production uses `ColumnFeatures.to_embedding_text()`
  which includes samples, cardinality, patterns, siblings — much
  richer signal.
- **Score magnitude**: MaxSim scores should be in the 0.4-0.8 range
  for correct matches. Scores near 0.3 suggest weak signal.
- **Margin**: Δ between top-1 and top-2 scores. The ColBERT probe
  on synthetic text showed Δ≈0.20. Real data with column-name-only
  queries will be narrower, but Δ<0.02 suggests the model can't
  discriminate (which is expected for name-only — the signal comes
  from samples in production).
- **Namespace alignment**: all returned codes should be in the user
  vocabulary. If you see ICE.* codes, the collection has the wrong
  annotation set.


## Step 3 — Mass function sanity check

Feed the same spot-check columns through the full mass function path
and verify the belief assignments.

```python
import sys, json; sys.path.insert(0, 'src')

from atelier.classify.colbert_encoder import get_encoder
from atelier.classify.mass_functions import late_interaction_to_mass
from qdrant_client import QdrantClient
from atelier.enrichment.qdrant_writer import COLBERT_VECTOR_NAME

# Load the frame (need taxonomy for the FrameOfDiscernment)
from atelier.classify.taxonomy import load_annotations_from_json

cat_set = load_annotations_from_json("build/data/vocab/default_annotations.json")
# Adjust the path above to wherever the serialized vocabulary lives.
# Alternative: load from the DB if the JSON isn't on disk:
#   from atelier.db.dao import AtelierDao
#   dao = AtelierDao()
#   cats = dao.get_vocabulary(...)

# Build a frame from the category set
from atelier.classify.belief import FrameOfDiscernment
frame = FrameOfDiscernment.from_category_set(cat_set)

client = QdrantClient(url="http://127.0.0.1:6333")
encoder = get_encoder()
COLLECTION = "annotations_default_v1_colbert_validation"

# Test with a known column
entity_text = "email_addr | varchar | john@example.com, jane.doe@company.org"
query_vectors = encoder.encode_single(entity_text)

results = client.query_points(
    collection_name=COLLECTION,
    query=query_vectors.tolist(),
    using=COLBERT_VECTOR_NAME,
    limit=min(len(frame.singletons) + len(frame.internal_nodes), 50),
    with_payload=True,
)

# Build scored_tags
scored_tags = []
for pt in results.points:
    code = (pt.payload or {}).get("code")
    if code and (code in frame.singletons or code in frame.internal_nodes):
        scored_tags.append((code, pt.score))

print(f"In-frame results: {len(scored_tags)} / {len(results.points)} total")

mass = late_interaction_to_mass(scored_tags, frame)

# Print top-5 focal elements by mass
sorted_masses = sorted(mass.masses.items(), key=lambda kv: -kv[1])
print("\nTop-5 focal elements:")
for fe, m in sorted_masses[:5]:
    if m < 0.001:
        continue
    print(f"  {fe.label or fe.codes}: {m:.4f}")

theta_mass = mass.masses.get(frame.theta, 0.0)
print(f"\nΘ (ignorance): {theta_mass:.4f}")
total = sum(mass.masses.values())
print(f"Total mass: {total:.6f} (should be ≈1.0)")

# Belief + plausibility for the expected code
from atelier.classify.belief import HierarchicalClassification
hc = HierarchicalClassification.from_mass(mass, frame)
# Print bel/pl for top candidates
print("\nBel / Pl for top candidates:")
for code in [t[0] for t in scored_tags[:5]]:
    bel = hc.belief(code)
    pl = hc.plausibility(code)
    print(f"  {code}: bel={bel:.4f}  pl={pl:.4f}  gap={pl-bel:.4f}")
```

### What to look for

- **Mass conservation**: total mass ≈ 1.0
- **Θ mass**: should be substantial (0.60-0.85) given the 0.20
  discount — the source is intentionally conservative
- **Top-1 mass**: the correct annotation should carry the largest
  non-Θ mass
- **Bel/Pl gap**: smaller gap = more focused evidence. Very large
  gap (>0.5) suggests the frame structure is diffusing mass across
  the hierarchy
- **No mass on non-vocabulary codes**: every focal element code
  should be in the frame


## Step 4 — Richer spot-check with ColumnFeatures (if available)

Steps 2-3 use column-name-only entity text. In production the bridge
receives `ColumnFeatures` with samples, type, cardinality, patterns,
siblings. If you can construct or load a `ColumnFeatures` for a known
column, the signal quality will be much more representative:

```python
from atelier.classify.features import ColumnFeatures

# Example: build features for a column you know
features = ColumnFeatures(
    column_name="email_addr",
    column_type="varchar",
    sample_values=["john@example.com", "jane.doe@corp.org", "admin@test.net"],
    cardinality=2847,
    null_ratio=0.02,
    # ... fill in what you can
)
entity_text = features.to_embedding_text()
print(f"Entity text: {entity_text}")
# Then feed to ColBERT + Qdrant as in Step 3
```

The richer text should produce stronger MaxSim scores and wider
margins vs the column-name-only baseline.


## Decision point

After steps 1-3:

- If top-1 accuracy on the spot-checks looks reasonable AND mass
  assignments are well-formed → proceed to a small sweep (3-4 cells,
  not the full 12-cell grid) to get quantitative accuracy numbers.
- If MaxSim scores are non-discriminative (margins < 0.01) or mass
  conservation is broken → investigate before spending sweep compute.
- If the collection re-population fails → debug the enrichment loop
  integration before anything else.

The goal is to confirm the architecture produces discriminative signal
on real data before committing 1-2 hours of sweep compute.
