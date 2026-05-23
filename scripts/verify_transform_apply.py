#!/usr/bin/env python
"""Phase 6 of /evolve-classification — subset verifier for a transform apply.

This is NOT a full pipeline re-run.  It computes concrete deltas on the
*affected columns* — those whose baseline cosine top-1 was one of the
transformed target codes (i.e. the columns whose ranking the transform
can plausibly move).  Output feeds Phase 7's change management guide.

Algorithm:

  1. Load manifest + per-transform records.  Discover source + target
     collection names.
  2. From the baseline run's classifications.json, identify columns
     whose predicted_code is in the set of accepted target codes.
  3. For each affected column, query Qdrant against BOTH collections at
     K=25, compute:
       - new top-1 vs old top-1
       - reference-code rank (old vs new)
       - cosine score delta on the reference code
  4. Roll-up per target_code and overall.
  5. Subset accuracy delta against the dual-format reference (captured
     codes — drift-stable).
  6. Write build/data/transforms/verify_<manifest_id>.json.

Explicitly does NOT: re-classify the full corpus, retrain SVM/CatBoost,
or touch the registry's current pointer.  Those remain operator-initiated.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Reference helpers ─────────────────────────────────────────────


def load_dual_reference(path: Path) -> dict[str, str]:
    """Return {qkey: captured_code} from a dual-format reference file."""
    raw = json.loads(path.read_text())
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict) and v.get("code"):
            out[k] = v["code"]
        elif isinstance(v, str) and v:
            # Legacy mnemonic-only — not drift-stable, but acceptable here
            # as fallback so the verifier doesn't crash on un-migrated refs.
            pass
    return out


# ── Qdrant queries ────────────────────────────────────────────────


def query_top_k(client, collection: str, vectors: list[list[float]],
                k: int = 25) -> list[tuple[str, float, bool]]:
    """Run a ColBERT multi-vector MaxSim query.

    Returns list of (code, score, is_leaf?) for the top-K hits.  When the
    payload lacks an explicit ``is_leaf`` field, returns None for that slot
    (callers can treat as unknown).
    """
    from atelier.enrichment.qdrant_writer import COLBERT_VECTOR_NAME

    result = client.query_points(
        collection_name=collection,
        query=vectors,
        using=COLBERT_VECTOR_NAME,
        limit=k,
        with_payload=True,
    )
    rows = []
    for p in result.points:
        payload = p.payload or {}
        code = payload.get("code")
        if not code:
            continue
        # Approximate is_leaf from the payload's parent_path depth, or
        # leave as None when payload doesn't carry it
        is_leaf = None
        rows.append((code, float(p.score), is_leaf))
    return rows


# ── Re-encode helpers ─────────────────────────────────────────────


def get_or_encode_entity_text(qkey: str, classification: dict, encoder) -> list[list[float]]:
    """Return entity-text vectors (preferring cached embedding_text).

    The classification row carries ``embedding_text`` — the same string
    the baseline pipeline encoded against the source collection.  Re-
    encoding it deterministically reproduces the baseline query vector;
    Qdrant lookups then differ ONLY because the target collection's
    annotation vectors differ.
    """
    text = classification.get("embedding_text") or ""
    if not text:
        return None
    vecs = encoder.encode_single(text)
    return vecs.tolist()


# ── Main flow ─────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("manifest_path",
        help="Path to build/data/transforms/manifests/<cohort>.json")
    parser.add_argument("--baseline-run", default=None,
        help="Run dir (build/results/<run_id>); default: manifest.source_run")
    parser.add_argument("--reference", default="build/data/agent_mediated/agent_mediated.json",
        help="Dual-format agent-mediated reference")
    parser.add_argument("--k", type=int, default=25,
        help="Top-K to query against each collection")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    manifest_path = Path(args.manifest_path)
    if not manifest_path.is_file():
        sys.exit(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    print(f"Manifest:        {manifest_path.name}")
    print(f"  source:        {manifest.get('source_collection')}")
    print(f"  target:        {manifest.get('target_collection')}")
    print(f"  applied_at:    {manifest.get('applied_at')}")

    # Load per-transform records to get target codes
    records_dir = Path("build/data/transforms/records")
    transform_records = []
    for tid in manifest.get("transform_ids") or []:
        rp = records_dir / f"{tid}.json"
        if rp.is_file():
            transform_records.append(json.loads(rp.read_text()))
    target_codes = {
        r["target"]["code"] for r in transform_records
        if r["target"].get("code") and r.get("status") in ("applied", "planned")
    }
    print(f"  target codes:  {len(target_codes)}")

    # ── Baseline run ──────────────────────────────────────────────
    baseline_run_dir = Path(args.baseline_run or f"build/results/{manifest.get('source_run')}")
    if not baseline_run_dir.is_dir():
        sys.exit(f"Baseline run dir not found: {baseline_run_dir}")
    cls_path = baseline_run_dir / "classifications.json"
    classifications = json.loads(cls_path.read_text())

    # ── Affected-column set ──────────────────────────────────────
    affected = []
    for c in classifications:
        pred = c.get("predicted_code")
        attribution = c.get("cosine_attribution") or {}
        top_k_baseline = attribution.get("top_k") or []
        old_top1 = (top_k_baseline[0]["code"]
                    if top_k_baseline else pred)
        if old_top1 in target_codes or pred in target_codes:
            affected.append(c)
    print(f"\nAffected columns: {len(affected)} "
          f"(baseline cosine top-1 in target_codes)")

    # ── Reference (drift-stable captured codes) ──────────────────
    ref = load_dual_reference(Path(args.reference))

    # ── Qdrant + encoder ─────────────────────────────────────────
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        sys.exit(f"qdrant_client not importable: {exc}")
    from atelier.classify.colbert_encoder import get_encoder

    encoder = get_encoder()
    qdrant_url = "http://127.0.0.1:6333"
    client = QdrantClient(url=qdrant_url)

    source_col = manifest["source_collection"]
    target_col = manifest["target_collection"]

    # ── Per-column verification ──────────────────────────────────
    per_column = []
    rank_improved = 0
    rank_regressed = 0
    rank_unchanged = 0
    top1_changed = 0
    new_matches_ref = 0
    old_matches_ref = 0
    per_target_code_movement = defaultdict(Counter)

    for c in affected:
        qkey = f"{c.get('table_name')}.{c.get('column_name')}"
        ref_code = ref.get(qkey)
        vecs = get_or_encode_entity_text(qkey, c, encoder)
        if vecs is None:
            continue

        # Query both collections
        try:
            old_topk = query_top_k(client, source_col, vecs, k=args.k)
            new_topk = query_top_k(client, target_col, vecs, k=args.k)
        except Exception as exc:
            logger.warning("Query failed for %s: %s", qkey, exc)
            continue

        old_top1 = old_topk[0][0] if old_topk else None
        new_top1 = new_topk[0][0] if new_topk else None

        def _rank_of(code, topk):
            for i, (kk, _, _) in enumerate(topk, start=1):
                if kk == code:
                    return i
            return None
        old_rank = _rank_of(ref_code, old_topk) if ref_code else None
        new_rank = _rank_of(ref_code, new_topk) if ref_code else None

        if old_top1 != new_top1:
            top1_changed += 1
        if ref_code:
            if old_top1 == ref_code:
                old_matches_ref += 1
            if new_top1 == ref_code:
                new_matches_ref += 1
            if old_rank is not None and new_rank is not None:
                if new_rank < old_rank:
                    rank_improved += 1
                elif new_rank > old_rank:
                    rank_regressed += 1
                else:
                    rank_unchanged += 1
            elif old_rank is None and new_rank is not None:
                rank_improved += 1
            elif old_rank is not None and new_rank is None:
                rank_regressed += 1

        # Per-target movement: when old_top1 was one of our targets,
        # where did the column move to?
        if old_top1 in target_codes:
            per_target_code_movement[old_top1][new_top1 or "<no_top1>"] += 1

        per_column.append({
            "qkey": qkey,
            "ref_code": ref_code,
            "old_top1": old_top1,
            "new_top1": new_top1,
            "top1_changed": old_top1 != new_top1,
            "old_rank": old_rank,
            "new_rank": new_rank,
            "rank_delta": (
                (new_rank or args.k + 1) - (old_rank or args.k + 1)
                if (old_rank or new_rank) else None
            ),
            "old_matches_ref": old_top1 == ref_code if ref_code else None,
            "new_matches_ref": new_top1 == ref_code if ref_code else None,
        })

    # ── Roll-up ──────────────────────────────────────────────────
    n_aff = len(per_column)
    print(f"\nVerification roll-up (n={n_aff}):")
    print(f"  top1_changed:         {top1_changed}")
    print(f"  rank_improved:        {rank_improved}")
    print(f"  rank_regressed:       {rank_regressed}")
    print(f"  rank_unchanged:       {rank_unchanged}")
    print(f"  old_top1 == ref:      {old_matches_ref}")
    print(f"  new_top1 == ref:      {new_matches_ref}")
    print(f"  subset Δ matches:     {new_matches_ref - old_matches_ref:+d}")

    # Per-target movement
    print(f"\nPer-target_code movement:")
    for code, dest_counter in per_target_code_movement.items():
        print(f"  {code}:")
        for dest, n in dest_counter.most_common(5):
            print(f"    → {dest}: {n}")

    # Regression watch list
    regression_watch = [
        r for r in per_column
        if r.get("old_matches_ref") and not r.get("new_matches_ref")
    ]
    print(f"\nRegression watch (was correct, no longer correct): "
          f"{len(regression_watch)}")
    for r in regression_watch[:10]:
        print(f"  {r['qkey']}  was={r['old_top1']} now={r['new_top1']}")

    # ── Persist ──────────────────────────────────────────────────
    out_dir = Path("build/data/transforms")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "manifest_id": manifest.get("manifest_id"),
        "manifest_path": str(manifest_path),
        "verified_at": _now_iso(),
        "baseline_run": str(baseline_run_dir),
        "source_collection": source_col,
        "target_collection": target_col,
        "k": args.k,
        "counts": {
            "n_columns_affected": n_aff,
            "n_top1_changed": top1_changed,
            "n_rank_improved": rank_improved,
            "n_rank_regressed": rank_regressed,
            "n_rank_unchanged": rank_unchanged,
            "n_old_matches_ref": old_matches_ref,
            "n_new_matches_ref": new_matches_ref,
            "subset_match_delta": new_matches_ref - old_matches_ref,
        },
        "per_target_code_movement": {
            code: dict(counter)
            for code, counter in per_target_code_movement.items()
        },
        "regression_watch": regression_watch,
        "per_column": per_column,
    }
    out_path = out_dir / f"verify_{manifest.get('manifest_id', 'unknown')}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
