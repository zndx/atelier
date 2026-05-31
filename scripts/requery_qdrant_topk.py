#!/usr/bin/env python
"""Re-query Qdrant for top-K cosine candidates per column.

Reads 5ef4868c's classifications.json (which has the per-column
embedding_text), re-encodes via ColBERT, queries the registered
Qdrant collection for top-K = 10, persists results so the union-focal
sweep can test K ∈ {3, 5, 7, 10}.

Output: build/runs/calibration/cosine_topk_5ef4868c.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classifications", type=Path,
                    default=Path("build/results/5ef4868c/classifications.json"))
    ap.add_argument("--collection", required=True,
                    help="Qdrant collection name")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--output", type=Path,
                    default=Path("build/runs/calibration/cosine_topk_5ef4868c.json"))
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only N columns (0 = all)")
    args = ap.parse_args()

    cls = json.loads(args.classifications.read_text())
    cols = [c for c in cls if c.get("reference_code") and c.get("embedding_text")]
    if args.limit:
        cols = cols[:args.limit]
    print(f"processing {len(cols)} columns; top-K={args.top_k}", flush=True)

    import qdrant_client
    from atelier.classify.colbert_encoder import get_encoder
    from atelier.enrichment.qdrant_writer import COLBERT_VECTOR_NAME

    client = qdrant_client.QdrantClient(url=args.qdrant_url, timeout=30)
    encoder = get_encoder()

    # Sanity: one query first to measure latency
    sample_text = cols[0]["embedding_text"]
    t0 = time.time()
    vec = encoder.encode_single(sample_text)
    t_enc = time.time() - t0
    t0 = time.time()
    _ = client.query_points(
        collection_name=args.collection,
        query=vec.tolist(),
        using=COLBERT_VECTOR_NAME,
        limit=args.top_k,
        with_payload=True,
    )
    t_q = time.time() - t0
    print(f"per-column timing: encode={t_enc*1000:.0f}ms  query={t_q*1000:.0f}ms",
          flush=True)
    print(f"estimated wall: {(t_enc+t_q)*len(cols)/60:.1f} min", flush=True)

    out = {}
    t_start = time.time()
    for i, c in enumerate(cols):
        text = c["embedding_text"]
        vec = encoder.encode_single(text)
        num_tokens = vec.shape[0]
        results = client.query_points(
            collection_name=args.collection,
            query=vec.tolist(),
            using=COLBERT_VECTOR_NAME,
            limit=args.top_k,
            with_payload=True,
        )
        # Per late_interaction_bridge: normalize by query token count
        top_k_list = []
        for p in results.points:
            code = (p.payload or {}).get("code")
            if code is None:
                continue
            top_k_list.append({
                "code": code,
                "maxsim_score": round(p.score / max(num_tokens, 1), 6),
            })
        key = f"{c['table_name']}.{c['column_name']}"
        out[key] = {
            "reference_code": c["reference_code"],
            "embedding_text": text[:200],
            "top_k": top_k_list,
        }
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (i+1) * (len(cols) - (i+1))
            print(f"  [{i+1}/{len(cols)}] elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                  flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    elapsed = time.time() - t_start
    print(f"\ndone: {len(out)} cols in {elapsed:.0f}s → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
