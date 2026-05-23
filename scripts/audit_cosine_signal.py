#!/usr/bin/env python
"""Audit cosine signal quality from a classifications.json run.

Three sub-audits:

  1. **Internal-node cosine top-1: subtree-correctness** (offline)
     For internal-node top-1 picks, what fraction have the reference
     code as a descendant?  Tells us whether the now-live subtree-FE
     routing is paying its keep, or whether cosine internal-node
     signal is cross-subtree noise.

  2. **Per-confusable-cluster top-K=25 rank distribution** (live mode)
     For the top error clusters from scoring_summary.md, re-query
     Qdrant at K=25 and bucket where the reference code ranks.  Tells
     us whether anti-example-as-positive-on-sibling can help (if the
     reference is in rank 2-3 it can; if it's not in top-K, deeper
     enrichment is needed).

  3. **Default-pick bias** (offline)
     What fraction of all columns share the same cosine top-1 code?
     A sticky default-pick is a smoking gun for embedding-space
     centroid bias on that tag.

Usage:
    python scripts/audit_cosine_signal.py build/results/7bbe4533
    python scripts/audit_cosine_signal.py build/results/7bbe4533 --live

Live mode must be invoked from the App pod's Web Terminal Agent
(same pattern as ``scripts/diag_late_interaction.py``) because Qdrant
on 127.0.0.1:6333 is only reachable inside that pod.

Output:
    build/diag/cosine_signal_audit.md
    build/diag/cosine_signal_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "src")


def is_descendant(child: str, parent: str) -> bool:
    """True iff child is a strict descendant of parent in dotted taxonomy."""
    return child.startswith(parent + ".")


def _summary_example(c: dict, top1: str, ref: str) -> dict:
    return {
        "column": f"{c.get('table_name')}.{c.get('column_name')}",
        "cosine_top1": top1,
        "reference": ref,
        "predicted": c.get("predicted_code"),
    }


def audit_subtree_correctness(rows: list[dict]) -> dict:
    """Audit 1: internal-node cosine top-1 → subtree-correctness."""
    subtree_correct = 0
    exact_internal = 0
    cross_subtree = 0
    skipped_no_ref = 0
    total = 0
    examples: dict[str, list] = defaultdict(list)

    for c in rows:
        tk = (c.get("cosine_attribution") or {}).get("top_k") or []
        if not tk:
            continue
        top1 = tk[0]
        if top1.get("is_leaf"):
            continue
        ref = c.get("reference_code")
        if not ref:
            skipped_no_ref += 1
            continue
        total += 1
        top1_code = top1["code"]
        if ref == top1_code:
            exact_internal += 1
            examples["exact_internal"].append(_summary_example(c, top1_code, ref))
        elif is_descendant(ref, top1_code):
            subtree_correct += 1
            examples["subtree_correct"].append(_summary_example(c, top1_code, ref))
        else:
            cross_subtree += 1
            examples["cross_subtree"].append(_summary_example(c, top1_code, ref))

    return {
        "total_internal_top1_with_ref": total,
        "subtree_correct": subtree_correct,
        "exact_internal": exact_internal,
        "cross_subtree": cross_subtree,
        "skipped_no_ref": skipped_no_ref,
        "examples": {k: v[:5] for k, v in examples.items()},
    }


def audit_default_pick_bias(rows: list[dict]) -> dict:
    """Audit 3: top-10 most-frequent cosine top-1 codes."""
    counter: Counter = Counter()
    for c in rows:
        tk = (c.get("cosine_attribution") or {}).get("top_k") or []
        if tk:
            counter[tk[0]["code"]] += 1
    total = sum(counter.values())
    top10 = []
    for code, n in counter.most_common(10):
        top10.append({
            "code": code,
            "count": n,
            "pct": round(100 * n / total, 1) if total else 0.0,
        })
    return {"total_with_top1": total, "top10": top10}


def audit_cluster_rank_distribution_live(
    rows: list[dict],
    error_clusters: list[tuple[str, str]],
    cfg,
    qdrant_url: str,
    collection: str,
    k: int = 25,
) -> dict:
    """Audit 2: re-query Qdrant at K=25 per cluster and bucket reference ranks.

    Reuses the bridge's encode-and-query path: same ColBERT model, same
    multi-vector query, same payload shape.  We only collect ranks; no
    mass conversion needed for the audit.
    """
    from qdrant_client import QdrantClient

    from atelier.classify.colbert_encoder import get_encoder, set_model_name
    from atelier.enrichment.qdrant_writer import COLBERT_VECTOR_NAME

    colbert_model = getattr(cfg, "classify_colbert_model", None)
    if colbert_model:
        set_model_name(colbert_model)
    encoder = get_encoder()
    client = QdrantClient(url=qdrant_url or "http://127.0.0.1:6333")

    results = {}
    for predicted_label, reference_label in error_clusters:
        affected = [
            c for c in rows
            if c.get("predicted_label") == predicted_label
            and c.get("reference_label") == reference_label
        ]
        if not affected:
            continue

        buckets = {
            "rank_1": 0,
            "rank_2_3": 0,
            "rank_4_10": 0,
            "rank_11_25": 0,
            "not_in_top_k": 0,
            "query_error": 0,
        }
        details = []

        for c in affected:
            entity_text = c.get("embedding_text")
            ref_code = c.get("reference_code")
            if not entity_text or not ref_code:
                continue

            try:
                vecs = encoder.encode_single(entity_text)
                points = client.query_points(
                    collection_name=collection,
                    query=vecs.tolist(),
                    using=COLBERT_VECTOR_NAME,
                    limit=k,
                    with_payload=True,
                ).points
            except Exception as exc:  # noqa: BLE001 — audit, never crash
                buckets["query_error"] += 1
                details.append({
                    "column": f"{c.get('table_name')}.{c.get('column_name')}",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue

            ordered_codes = [(p.payload or {}).get("code") for p in points]
            rank = next(
                (i + 1 for i, code in enumerate(ordered_codes) if code == ref_code),
                None,
            )

            if rank is None:
                buckets["not_in_top_k"] += 1
            elif rank == 1:
                buckets["rank_1"] += 1
            elif rank <= 3:
                buckets["rank_2_3"] += 1
            elif rank <= 10:
                buckets["rank_4_10"] += 1
            else:
                buckets["rank_11_25"] += 1

            details.append({
                "column": f"{c.get('table_name')}.{c.get('column_name')}",
                "reference_code": ref_code,
                "rank": rank,
                "top3": ordered_codes[:3],
            })

        results[f"{predicted_label} -> {reference_label}"] = {
            "affected": len(affected),
            "buckets": buckets,
            "details": details[:10],
        }

    return results


# ── Markdown rendering ────────────────────────────────────────────


def render_markdown(
    audit_1: dict,
    audit_2: dict | None,
    audit_3: dict,
    run_id: str,
    n_rows: int,
) -> str:
    lines = [f"# Cosine signal audit — run {run_id}", ""]
    lines.append(f"Classifications loaded: **{n_rows}**")
    lines.append("")

    # ── Audit 1 ──
    lines.append("## Audit 1 — Internal-node cosine top-1: subtree-correctness")
    lines.append("")
    n = audit_1["total_internal_top1_with_ref"]
    lines.append(f"Total internal-node top-1 picks with reference: **{n}**")
    lines.append("")
    lines.append("| bucket | count | % |")
    lines.append("|---|---:|---:|")
    for key in ("subtree_correct", "exact_internal", "cross_subtree"):
        cnt = audit_1[key]
        pct = (100 * cnt / n) if n else 0.0
        lines.append(f"| {key} | {cnt} | {pct:.1f}% |")
    lines.append("")
    lines.append("**Interpretation:**")
    if n == 0:
        lines.append("- No internal-node top-1 picks to analyze.")
    else:
        sc = audit_1["subtree_correct"]
        cs = audit_1["cross_subtree"]
        ei = audit_1["exact_internal"]
        productive_pct = 100 * (sc + ei) / n
        cross_pct = 100 * cs / n
        if productive_pct > 50:
            lines.append(
                f"- Productive (subtree-correct or exact-internal) rate "
                f"{productive_pct:.1f}% — routing-fix is paying its keep."
            )
        else:
            lines.append(
                f"- Productive rate only {productive_pct:.1f}%; cross-subtree "
                f"rate {cross_pct:.1f}% dominates.  Cosine internal-node "
                f"signal is largely cross-subtree noise — enriched text on "
                f"the confused codes is the right next move."
            )
    lines.append("")
    lines.append("### Sample picks (first 5 per bucket)")
    lines.append("")
    for key in ("subtree_correct", "exact_internal", "cross_subtree"):
        if audit_1["examples"].get(key):
            lines.append(f"**{key}**")
            for ex in audit_1["examples"][key]:
                lines.append(
                    f"- `{ex['column']}`  cosine_top1=`{ex['cosine_top1']}`  "
                    f"ref=`{ex['reference']}`  predicted=`{ex['predicted']}`"
                )
            lines.append("")

    # ── Audit 2 ──
    lines.append("## Audit 2 — Per-confusable-cluster top-K=25 rank distribution")
    lines.append("")
    if audit_2 is None:
        lines.append("*Live-mode only.  Re-run with `--live` inside the App pod*")
        lines.append("*to populate this section.*")
        lines.append("")
    else:
        lines.append(
            "For each error cluster, the audit re-queries Qdrant at K=25 "
            "using the column's recorded embedding text + the same ColBERT "
            "encoder the bridge uses.  Bucket = where the reference code "
            "ranks in the returned candidate list."
        )
        lines.append("")
        for pattern, res in audit_2.items():
            lines.append(f"### {pattern}")
            lines.append("")
            lines.append(f"Affected columns: **{res['affected']}**")
            lines.append("")
            lines.append("| bucket | count |")
            lines.append("|---|---:|")
            for bkey, bval in res["buckets"].items():
                if bval > 0 or bkey != "query_error":
                    lines.append(f"| {bkey} | {bval} |")
            lines.append("")
            if res["details"]:
                lines.append("Sample (first 10):")
                lines.append("")
                for d in res["details"]:
                    if "error" in d:
                        lines.append(f"- `{d['column']}`  ERROR: {d['error']}")
                    else:
                        rank = d.get("rank")
                        rank_s = "not in top-25" if rank is None else f"rank {rank}"
                        lines.append(
                            f"- `{d['column']}`  ref=`{d['reference_code']}`  "
                            f"→ {rank_s}  top3={d.get('top3')}"
                        )
                lines.append("")

    # ── Audit 3 ──
    lines.append("## Audit 3 — Default-pick bias")
    lines.append("")
    lines.append(
        f"Most-frequent cosine top-1 codes "
        f"(of {audit_3['total_with_top1']} columns):"
    )
    lines.append("")
    lines.append("| code | count | % |")
    lines.append("|---|---:|---:|")
    for r in audit_3["top10"]:
        lines.append(f"| `{r['code']}` | {r['count']} | {r['pct']}% |")
    lines.append("")
    lines.append("**Interpretation:**")
    if audit_3["top10"]:
        top_pct = audit_3["top10"][0]["pct"]
        top_code = audit_3["top10"][0]["code"]
        if top_pct > 15:
            lines.append(
                f"- A single code (`{top_code}`) dominates {top_pct}% of "
                f"all cosine top-1 picks.  Strong centroid-bias signal — "
                f"that tag's enrichment is likely too generic.  Tightening "
                f"its embedding text should precede other signal-quality "
                f"work."
            )
        else:
            lines.append(
                f"- Top code (`{top_code}`) at {top_pct}% — no single "
                f"dominant default pick.  Distribution looks reasonably "
                f"spread."
            )
    lines.append("")

    return "\n".join(lines) + "\n"


# ── CLI ───────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dir",
        help="Path to build/results/<run_id> containing classifications.json",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Enable Audit 2 (requires App pod with Qdrant on 127.0.0.1:6333)",
    )
    parser.add_argument(
        "--out-dir", default="build/diag",
        help="Output directory (default: build/diag)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    classifications_path = run_dir / "classifications.json"
    if not classifications_path.exists():
        print(f"ERROR: {classifications_path} not found", file=sys.stderr)
        return 2

    with open(classifications_path) as f:
        rows = json.load(f)
    run_id = run_dir.name
    print(f"Loaded {len(rows)} classifications from {classifications_path}")

    print("Running Audit 1 (subtree-correctness, offline)...")
    audit_1 = audit_subtree_correctness(rows)
    print(
        f"  internal_top1_with_ref={audit_1['total_internal_top1_with_ref']} "
        f"subtree_correct={audit_1['subtree_correct']} "
        f"exact_internal={audit_1['exact_internal']} "
        f"cross_subtree={audit_1['cross_subtree']}"
    )

    print("Running Audit 3 (default-pick bias, offline)...")
    audit_3 = audit_default_pick_bias(rows)
    if audit_3["top10"]:
        top = audit_3["top10"][0]
        print(f"  most-frequent top-1: {top['code']} ({top['count']}, {top['pct']}%)")

    audit_2: dict | None = None
    if args.live:
        print("Running Audit 2 (per-cluster K=25 rank distribution, live)...")
        try:
            from atelier.classify.late_interaction_bridge import _resolve_qdrant_collection
            from atelier.config import load_config
        except ImportError as exc:
            print(f"ERROR: live mode requires atelier installed: {exc}", file=sys.stderr)
            return 2
        cfg = load_config()
        resolved = _resolve_qdrant_collection(cfg)
        if resolved is None:
            print(
                "  WARNING: no 'current' taxonomy collection in registry — "
                "Audit 2 skipped.  Re-run after promoting a taxonomy.",
                file=sys.stderr,
            )
        else:
            qdrant_url, collection = resolved
            print(f"  qdrant_url={qdrant_url!r} collection={collection!r}")
            error_clusters = [
                ("Internal Non-Sensitive", "System State"),
                ("Entity Identifiers", "Internal Non-Sensitive"),
                ("System State", "Internal Non-Sensitive"),
                ("Type", "Internal Non-Sensitive"),
                ("Transaction Timestamp", "Transaction Date"),
            ]
            audit_2 = audit_cluster_rank_distribution_live(
                rows, error_clusters, cfg, qdrant_url, collection,
            )
            for pat, res in audit_2.items():
                bkts = res["buckets"]
                print(
                    f"  {pat}: affected={res['affected']} "
                    f"rank_1={bkts['rank_1']} rank_2_3={bkts['rank_2_3']} "
                    f"rank_4_10={bkts['rank_4_10']} "
                    f"rank_11_25={bkts['rank_11_25']} "
                    f"not_in_top_k={bkts['not_in_top_k']}"
                )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "cosine_signal_audit.md"
    js_path = out_dir / "cosine_signal_audit.json"

    md_path.write_text(render_markdown(audit_1, audit_2, audit_3, run_id, len(rows)))
    js_path.write_text(json.dumps({
        "run_id": run_id,
        "n_classifications": len(rows),
        "audit_1_subtree_correctness": audit_1,
        "audit_2_cluster_rank_distribution": audit_2,
        "audit_3_default_pick_bias": audit_3,
    }, indent=2, default=str))

    print(f"\nWrote {md_path}")
    print(f"Wrote {js_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
