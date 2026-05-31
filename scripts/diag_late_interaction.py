#!/usr/bin/env python
"""Live diagnosis for late_interaction bridge failures.

Run this *inside the App pod* (where PGlite on :5440 and Qdrant on
:6333 are reachable) to capture the actual exception the bridge
raises that the pipeline currently catches as 'degraded_bridge_error'.

Two execution paths:

1. **From the Web Terminal Agent in the Atelier UI** — just paste
   ``python scripts/diag_late_interaction.py`` into the terminal.

2. **From a session where you can curl the gateway** — call
   ``GET /api/diag/late-interaction`` (not wired by default; left as
   the option of last resort).

The script writes a step-by-step probe report to stdout and to
``build/diag/late_interaction.json``.  Every step is independently
``try``-wrapped so a failure early on does not hide later signals.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, "src")

report: dict = {"steps": []}


def step(name: str, fn):
    entry = {"step": name}
    try:
        result = fn()
        entry["status"] = "ok"
        entry["result"] = result
        print(f"  OK  {name}: {str(result)[:200]}")
    except BaseException as exc:
        entry["status"] = "FAIL"
        entry["exception_type"] = type(exc).__name__
        entry["exception_repr"] = repr(exc)
        entry["traceback"] = traceback.format_exc()
        print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
        print(f"       {entry['traceback']}")
    report["steps"].append(entry)
    return entry


# ── Environment ──────────────────────────────────────────────────
print("=== Environment ===")
step("ATELIER_DB_URL", lambda: os.environ.get("ATELIER_DB_URL", "<unset>"))
step("PWD", lambda: os.getcwd())

# ── Imports ──────────────────────────────────────────────────────
print("\n=== Bridge imports ===")
step("import bridge", lambda: __import__(
    "atelier.classify.late_interaction_bridge",
    fromlist=["try_compute_cosine_mass", "is_enabled",
              "_resolve_qdrant_collection", "_auto_promote_latest"],
).__name__)

from atelier.classify.late_interaction_bridge import (  # noqa: E402
    try_compute_cosine_mass, is_enabled, _resolve_qdrant_collection,
    _auto_promote_latest,
)

# ── Config probe ─────────────────────────────────────────────────
print("\n=== Config flag ===")
from atelier.config import load_config  # noqa: E402
cfg = load_config()
step("is_enabled(cfg)", lambda: is_enabled(cfg))
step("cfg.classify_cosine_late_interaction_enabled",
     lambda: getattr(cfg, "classify_cosine_late_interaction_enabled", "<missing>"))
step("cfg.classify_taxonomy_id",
     lambda: getattr(cfg, "classify_taxonomy_id", "<missing>"))

# ── DAO probe ────────────────────────────────────────────────────
print("\n=== DAO + taxonomy_registry ===")
from atelier.db.dao import AtelierDao  # noqa: E402
dao = AtelierDao()
step("dao.list_taxonomy_collections('default')",
     lambda: dao.list_taxonomy_collections(taxonomy_id="default"))
step("dao.get_current_taxonomy_collection('default')",
     lambda: dao.get_current_taxonomy_collection("default"))

# ── Bridge internal resolution ──────────────────────────────────
print("\n=== Bridge collection resolution ===")
step("_resolve_qdrant_collection(cfg)", lambda: _resolve_qdrant_collection(cfg))

# ── ColBERT encoder load ────────────────────────────────────────
print("\n=== ColBERT encoder ===")
from atelier.classify.colbert_encoder import get_encoder  # noqa: E402
step("get_encoder()", lambda: type(get_encoder()).__name__)
step("encode_single", lambda: list(get_encoder().encode_single("user id integer").shape))

# ── Live Qdrant connect ─────────────────────────────────────────
print("\n=== Qdrant ===")
from qdrant_client import QdrantClient  # noqa: E402
step("qdrant collections list", lambda: [c.name for c in QdrantClient(
    url="http://127.0.0.1:6333"
).get_collections().collections])

# ── End-to-end bridge call ──────────────────────────────────────
print("\n=== End-to-end bridge call ===")


def _live_bridge_call():
    """Reproduce the pipeline's bridge invocation against real Qdrant.

    Builds the frame the cheap way — read codes straight out of the
    target Qdrant collection's first 200 payloads, hand them to a
    minimal HierarchicalCategorySet stand-in.  Good enough for the
    bridge to satisfy ``code in frame.singletons``.
    """
    from atelier.classify.belief import FrameOfDiscernment
    from atelier.classify.taxonomy import HierarchicalCategorySet, ReferenceCategory
    from qdrant_client import QdrantClient

    resolved = _resolve_qdrant_collection(cfg)
    if resolved is None:
        raise RuntimeError("No Qdrant collection resolved")
    qdrant_url, collection = resolved
    client = QdrantClient(url=qdrant_url or "http://127.0.0.1:6333")

    # Pull a sample of payloads so we have a few codes for the frame.
    scroll_result = client.scroll(
        collection_name=collection,
        limit=200, with_payload=True, with_vectors=False,
    )
    points = scroll_result[0]
    codes = sorted({(p.payload or {}).get("code")
                    for p in points if (p.payload or {}).get("code")})
    # ReferenceCategory requires ``embedding_text`` since the late-2026
    # signature change; the diag frame doesn't actually need realistic
    # text for the bridge call to exercise Qdrant — only ``code`` is
    # consulted via ``frame.singletons``/``frame.internal_nodes``.
    cats = [ReferenceCategory(code=c, label=c, embedding_text=c) for c in codes]
    cs = HierarchicalCategorySet(name="diag", categories=cats)
    frame = FrameOfDiscernment(cs)

    class Features:
        sample_values = ["1", "2", "3"]
        pattern_summary = None
        def to_embedding_text(self):
            return ("row id | int64 | 1, 2, 3 | cardinality=1000 | "
                    "avg_len=1.0 | numeric=1.00")

    out = try_compute_cosine_mass(
        cfg=cfg, column_features=Features(),
        column_name="row_id", table_name="academic_records",
        samples=["1", "2", "3"],
        neighbor_column_names=None, pattern_summary=None,
        frame=frame, embed=None,
    )
    return [type(out[0]).__name__ if out[0] is not None else None,
            out[1], type(out[2]).__name__ if out[2] is not None else None]


step("live bridge call", _live_bridge_call)

# ── Dump ────────────────────────────────────────────────────────
out_path = Path("build/diag/late_interaction.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    json.dump(report, f, indent=2, default=str)

print(f"\nWrote {out_path}")
fails = [s for s in report["steps"] if s["status"] == "FAIL"]
print(f"\n{len(fails)} step(s) failed.")
sys.exit(1 if fails else 0)
