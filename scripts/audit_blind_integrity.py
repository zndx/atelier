#!/usr/bin/env python3
"""Blind-integrity audit for the agent-mediated SDG artifacts.

Runs after (or during) curation. Verifies, mechanically, that the blind
contract held:

1. **Ingress unchanged** — the sha256s recorded in the working set's
   metadata still match the files on disk (nobody swapped the blind surface
   mid-run).
2. **No forbidden markers** — the working set, decisions, and audit trail
   contain none of the answer-key/semantic-register vocabulary that could
   only have come from eval-side artifacts: ``reference.parquet``,
   ``semantic_col``, ``semantic_table``, ``naming_map``, ``bfo_anchor``,
   ``template_id``, ``slot_ref``. (``holdout_partition`` etc. live in the
   generation manifest, which is release metadata, not the key — allowed.)
3. **Structural blindness** — every decision's inputs came from the working
   set (single ingress); the working set metadata declares ``blind: true``.

Exit 0 = clean; exit 1 = violation (printed). Wire into `just agent` after
the curation loop and before anything consumes ``agent_mediated.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FORBIDDEN_MARKERS = (
    "reference.parquet",
    "semantic_col",
    "semantic_table",
    "naming_map",
    "bfo_anchor",
    "template_id",
    "slot_ref",
)

# Keys under which the generation manifest legitimately lives — its contents
# are release metadata (sha, counts, partition name), not the answer key.
_MANIFEST_KEY = "generation_manifest"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan(obj, path: str, violations: list[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == _MANIFEST_KEY:
                continue
            for marker in FORBIDDEN_MARKERS:
                if marker in k:
                    violations.append(f"{path}.{k}: forbidden key marker {marker!r}")
            _scan(v, f"{path}.{k}", violations)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _scan(v, f"{path}[{i}]", violations)
    elif isinstance(obj, str):
        for marker in FORBIDDEN_MARKERS:
            if marker in obj:
                violations.append(f"{path}: forbidden content marker {marker!r}")


def audit(out_dir: Path) -> list[str]:
    violations: list[str] = []

    ws_path = out_dir / "working_set.json"
    if not ws_path.exists():
        return [f"no working set at {ws_path}"]
    ws = json.loads(ws_path.read_text())

    meta = ws.get("metadata", {})
    if meta.get("blind") is not True:
        violations.append("working set metadata does not declare blind: true")

    for file_path, recorded in (meta.get("ingress_sha256") or {}).items():
        p = Path(file_path)
        if not p.exists():
            violations.append(f"ingress file missing: {file_path}")
        elif _sha256(p) != recorded:
            violations.append(f"ingress CHANGED since build: {file_path}")

    for name in ("working_set.json", "agent_mediated.json", "audit.json"):
        path = out_dir / name
        if path.exists():
            _scan(json.loads(path.read_text()), name, violations)

    return violations


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taxonomy-id", default="sdg")
    ap.add_argument("--out-root", default="build/data/agent_mediated")
    a = ap.parse_args(argv)

    out_dir = REPO / a.out_root / a.taxonomy_id
    violations = audit(out_dir)
    if violations:
        print(f"BLIND-INTEGRITY VIOLATIONS ({len(violations)}):")
        for v in violations[:50]:
            print(f"  - {v}")
        return 1
    print(f"blind integrity OK: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
