#!/usr/bin/env python
"""Phase 7 of /evolve-classification — render an operator-readable change
management guide for a transform apply cycle.

Reads:
  - manifest_path (Phase 5 output)
  - --verify <verify_*.json> (Phase 6 output, optional)
  - --audit <cosine_signal_audit.json> (Phase 3 output, optional)

Writes the guide as markdown alongside the manifest:
  build/data/transforms/manifests/<cohort>_<timestamp>.md

Four sections:
  1. What was applied        (transform list with diffs)
  2. Material deltas         (from Phase 6 verify, when present)
  3. Production state        (registry rows, Qdrant collections)
  4. Remediation options     (forward-only — corrective forward,
                              inverse-transform-as-forward, continue forward)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_transform_records(manifest: dict) -> list[dict]:
    records_dir = Path("build/data/transforms/records")
    out = []
    for tid in manifest.get("transform_ids") or []:
        rp = records_dir / f"{tid}.json"
        if rp.is_file():
            out.append(json.loads(rp.read_text()))
    return out


def diff_snippet(prior: str | None, new: str | None, n: int = 120) -> str:
    """Format an old→new diff for a single field, truncated."""
    p = (prior or "").strip().replace("\n", " ")
    nx = (new or "").strip().replace("\n", " ")
    if len(p) > n:
        p = p[:n] + "…"
    if len(nx) > n:
        nx = nx[:n] + "…"
    return f"`{p}` → `{nx}`"


def render_section_applied(manifest: dict, records: list[dict]) -> list[str]:
    lines = ["## What was applied", ""]
    counts = manifest.get("counts") or {}
    lines.append(
        f"- Cohort: `{manifest.get('cohort')}`"
    )
    lines.append(
        f"- Applied at: `{manifest.get('applied_at')}`"
    )
    lines.append(
        f"- Model: `{manifest.get('model')}`"
    )
    lines.append(
        f"- Source run: `{manifest.get('source_run')}`"
    )
    lines.append("")
    lines.append(
        f"- Candidates: {counts.get('candidates_total', '?')}  "
        f"Accepted: {counts.get('accepted', '?')}  "
        f"Rewritten: {counts.get('rewritten', '?')}  "
        f"Copied unchanged: {counts.get('copied_unchanged', '?')}  "
        f"Skipped (missing code): {counts.get('skipped_missing', '?')}"
    )
    lines.append("")
    lines.append("| code | mnemonic | status | conf | old→new (definition) |")
    lines.append("|---|---|---|---|---|")
    for r in records:
        tgt = r.get("target", {})
        old_desc = (r.get("prior_text") or {}).get("description")
        new_desc = (r.get("new_text") or {}).get("description")
        lines.append(
            f"| `{tgt.get('code')}` | `{tgt.get('mnemonic')}` | "
            f"{r.get('status')} | {r.get('confidence')} | "
            f"{diff_snippet(old_desc, new_desc)} |"
        )
    lines.append("")
    lines.append(
        f"Per-transform records under "
        f"`build/data/transforms/records/<transform_id>.json`."
    )
    lines.append("")
    return lines


def render_section_deltas(verify: dict | None) -> list[str]:
    lines = ["## Material deltas", ""]
    if verify is None:
        lines.append("*Phase 6 verification not run (or output not provided).*")
        lines.append("")
        return lines
    c = verify.get("counts") or {}
    lines.append(f"Verified against baseline run: `{verify.get('baseline_run')}`")
    lines.append(f"Top-K queried: {verify.get('k')}")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| affected columns | {c.get('n_columns_affected', '—')} |")
    lines.append(f"| cosine top-1 changed | {c.get('n_top1_changed', '—')} |")
    lines.append(f"| reference rank improved | {c.get('n_rank_improved', '—')} |")
    lines.append(f"| reference rank regressed | {c.get('n_rank_regressed', '—')} |")
    lines.append(f"| reference rank unchanged | {c.get('n_rank_unchanged', '—')} |")
    lines.append(f"| old top-1 == reference | {c.get('n_old_matches_ref', '—')} |")
    lines.append(f"| new top-1 == reference | {c.get('n_new_matches_ref', '—')} |")
    delta = c.get("subset_match_delta")
    lines.append(
        f"| **subset match Δ** | **{delta:+d}** "
        f"(net new top-1=ref on affected subset) |"
        if isinstance(delta, int) else
        f"| subset match Δ | — |"
    )
    lines.append("")
    lines.append(
        "*Caveat: this is a subset metric over the columns the transforms "
        "can plausibly move, not a whole-pipeline accuracy delta.  Full "
        "lift is measured by re-running the pipeline against the new "
        "`current` collection.*"
    )
    lines.append("")

    # Per-target movement
    movement = verify.get("per_target_code_movement") or {}
    if movement:
        lines.append("### Per-target movement")
        lines.append("")
        for code, dests in movement.items():
            lines.append(f"- `{code}`:")
            for dest, n in sorted(dests.items(), key=lambda kv: -kv[1])[:5]:
                lines.append(f"  - → `{dest}`: {n}")
        lines.append("")

    # Regression watch
    rw = verify.get("regression_watch") or []
    if rw:
        lines.append(f"### Regression watch ({len(rw)} columns)")
        lines.append("")
        lines.append("Columns that were correctly matched before the apply "
                     "and are no longer correctly matched.  These are "
                     "first-cut candidates for a corrective-forward cohort.")
        lines.append("")
        lines.append("| column | was | now |")
        lines.append("|---|---|---|")
        for r in rw[:20]:
            lines.append(
                f"| `{r['qkey']}` | `{r['old_top1']}` | `{r['new_top1']}` |"
            )
        if len(rw) > 20:
            lines.append(f"\n(showing 20 of {len(rw)})")
        lines.append("")
    else:
        lines.append("### Regression watch")
        lines.append("")
        lines.append("*No regressions on affected columns.*")
        lines.append("")
    return lines


def render_section_production_state(manifest: dict) -> list[str]:
    lines = ["## Production state", ""]
    try:
        from atelier.db.dao import AtelierDao
        dao = AtelierDao()
        rows = dao.list_taxonomy_collections(
            taxonomy_id="default", include_archived=True,
        )
    except Exception as exc:
        lines.append(
            f"*Could not query taxonomy_registry: {type(exc).__name__}: "
            f"{exc}.  This typically means the script is running outside "
            f"the App pod (PGlite not reachable on 127.0.0.1:5440).*"
        )
        lines.append("")
        lines.append(
            f"- Manifest declares target collection: "
            f"`{manifest.get('target_collection')}`"
        )
        lines.append(
            f"- Manifest declares target registry id: "
            f"`{manifest.get('target_registry_id')}`"
        )
        lines.append(
            f"- Manifest promoted flag: `{manifest.get('promoted')}`"
        )
        lines.append("")
        return lines

    lines.append("| status | collection | registry id | augmentation version | model |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r.get('status')} | `{r.get('qdrant_collection')}` | "
            f"`{(r.get('id') or '')[:8]}...` | "
            f"`{r.get('augmentation_version')}` | "
            f"`{r.get('embedding_model')}` |"
        )
    lines.append("")
    lines.append(
        "*Forward-only invariant: at most one `current` per taxonomy.  "
        "Previous `current` rows move to `stale` automatically via "
        "`set_current_taxonomy_collection`.  Nothing is archived or "
        "deleted by this loop.*"
    )
    lines.append("")
    return lines


def render_section_remediation(manifest: dict, records: list[dict],
                              verify: dict | None) -> list[str]:
    lines = ["## Remediation options (forward-only)", ""]
    lines.append(
        "Per the roll-forward design, all remediation is expressed as a "
        "*fresh forward action*.  No rollback machinery; we never undo a "
        "prior transform.  When the apply's outcome falls short of "
        "expectations, choose one of the three forward paths below."
    )
    lines.append("")

    # (a) Corrective forward
    rw = verify.get("regression_watch") if verify else None
    regression_codes = set()
    if rw:
        for r in rw:
            # The regressing column's old top-1 was a target_code we
            # transformed.  Mark that code as a candidate for exclusion
            # in the next cohort iteration.
            old_top1 = r.get("old_top1")
            if old_top1:
                regression_codes.add(old_top1)

    lines.append("### (a) Corrective forward — adjust acceptance, iterate")
    lines.append("")
    lines.append(
        "Re-run `/evolve-classification` for the next iteration with an "
        "acceptance file that excludes the regressing target codes.  The "
        "next cohort (`v(N+1)`) produces a fresh transform set that the "
        "apply step layers on top of the current collection."
    )
    if regression_codes:
        lines.append("")
        lines.append("Excluded codes derived from regression watch:")
        lines.append("```")
        lines.append(json.dumps({
            "_comment": "Place this at <cohort_dir>/acceptance.json",
            "exclude_codes": sorted(regression_codes),
        }, indent=2))
        lines.append("```")
    lines.append("")

    # (b) Inverse transform as forward
    lines.append("### (b) Inverse transform as forward — restore prior text")
    lines.append("")
    lines.append(
        "To restore the prior text for one or more transformed codes, "
        "synthesize a `cohort_revert_<source>_v1` whose `new_text` equals "
        "the manifest's `prior_text` blocks.  Apply via Phase 5 against "
        "the current collection.  Result: a new versioned collection in "
        "which those codes read the old text.  Intermediate collections "
        "stay `stale` and queryable — no data lost."
    )
    lines.append("")
    revert_records = [
        {
            "code": r["target"].get("code"),
            "annotation": r["target"].get("mnemonic"),
            "current_definition": (r.get("new_text") or {}).get("description"),
            "proposals": [{
                "new_definition": (r.get("prior_text") or {}).get("description"),
                "new_common_names": (r.get("prior_text") or {}).get("common_names"),
                "diagnosis": (
                    f"Inverse of transform {r.get('transform_id')} "
                    f"applied at {r.get('applied_at')}"
                ),
            }],
            "trace_summary": {"inverse_of": r.get("transform_id")},
        }
        for r in records
        if r.get("status") == "applied"
    ]
    if revert_records:
        lines.append("Ready-to-paste cohort skeleton (save as "
                     "`build/enrichment_evolution/cohort_revert_<source>_v1/"
                     "candidates.json`, then `python scripts/"
                     "apply_enrichment_transforms.py <that_dir>`):")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps({
            "cohort_name": f"revert_{manifest.get('cohort')}",
            "version": "v1",
            "model": "manual_inverse_synthesis",
            "run_id": manifest.get("source_run"),
            "candidates": revert_records[:5],  # show first 5 only
        }, indent=2)[:2400])
        if len(revert_records) > 5:
            lines.append(f"// ... + {len(revert_records) - 5} more inverse records")
        lines.append("```")
        lines.append("")

    # (c) Continue forward
    lines.append("### (c) Continue forward — layer additional refinements")
    lines.append("")
    lines.append(
        "Outcome was workable but not complete.  Run the next "
        "`/evolve-classification` iteration to layer additional refinements "
        "on top of the new current collection.  The `taxonomy_registry` "
        "preserves the lineage; replaying manifests against the immutable "
        "`default.annotations` reconstructs any historical state."
    )
    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("manifest_path",
        help="Path to build/data/transforms/manifests/<cohort>.json")
    parser.add_argument("--verify", default=None,
        help="Path to build/data/transforms/verify_<manifest_id>.json (optional)")
    parser.add_argument("--audit", default=None,
        help="Path to build/diag/cosine_signal_audit.json (optional)")
    parser.add_argument("--out", default=None,
        help="Override output path (default: alongside manifest as .md)")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    if not manifest_path.is_file():
        sys.exit(f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())

    verify = None
    if args.verify:
        vp = Path(args.verify)
        if vp.is_file():
            verify = json.loads(vp.read_text())

    records = load_transform_records(manifest)

    lines: list[str] = []
    lines.append(f"# Change management guide — {manifest.get('cohort')}")
    lines.append("")
    lines.append(f"Generated: `{_now_iso()}`")
    lines.append(f"Manifest:  `{manifest_path}`")
    if verify:
        lines.append(f"Verify:    `{args.verify}`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.extend(render_section_applied(manifest, records))
    lines.extend(render_section_deltas(verify))
    lines.extend(render_section_production_state(manifest))
    lines.extend(render_section_remediation(manifest, records, verify))

    body = "\n".join(lines) + "\n"

    out_path = Path(args.out) if args.out else manifest_path.with_suffix(".md")
    out_path.write_text(body)
    print(body)  # echo to stdout for the skill operator
    print(f"\nWrote guide → {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
