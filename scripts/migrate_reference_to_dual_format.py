#!/usr/bin/env python
"""Migrate agent_mediated.json from mnemonic-only to dual format.

Per the no-silent-structural-drift policy: persistent taxonomy
references must store both the mnemonic AND the hierarchical code so
that ontology motion (curator moves a mnemonic to a different code or
subtree) is detectable at load time rather than silently shifting the
artifact's meaning.

This is a one-shot migration that:

* Reads ``build/data/agent_mediated/agent_mediated.json`` (any shape:
  legacy mnemonic-only, legacy nested-by-table, or already-dual).
* Resolves each mnemonic against the cached taxonomy
  (``build/data/taxonomy/taxonomy_cache.json``).
* Emits dual-format entries:
    {"mnemonic": "EMAIL", "code": "1.1.1.9.3.1",
     "captured_at": "<ISO>", "source": "<provenance>"}
* Snapshots the original at ``agent_mediated.json.bak.pre-dual``.
* Writes the migrated reference + a migration report listing any
  mnemonics that didn't resolve (vocabulary gaps that need attention).

Idempotent: already-dual entries are passed through with refreshed
``captured_at`` only if ``--refresh-captured`` is set; otherwise left
alone.  Legacy entries are upgraded.

Usage:
    python scripts/migrate_reference_to_dual_format.py
    python scripts/migrate_reference_to_dual_format.py --refresh-captured
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from update_reference_from_xlsx import (  # noqa: E402
    REFERENCE_PATH, build_code_indexes, load_taxonomy,
)


def _is_dual(v: object) -> bool:
    return isinstance(v, dict) and ("mnemonic" in v or "code" in v)


def migrate_entry(
    qkey: str, v: object, by_mnem: dict, now_iso: str, source: str,
    refresh_captured: bool,
) -> tuple[dict | None, str | None]:
    """Convert a single entry to dual format.  Returns (entry, unresolved_mnem)."""
    if _is_dual(v):
        if refresh_captured:
            mn = v.get("mnemonic")
            if mn:
                entry = by_mnem.get(mn.strip().upper())
                if entry:
                    return (
                        {
                            "mnemonic": mn,
                            "code": entry["code"],
                            "captured_at": now_iso,
                            "source": v.get("source") or source,
                        },
                        None,
                    )
                return (v, mn)
        return (v, None)
    if isinstance(v, str) and v:
        mn = v.strip()
        entry = by_mnem.get(mn.upper())
        if entry is None:
            return (None, mn)
        return (
            {
                "mnemonic": mn,
                "code": entry["code"],
                "captured_at": now_iso,
                "source": source,
            },
            None,
        )
    return (None, None)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--refresh-captured", action="store_true",
        help="Re-resolve mnemonics for already-dual entries (refreshes "
             "captured codes against current taxonomy).  Use when "
             "intentionally accepting the current taxonomy snapshot.",
    )
    parser.add_argument(
        "--source", default="legacy_migration",
        help="Provenance tag stored on each migrated entry "
             "(default: legacy_migration)",
    )
    parser.add_argument(
        "--reference", default=str(REFERENCE_PATH),
        help=f"Reference path (default: {REFERENCE_PATH})",
    )
    args = parser.parse_args()

    ref_path = Path(args.reference)
    if not ref_path.exists():
        sys.exit(f"Reference not found: {ref_path}")

    print(f"Loading taxonomy cache...")
    taxonomy = load_taxonomy()
    _, by_mnem, _ = build_code_indexes(taxonomy)
    print(f"  {len(taxonomy)} entries, {len(by_mnem)} mnemonics indexed")

    print(f"Loading reference {ref_path}...")
    with open(ref_path) as f:
        raw = json.load(f)
    print(f"  {len(raw)} top-level entries")

    now_iso = datetime.now(timezone.utc).isoformat()
    migrated: dict = {}
    legacy_string_count = 0
    legacy_nested_count = 0
    already_dual_count = 0
    refreshed_count = 0
    unresolved: dict[str, list[str]] = {}  # mnemonic → list of qkeys
    skipped_empty = 0

    for k, v in raw.items():
        # Legacy nested {table: {col: mnemonic}}
        if isinstance(v, dict) and not _is_dual(v):
            legacy_nested_count += 1
            for col, ann in v.items():
                qkey = f"{k}.{col}"
                entry, unres = migrate_entry(
                    qkey, ann, by_mnem, now_iso, args.source,
                    args.refresh_captured,
                )
                if entry is None:
                    if unres:
                        unresolved.setdefault(unres, []).append(qkey)
                    else:
                        skipped_empty += 1
                    continue
                migrated[qkey] = entry
            continue

        # Dual-format or legacy string
        was_dual = _is_dual(v)
        entry, unres = migrate_entry(
            k, v, by_mnem, now_iso, args.source, args.refresh_captured,
        )
        if entry is None:
            if unres:
                unresolved.setdefault(unres, []).append(k)
            else:
                skipped_empty += 1
            continue
        migrated[k] = entry
        if was_dual:
            if args.refresh_captured:
                refreshed_count += 1
            else:
                already_dual_count += 1
        else:
            legacy_string_count += 1

    # Stats
    print(f"\nMigration summary:")
    print(f"  Legacy mnemonic-only entries → dual: {legacy_string_count}")
    print(f"  Legacy nested-table entries → dual:  {legacy_nested_count} "
          f"(flattened to {sum(len(v) for v in raw.values() if isinstance(v, dict) and not _is_dual(v))} qkeys)")
    print(f"  Already-dual passed through:         {already_dual_count}")
    print(f"  Already-dual refreshed:              {refreshed_count}")
    print(f"  Empty entries skipped:               {skipped_empty}")
    print(f"  Unresolved mnemonics:                {len(unresolved)} "
          f"(affecting {sum(len(qks) for qks in unresolved.values())} qkeys)")

    # Sanity: dimensions match expectation
    print(f"\n  Total migrated entries written: {len(migrated)}")

    # Snapshot + write
    bak_path = ref_path.with_suffix(".json.bak.pre-dual")
    shutil.copy2(ref_path, bak_path)
    print(f"\n  Snapshot → {bak_path}")
    ref_path.write_text(json.dumps(migrated, indent=2, default=str))
    print(f"  Wrote migrated reference → {ref_path}")

    # Migration report
    report_path = ref_path.parent / "migration_report.md"
    lines = [
        "# Agent-mediated reference migration to dual format",
        "",
        f"- Migrated at: `{now_iso}`",
        f"- Source artifact: `{ref_path}`",
        f"- Pre-migration snapshot: `{bak_path}`",
        f"- Taxonomy cache: `build/data/taxonomy/taxonomy_cache.json`",
        f"- Provenance tag: `{args.source}`",
        "",
        "## Counts",
        "",
        f"| category | count |",
        f"|---|---:|",
        f"| Legacy mnemonic-only → dual | {legacy_string_count} |",
        f"| Legacy nested-by-table → dual | {legacy_nested_count} |",
        f"| Already-dual passed through | {already_dual_count} |",
        f"| Already-dual refreshed | {refreshed_count} |",
        f"| Empty/skipped | {skipped_empty} |",
        f"| Unresolved mnemonics (kept-out) | {len(unresolved)} |",
        f"| **Total migrated entries** | **{len(migrated)}** |",
        "",
    ]
    if unresolved:
        lines.append("## Unresolved mnemonics")
        lines.append("")
        lines.append(
            "These mnemonics appear in the reference but do not resolve "
            "against the cached taxonomy.  Likely causes: vocabulary "
            "drift, mnemonic renamed by curator, mnemonic from a "
            "different annotations source.  Affected entries were "
            "**kept out** of the migrated output — investigate and "
            "either fix the mnemonic in the source or update the "
            "taxonomy cache."
        )
        lines.append("")
        lines.append("| mnemonic | affected qkeys |")
        lines.append("|---|---|")
        for mn, qks in sorted(unresolved.items()):
            sample = ", ".join(f"`{q}`" for q in qks[:5])
            more = f" (+ {len(qks)-5} more)" if len(qks) > 5 else ""
            lines.append(f"| `{mn}` | {sample}{more} |")
        lines.append("")
    report_path.write_text("\n".join(lines))
    print(f"  Wrote {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
