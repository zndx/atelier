"""CLI dispatcher for governance operations.

Reachable via ``python -m atelier.governance <subcommand>`` (see ``__main__.py``)
or the ``just sync-taxonomy`` recipe.

Subcommands:
    taxonomy    Push ``default.annotations`` to Atlas as classification typedefs.

Reserved names: ``glossary``, ``tag-results``, ``derive-policies``, ``ping``.
"""

from __future__ import annotations

import argparse
import logging
import sys

log = logging.getLogger("atelier.governance.cli")


def _cmd_taxonomy(args: argparse.Namespace) -> int:
    from atelier.config import load_config
    from atelier.classify.taxonomy import (
        load_annotations_from_hive,
        HierarchicalCategorySet,
    )
    from atelier.governance.client import GovernanceClient
    from atelier.governance.sync import TaxonomyNode, sync_taxonomy_to_atlas

    cfg = load_config()
    if not cfg.has_atlas:
        print(
            "ERROR: Atlas not configured. Set ATELIER_ATLAS_URL/USER/PASSWORD "
            "(see config/base.conf:494-501) and re-run.",
            file=sys.stderr,
        )
        return 2

    gc = GovernanceClient.from_atelier_config(cfg)

    ping = gc.atlas.ping()
    if not ping.get("ok"):
        print(
            f"ERROR: Atlas not reachable at {cfg.governance_atlas_url}: "
            f"{ping.get('error', 'unknown')}",
            file=sys.stderr,
        )
        return 2

    print(
        f"Atlas ok: classifications={ping.get('classification_count')}, "
        f"entities={ping.get('entity_type_count')}"
    )
    if args.ping_only:
        return 0

    connection = args.connection or cfg.classify_connection_name or None
    try:
        # hierarchical=False intentional: Atlas's classification-type model is
        # flat (one entity-tag per category, no parent/child relationship).
        # This CLI emits TaxonomyNode records for Atlas sync, not for training.
        # Atelier's classifiers are still hierarchical — see
        # feedback_hierarchical_svm_only.md.
        cset = load_annotations_from_hive(
            cfg,
            connection_name=connection,
            database=cfg.classify_database or "default",
            hierarchical=False,
        )
    except (ValueError, RuntimeError) as e:
        print(f"ERROR: load_annotations_from_hive failed: {e}", file=sys.stderr)
        return 3

    nodes: list[TaxonomyNode] = []
    seen: set[str] = set()
    skipped_empty = 0
    skipped_dup = 0
    categories = (
        cset.all_categories
        if isinstance(cset, HierarchicalCategorySet)
        else cset.categories
    )
    for cat in categories:
        abbrev = (cat.abbrev or "").strip()
        if not abbrev:
            skipped_empty += 1
            continue
        if abbrev in seen:
            log.warning("Duplicate annotation %r — keeping first", abbrev)
            skipped_dup += 1
            continue
        seen.add(abbrev)
        definition = (cat.description or "").strip()[:500]
        nodes.append(TaxonomyNode(
            code=abbrev,
            label=cat.label or abbrev,
            notation=(cat.notation or "").strip(),
            parent_code="",
            description=definition,
        ))

    if not nodes:
        print(
            "ERROR: no usable rows in default.annotations (Annotation column empty?)",
            file=sys.stderr,
        )
        return 4

    print(
        f"Loaded {len(nodes)} annotations "
        f"(skipped {skipped_empty} empty, {skipped_dup} duplicate)"
    )

    report = sync_taxonomy_to_atlas(gc.atlas, nodes, dry_run=args.dry_run)
    suffix = " (dry run)" if args.dry_run else ""
    print(
        f"created={len(report.created)} skipped={len(report.skipped)} "
        f"failed={len(report.failed)}{suffix}"
    )
    if report.failed:
        for name in report.failed:
            print(f"  failed: {name}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="python -m atelier.governance",
        description="Atelier governance CLI (Atlas, Ranger).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tax = sub.add_parser(
        "taxonomy",
        help="Push default.annotations to Atlas as classification typedefs.",
    )
    p_tax.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing to Atlas.",
    )
    p_tax.add_argument(
        "--connection",
        default=None,
        help="CAI Data Connection name (defaults to classify.connection_name).",
    )
    p_tax.add_argument(
        "--ping-only",
        action="store_true",
        help="Probe Atlas connectivity and exit; do not load annotations.",
    )
    p_tax.set_defaults(func=_cmd_taxonomy)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
