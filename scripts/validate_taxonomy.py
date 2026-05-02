#!/usr/bin/env python3
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Standalone taxonomy validator — vocabulary-team CI / data cleanup tool.

Runs ``atelier.classify.taxonomy.validate_taxonomy`` against any
annotations source and prints a YAML-shaped report of the findings
(label collisions, duplicate codes, orphaned aliases).  Useful for
catching vocabulary quality issues before they manifest as
non-deterministic name-match resolution downstream.

Usage::

    uv run python scripts/validate_taxonomy.py                      # OOTB sample
    uv run python scripts/validate_taxonomy.py --source universal   # universal vocab
    uv run python scripts/validate_taxonomy.py --json /path/to/annotations.json
    uv run python scripts/validate_taxonomy.py --hive default       # from hive

Exit codes:
    0 — no findings (vocabulary is clean)
    1 — warnings only (label collisions, orphaned aliases)
    2 — errors present (duplicate codes; structural issue)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an Atelier taxonomy / annotations source",
    )
    parser.add_argument(
        "--source",
        default="sample",
        choices=("sample", "universal"),
        help="built-in source to validate (default: sample)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="path to an annotations JSON file (overrides --source)",
    )
    parser.add_argument(
        "--hive",
        metavar="DATABASE",
        help="hive database to load (loads from cml.data_v1 default connection)",
    )
    args = parser.parse_args()

    from atelier.classify.taxonomy import (
        validate_taxonomy,
        load_sample_vocabulary,
        load_universal_vocabulary,
        load_annotations_from_json,
        load_annotations_from_hive,
    )

    if args.json:
        category_set = load_annotations_from_json(args.json, hierarchical=True)
        source_label = f"json:{args.json}"
    elif args.hive:
        category_set = load_annotations_from_hive(args.hive)
        source_label = f"hive:{args.hive}"
    elif args.source == "universal":
        category_set = load_universal_vocabulary(hierarchical=True)
        source_label = "universal"
    else:
        category_set = load_sample_vocabulary(hierarchical=True)
        source_label = "sample"

    findings = validate_taxonomy(category_set)

    print(f"source: {source_label}")
    print(f"categories: {len(category_set.categories)}")
    print(f"findings: {len(findings)}")
    print("---")

    if not findings:
        print("status: clean")
        return 0

    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warning"]
    print(f"errors: {len(errors)}")
    print(f"warnings: {len(warnings)}")
    print()

    by_kind: dict[str, list] = {}
    for f in findings:
        by_kind.setdefault(f.kind, []).append(f)

    for kind, group in sorted(by_kind.items()):
        print(f"{kind}:")
        for f in group:
            codes = ", ".join(f.codes)
            print(f"  - severity: {f.severity}")
            print(f"    codes: [{codes}]")
            print(f"    detail: {f.detail}")
        print()

    return 2 if errors else 1


if __name__ == "__main__":
    sys.exit(main())
