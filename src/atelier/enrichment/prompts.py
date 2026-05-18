# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""System + user prompt templates for annotation enrichment.

The enrichment generator produces a six-field structured payload per
taxonomy node.  Prompts come in two variants because the principle
that drives the architecture — *every node, leaf or internal, is a
first-class tagging target* — means parent and leaf nodes describe
different kinds of column.

A **leaf** prompt asks: "what would a column tagged exactly here
look like?"  Concrete value examples, narrow regex patterns, name
hints that uniquely imply this tag.

A **parent** prompt asks: "what does a column tagged at this
generality level — without further specificity to one of these
children — look like?"  Broader signals, with the children listed
so the model knows what specializations would NOT route here.

Both produce the same JSON schema (six fields) so downstream code
treats them identically.  The framing difference shapes the
generated content quality, not the structural payload.

Anti-examples are hierarchically aware in both variants: the
confusable_tag may be a sibling at the same level, but it may also
be a sibling of an ancestor (cross-level confusion).  The model is
explicitly invited to reach across levels because the late-
interaction architecture's anti-example evidence applies regardless
of where in the tree the confusable target lives.
"""

from __future__ import annotations

import json
from typing import Any

SCHEMA_DESCRIPTION = """\
You produce a single JSON object with these exact fields:

{
  "prototype_values": [<5-10 representative values a column tagged here would carry>],
  "value_patterns": [
    {"kind": "regex"|"format"|"length", "expr": "<pattern>"}
  ],
  "name_hints": [<column-name fragments that strongly imply this tag>],
  "anti_examples": [
    {"value": "<value that LOOKS like this tag but isn't>",
     "confusable_tag": "<the tag it actually IS>",
     "reason": "<short explanation of the confusion>"}
  ],
  "parent_path": [<root>, <intermediate>, ..., <this tag's parent label>]
}

Hard rules:
- Output ONLY the JSON object. No prose before or after. No code fences.
- Every regex pattern MUST be valid Python re-compilable syntax.
- prototype_values MUST match at least one declared regex pattern.
- anti_examples.confusable_tag MUST reference a tag from the provided taxonomy.
- parent_path MUST match the taxonomy hierarchy exactly (root → ancestors → parent).
- A prototype value MUST NOT appear in anti_examples.value (no self-contradiction).
"""

LEAF_FRAMING = """\
You are generating enrichment for a LEAF tag in a hierarchical
data-governance taxonomy.  A column tagged here is tagged at
maximum specificity — no further child tag would refine the
classification.  Your enrichment should describe what THIS specific
tag's columns look like, with patterns narrow enough to discriminate
against sibling leaves and cousin leaves under shared ancestors.
"""

PARENT_FRAMING = """\
You are generating enrichment for an INTERNAL (parent) tag in a
hierarchical data-governance taxonomy.  A column tagged here is
tagged at this level of generality WITHOUT being specialized to one
of this tag's children — the operator's evidence was sufficient to
place the column in this family but not specific enough to choose a
child.  Your enrichment should describe what columns tagged AT THIS
LEVEL (not at the children's level) look like: broader patterns,
name hints that signal the family without singling out a specialty,
and anti-examples that distinguish this family from its siblings
and from its own children's tighter signals.
"""


def build_system_prompt(*, is_leaf: bool) -> str:
    """System prompt for one enrichment generation.

    Composes the framing (leaf vs parent) with the schema description.
    """
    framing = LEAF_FRAMING if is_leaf else PARENT_FRAMING
    return framing + "\n" + SCHEMA_DESCRIPTION


def build_user_prompt(
    *,
    source_row: dict[str, Any],
    taxonomy_context: dict[str, Any],
    is_leaf: bool,
    prior_attempt_json: str | None = None,
    verifier_feedback: dict[str, Any] | None = None,
) -> str:
    """User prompt with source-row + taxonomy context.

    When ``prior_attempt_json`` and ``verifier_feedback`` are non-None,
    appends a refinement section that tells the model which verifier
    checks failed and asks for a corrected JSON.  The schema and
    framing in the system prompt remain unchanged so the model
    rebuilds against the same target.
    """
    sections: list[str] = []

    code = source_row.get("code", "")
    label = source_row.get("label", "")
    mnemonic = source_row.get("mnemonic") or source_row.get("abbrev") or ""
    description = source_row.get("description", "")
    parent_code = source_row.get("parent_code", "")
    common_names = source_row.get("common_names", "")

    sections.append(
        "Tag under enrichment:\n"
        f"  code: {code}\n"
        f"  label: {label}\n"
        f"  mnemonic: {mnemonic}\n"
        f"  description: {description}\n"
        f"  parent_code: {parent_code}\n"
        f"  common_names: {common_names}\n"
        f"  kind: {'leaf' if is_leaf else 'parent (internal)'}"
    )

    expected_parent_path = taxonomy_context.get("expected_parent_path")
    if expected_parent_path:
        sections.append(
            "Required parent_path (use this exact list in the response):\n  "
            + json.dumps(expected_parent_path)
        )

    if not is_leaf:
        children = taxonomy_context.get("children") or []
        if children:
            child_lines = "\n".join(
                f"  - {c.get('code', '?')}: {c.get('label', '')}"
                for c in children[:30]
            )
            sections.append(
                "This parent tag's children (a column specialized to ANY of "
                "these would NOT receive this parent tag; your enrichment "
                "should NOT describe values better suited to a child):\n"
                + child_lines
            )

    siblings = taxonomy_context.get("siblings") or []
    if siblings:
        sib_lines = "\n".join(
            f"  - {s.get('code', '?')}: {s.get('label', '')}"
            for s in siblings[:20]
        )
        sections.append(
            "Sibling tags under the same parent (use these as confusable "
            "candidates in anti_examples when the values are genuinely "
            "confusable):\n" + sib_lines
        )

    cousins = taxonomy_context.get("cross_level_confusables") or []
    if cousins:
        cousin_lines = "\n".join(
            f"  - {c.get('code', '?')}: {c.get('label', '')}"
            for c in cousins[:20]
        )
        sections.append(
            "Cross-level confusable candidates (tags at other depths that "
            "carry visually-similar values to this one — use in "
            "anti_examples when applicable):\n" + cousin_lines
        )

    valid_codes = taxonomy_context.get("valid_tag_codes")
    if valid_codes:
        # Show the full valid set so the model never hallucinates codes.
        # For typical taxonomies (≤500 codes with short dotted IDs) this
        # adds ~2-3K chars — well within prompt budget.
        all_sorted = sorted(valid_codes)
        codes_str = ", ".join(all_sorted)
        sections.append(
            "Valid taxonomy codes (any anti_examples.confusable_tag MUST "
            f"appear in this set — {len(all_sorted)} total):\n  {codes_str}"
        )

    if prior_attempt_json and verifier_feedback:
        failed_checks = verifier_feedback.get("failed_checks") or []
        check_summary = "\n".join(
            f"  - {c.get('name', '?')}: {c.get('detail', '')}"
            for c in failed_checks
        )
        sections.append(
            "Your previous attempt failed these verifier checks:\n"
            + check_summary
            + "\n\nPrevious attempt:\n"
            + prior_attempt_json
            + "\n\nProduce a corrected JSON object that fixes these "
            "specific failures while keeping the rest of the structure "
            "intact."
        )
    else:
        sections.append(
            "Produce the JSON object now.  Output ONLY the JSON, no prose."
        )

    return "\n\n".join(sections)
