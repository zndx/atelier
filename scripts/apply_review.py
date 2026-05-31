#!/usr/bin/env python
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Merge per-table review decisions into agent-mediated reference artifacts.

This is the persistence helper for the agent-mediated reference review workflow
(see ``project_agent_mediated_principles.md``).  The reviewer (this
session) produces a per-table decisions JSON; this script validates
it against the vocabulary + working_set and merges it into:

  * ``build/data/agent_mediated/agent_mediated.json``  — flat {table.col: tag}
  * ``build/data/agent_mediated/audit.json``            — per-column reasoning
  * ``build/data/agent_mediated/review_state.json``     — per-table progress

Resume-safe: existing entries are merged, not overwritten, unless
``--force`` is passed.

Decisions input format (passed via ``--decisions-file`` or stdin)
----------------------------------------------------------------

::

  {
    "table_name": "ecommerce_orders",
    "reviewed_at": "2026-05-15T...",
    "table_notes": "Optional notes about table-level patterns.",
    "decisions": {
      "row_id": {
        "tag": "INOS",
        "confidence": "high",        # high | medium | low | unsure
        "reasoning": "Sequential int surrogate key; no PII signal.",
        "sources": {
          "atelier_current": "INOS",   # what 981f69be said (may be null)
          "atelier_xlsx":    "INOS",   # what 522d89ae said
          "llm_solution":    null      # what the comparison LLM said (may be null)
        },
        "needs_operator_review": false
      },
      ...
    }
  }

Validation rules
----------------

* ``table_name`` must exist in working_set.json.
* Every column-name in ``decisions`` must exist in working_set for
  this table.  Spurious column names are rejected.
* By default every column in working_set for this table must be
  decided (no partial submissions); pass ``--allow-partial`` to
  override.
* ``tag`` must be one of: a key in the vocabulary, or the literal
  ``null`` (which produces a ``null`` GT entry that ``score_sweep.py``
  skips at scoring time but preserves in the diff CSV).
* ``confidence`` must be one of: high, medium, low, unsure.

Usage
-----

::

  cat decisions.json | python scripts/apply_review.py
  python scripts/apply_review.py --decisions-file decisions.json
  python scripts/apply_review.py --decisions-file decisions.json --force
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

VALID_CONFIDENCE = {"high", "medium", "low", "unsure"}
ROOT = Path("build/data/agent_mediated")
ARCHIVE = ROOT / "archive"
WORKING_SET = ROOT / "working_set.json"
AGENT_MEDIATED = ROOT / "agent_mediated.json"
AUDIT = ROOT / "audit.json"
REVIEW_STATE = ROOT / "review_state.json"


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _archive_artifacts() -> Path | None:
    """Copy current agent_mediated.json + audit.json to archive/<iso-date>_.

    Runs once per calendar day; subsequent calls on the same day are
    no-ops (the first snapshot of the day is the one that matters).
    Returns the archive directory, or None if nothing to archive.
    """
    if not AGENT_MEDIATED.exists():
        return None
    date_tag = _utc_date()
    dest = ARCHIVE / date_tag
    sentinel = dest / f"{date_tag}_agent_mediated.json"
    if sentinel.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    import shutil
    for src in (AGENT_MEDIATED, AUDIT, REVIEW_STATE):
        if src.exists():
            shutil.copy2(src, dest / f"{date_tag}_{src.name}")
    print(f"Archived pre-update snapshot → {dest}", file=sys.stderr)
    return dest


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _classify_agreement(decision_tag, sources: dict) -> str:
    """Summarize how the decision relates to the input signals."""
    s = {k: v for k, v in sources.items() if v}
    if not s:
        return "no_signals"
    matches = {k for k, v in s.items() if v == decision_tag}
    if len(matches) == len(s) and matches:
        return "all_agree"
    if matches:
        return "partial_agree"
    return "disagree_all"


def _validate_decisions(payload: dict, working_set: dict, *, allow_partial: bool):
    if "table_name" not in payload or "decisions" not in payload:
        raise ValueError("Payload missing required keys: table_name, decisions")
    table_name = payload["table_name"]
    decisions = payload["decisions"]
    if not isinstance(decisions, dict):
        raise ValueError("decisions must be an object")

    vocab = set(working_set["vocabulary"].keys())
    ws_table_cols = {
        c["column_name"] for k, c in working_set["columns"].items()
        if c["table_name"] == table_name
    }
    if not ws_table_cols:
        raise ValueError(
            f"Table {table_name!r} not found in working_set. "
            f"Available tables: {sorted({c['table_name'] for c in working_set['columns'].values()})}"
        )

    dec_cols = set(decisions.keys())
    spurious = dec_cols - ws_table_cols
    missing = ws_table_cols - dec_cols
    if spurious:
        raise ValueError(
            f"Decisions for columns not in working_set: {sorted(spurious)}"
        )
    if missing and not allow_partial:
        raise ValueError(
            f"Missing decisions for columns: {sorted(missing)}.  "
            f"Pass --allow-partial to submit a partial table."
        )

    for col, d in decisions.items():
        if not isinstance(d, dict):
            raise ValueError(f"Decision for {col} must be an object")
        tag = d.get("tag")
        if tag is not None and tag not in vocab:
            raise ValueError(
                f"{col}: tag {tag!r} is not in vocabulary "
                f"(showed {len(vocab)} entries; use null for unsure)"
            )
        conf = d.get("confidence")
        if conf not in VALID_CONFIDENCE:
            raise ValueError(
                f"{col}: confidence {conf!r} not in {sorted(VALID_CONFIDENCE)}"
            )
        if not d.get("reasoning"):
            raise ValueError(f"{col}: missing reasoning")

    return table_name, decisions, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decisions-file",
        type=Path,
        default=None,
        help="Path to decisions JSON; if omitted, reads from stdin.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing GT/audit entries for this table.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit partial table submissions (missing-column decisions).",
    )
    args = parser.parse_args()

    if not WORKING_SET.exists():
        print(
            f"No working_set at {WORKING_SET}.  Run "
            f"`python scripts/build_agent_mediated.py` first.",
            file=sys.stderr,
        )
        return 2
    working_set = json.loads(WORKING_SET.read_text())
    vocab = working_set["vocabulary"]

    if args.decisions_file:
        payload = json.loads(args.decisions_file.read_text())
    else:
        payload = json.loads(sys.stdin.read())

    table_name, decisions, missing = _validate_decisions(
        payload, working_set, allow_partial=args.allow_partial,
    )

    gt = _load_json(AGENT_MEDIATED, {})
    audit = _load_json(AUDIT, {})
    state = _load_json(REVIEW_STATE, {})

    now = _utc_iso()
    added = updated = skipped = 0
    for col_name, d in decisions.items():
        key = f"{table_name}.{col_name}"
        existing_gt = gt.get(key, "__missing__")
        if existing_gt != "__missing__" and not args.force:
            skipped += 1
            continue
        sources = d.get("sources", {})
        tag = d.get("tag")
        agreement = _classify_agreement(tag, sources) if tag else "no_decision"
        code = vocab.get(tag, {}).get("code") if tag else None
        label = vocab.get(tag, {}).get("label") if tag else None
        audit_entry = {
            "table_name": table_name,
            "column_name": col_name,
            "tag": tag,
            "code": code,
            "label": label,
            "confidence": d["confidence"],
            "reasoning": d["reasoning"],
            "sources": sources,
            "agreement": agreement,
            "needs_operator_review": bool(d.get("needs_operator_review", False)),
            "reviewed_at": now,
        }
        if key in audit:
            updated += 1
        else:
            added += 1
        # Canonical dual format — persistent taxonomy references must store
        # mnemonic AND code together to guard against structural drift (see
        # feedback memory: mnemonics-with-codes).  Writing the bare mnemonic
        # string here would silently change meaning if a curator moves the
        # mnemonic to a different code/subtree.
        gt[key] = {
            "mnemonic": tag,
            "code": code,
            "captured_at": now,
            "source": "agent_mediated",
        }
        audit[key] = audit_entry

    # Update review state.  Read PRIOR state for this table and merge —
    # if earlier batches already wrote decisions for some of the "missing"
    # columns in THIS submission, they're not actually missing.  Compute
    # completion from agent_mediated.json rather than the per-call view.
    prior = state.get(table_name) or {}
    already_decided = {
        col for k in gt
        if k.startswith(f"{table_name}.")
        for col in [k.split(".", 1)[1]]
    }
    truly_missing = [c for c in missing if c not in already_decided]
    state[table_name] = {
        "status": "complete" if not truly_missing else "partial",
        "reviewed_at": now,
        "decided_count": sum(1 for k in gt if k.startswith(f"{table_name}.")),
        "missing_count": len(truly_missing),
        "table_notes": payload.get("table_notes") or prior.get("table_notes"),
    }

    _archive_artifacts()
    _write_json(AGENT_MEDIATED, gt)
    _write_json(AUDIT, audit)
    _write_json(REVIEW_STATE, state)

    print(
        f"[{table_name}] added={added} updated={updated} skipped={skipped} "
        f"missing={len(missing)} status={state[table_name]['status']}",
        file=sys.stderr,
    )
    # Aggregate progress
    total_cols = len(working_set["columns"])
    tables_complete = sum(1 for v in state.values() if v["status"] == "complete")
    tables_total = working_set["metadata"]["table_count"]
    print(
        f"Overall: {len(gt)}/{total_cols} columns decided, "
        f"{tables_complete}/{tables_total} tables complete",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
