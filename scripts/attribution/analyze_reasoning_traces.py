#!/usr/bin/env python
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Light-touch analysis of reasoning_traces.jsonl from a pilot run.

Questions this script answers (intentionally simple, not a formal
ablation):

  1.  On pass-2 (revisit), when the LLM changes its mind, does its
      thinking text cite the CatBoost top-3 / the UNCERTAIN
      affordance / the prescriptive-prompt language?  We count how
      often each appears in pass-2 reasoning text.  A high rate of
      CatBoost citations is indirect evidence that the information-
      fusion loop is actually doing work (vs mere prompt compliance).

  2.  What is the relative reasoning-text length on pass-1 vs pass-2?
      More thinking on revisit is expected; dramatic change in either
      direction is informative.

The analysis is a grep over the traces, not a semantic parse.  It is
deliberately conservative: if GLM-4.7's phrasing varies (``catboost``,
``ML``, ``the classifier``), this may under-count.  We report the
matches that do surface and flag the method's limits in the output.

Usage::

    uv run python scripts/attribution/analyze_reasoning_traces.py <pilot_run_id>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


PROBE_TERMS = {
    "catboost_top_n": re.compile(r"\b(catboost|ml prediction|top-?3|top_?3|ml classifier|prior classifier)\b", re.IGNORECASE),
    "shap_top_n": re.compile(r"\b(shap|attribution|top (features|feature)|feature importance)\b", re.IGNORECASE),
    "uncertain_token": re.compile(r"\bUNCERTAIN\b"),
    "neighbor_labels": re.compile(r"\b(neighbor|nearest[-\s]?neighbor|similar column)\b", re.IGNORECASE),
    "revised_pass1": re.compile(r"\b(update|revis(e|ed|ing)|reconsider(ed)?|change(d)? my|on reflection)\b", re.IGNORECASE),
    "strict_language_ack": re.compile(r"\b(STRICT|MUST|REQUIRED|strictly|required to)\b"),
}


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: analyze_reasoning_traces.py <pilot_run_id>", file=sys.stderr)
        return 1

    run_dir = Path("build/sotab_pilot") / sys.argv[1]
    path = run_dir / "reasoning_traces.jsonl"
    if not path.is_file():
        print(f"missing {path}", file=sys.stderr); return 1

    traces: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traces.append(json.loads(line))

    by_pass: dict[str, list[dict]] = {"pass1": [], "pass2_revisit": []}
    for t in traces:
        by_pass.setdefault(t.get("pass", "unknown"), []).append(t)

    print(f"=== reasoning-trace analysis · {sys.argv[1]} ===\n")
    for tag, entries in by_pass.items():
        if not entries:
            continue
        total_chars = sum(len(e["reasoning_text"]) for e in entries)
        total_reasoning_tokens = sum(e.get("reasoning_tokens", 0) for e in entries)
        avg_chars = total_chars / len(entries) if entries else 0.0
        print(f"{tag:<15s}  batches={len(entries):3d}  "
              f"reasoning_chars={total_chars:>8d}  "
              f"reasoning_tokens={total_reasoning_tokens:>7d}  "
              f"avg_chars/batch={avg_chars:>6.0f}")
        print(f"  term-citation counts (matched batches / total {len(entries)}):")
        for name, pat in PROBE_TERMS.items():
            hits = sum(1 for e in entries if pat.search(e["reasoning_text"]))
            rate = hits / len(entries) if entries else 0.0
            print(f"    {name:<24s}  {hits:>3d}  ({rate:.0%})")
        print()

    # Headline comparison — pass-2 vs pass-1 on citation rates of the
    # evidence we explicitly injected into the revisit prompt.
    p1 = by_pass.get("pass1", [])
    p2 = by_pass.get("pass2_revisit", [])
    if p1 and p2:
        print("pass-2 vs pass-1 citation-rate delta (positive = pass-2 more often cites):")
        for name, pat in PROBE_TERMS.items():
            p1_rate = sum(1 for e in p1 if pat.search(e["reasoning_text"])) / len(p1)
            p2_rate = sum(1 for e in p2 if pat.search(e["reasoning_text"])) / len(p2)
            delta = p2_rate - p1_rate
            print(f"  {name:<24s}  p1={p1_rate:.0%}  p2={p2_rate:.0%}  Δ={delta:+.0%}")

    print("\nNote: this is a regex-based surface count, not a semantic parse. "
          "Missing matches may reflect GLM-4.7 using paraphrases we didn't anticipate. "
          "Use it as directional evidence, not a definitive attribution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
