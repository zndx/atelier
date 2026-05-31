#!/usr/bin/env python
"""Reflective ontology-enrichment evolution — GEPA-inspired.

GEPA (Agrawal et al., 2025; arXiv:2507.19457) uses natural-language
reflection over LLM trajectories to evolve prompts, maintaining a
Pareto front of candidates rather than a single best — outperforming
RL baselines (GRPO, MIPROv2) with 35× fewer rollouts.

This script adapts the same shape to ontology enrichment text:

* **Prompts being optimized** → embedding-text definitions per
  taxonomy node.
* **Trajectories** → classifications of columns where this node was
  cosine top-1 (over-attraction) or the reference (under-attraction).
* **Natural-language feedback** → diff between cosine top-1 picks and
  agent-mediated reference codes, surfaced as concrete confusion
  evidence.
* **Reflection** → an LLM examines current text + traces and proposes
  rewrites with explicit rationale.
* **Pareto front** → candidates are stored as a *list per node* so
  future iterations can keep multiple variants and ablate, rather
  than collapse to one best.
* **Mutation operators** → today: template rewrite + discriminator
  insertion + child citation + umbrella semantics.  Future: anti-
  example placement, common_names expansion, cross-node refactoring.

Current scope: Phase 0 (trace gathering) + Phase 1 (reflection +
proposal).  Phase 2 (apply + re-encode), Phase 3 (re-score against
agent-mediated reference), Phase 4 (iterate) are deliberately
deferred — the operator reviews candidates and decides apply manually
for now.  This is the seed of a methodology; the full loop will
follow once the seed proves out.

Usage:
    # Default — umbrella-template cohort, dry-run (prompts + traces only)
    python scripts/enrichment_evolution.py build/results/7bbe4533

    # Actually call the LLM
    python scripts/enrichment_evolution.py build/results/7bbe4533 --llm

    # Scope to one node
    python scripts/enrichment_evolution.py build/results/7bbe4533 \\
        --llm --only 1.1.2.2.1

Output: build/enrichment_evolution/<cohort_name>_v<N>/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")

logger = logging.getLogger(__name__)

# ── Cohort definitions ────────────────────────────────────────────

UMBRELLA_TEMPLATE_PHRASE = "unstructured, undefined, user-defined"
DEFAULT_COHORT_NAME = "umbrellas"


# ── Phase 0: trace gathering ──────────────────────────────────────


def load_classifications(run_dir: Path) -> list[dict]:
    path = run_dir / "classifications.json"
    if not path.exists():
        sys.exit(f"No classifications.json at {path}")
    with open(path) as f:
        return json.load(f)


def fetch_annotations(cohort_codes: list[str] | None = None) -> list[dict]:
    """Pull annotations from hive-poc/default.annotations.

    When cohort_codes is None, returns all rows matching the
    umbrella-template phrase.  When provided, returns those specific
    codes plus their immediate children.
    """
    import cml.data_v1 as cmldata
    conn = cmldata.get_connection("hive-poc")

    if cohort_codes is None:
        query = (
            "SELECT id, annotation, definition, common_names, deprecated "
            "FROM default.annotations "
            f"WHERE LOWER(definition) LIKE '%{UMBRELLA_TEMPLATE_PHRASE}%' "
            "ORDER BY id"
        )
    else:
        ids_str = ", ".join(f"'{c}'" for c in cohort_codes)
        like_clauses = " OR ".join(f"id LIKE '{c}.%'" for c in cohort_codes)
        query = (
            "SELECT id, annotation, definition, common_names, deprecated "
            "FROM default.annotations "
            f"WHERE id IN ({ids_str}) OR {like_clauses} "
            "ORDER BY id"
        )

    df = conn.get_pandas_dataframe(query)
    return df.to_dict(orient="records")


def gather_traces(
    classifications: list[dict],
    cohort_codes: set[str],
) -> dict[str, dict]:
    """For each cohort node, collect over-/under-/correctly-attracted columns.

    A column is:
    * **over-attracted** to N when cosine_top_1 == N and the reference
      is NOT N and NOT a descendant of N.  Conflicting evidence.
    * **correctly-attracted** to N when cosine_top_1 == N and the
      reference is N or a descendant.  Productive under subtree-FE
      routing.
    * **under-attracted** to N when reference is N (or a descendant
      of N for an umbrella) and cosine_top_1 != N and is NOT in N's
      subtree.  Missed signal.
    """
    traces: dict[str, dict] = {
        code: {
            "over_attracted": [],
            "correctly_attracted": [],
            "under_attracted": [],
            "total_top1_count": 0,
            "total_ref_count": 0,
        }
        for code in cohort_codes
    }

    for c in classifications:
        tk = (c.get("maxsim_attribution") or {}).get("top_k") or []
        top1_code = tk[0]["code"] if tk else None
        ref = c.get("reference_code") or None
        col = f"{c.get('table_name')}.{c.get('column_name')}"
        sample = {
            "column": col,
            "embedding_text": (c.get("embedding_text") or "")[:200],
            "cosine_top1": top1_code,
            "reference": ref,
            "predicted": c.get("predicted_code"),
        }

        # Attraction side (top-1 was a cohort node)
        if top1_code in cohort_codes:
            traces[top1_code]["total_top1_count"] += 1
            if ref is None:
                continue
            if ref == top1_code or ref.startswith(top1_code + "."):
                traces[top1_code]["correctly_attracted"].append(sample)
            else:
                traces[top1_code]["over_attracted"].append(sample)

        # Reference side — only meaningful for nodes that are
        # candidate reference targets.  Umbrella nodes almost never
        # appear as the reference themselves, but their subtree leaves
        # do; we report under-attraction when reference is *in* the
        # subtree but cosine top-1 went elsewhere.
        if ref is None:
            continue
        for code in cohort_codes:
            if ref == code or ref.startswith(code + "."):
                traces[code]["total_ref_count"] += 1
                if (top1_code is None
                        or (top1_code != code
                            and not top1_code.startswith(code + "."))):
                    traces[code]["under_attracted"].append(sample)

    return traces


# ── Phase 1: reflection prompt building ───────────────────────────


REFLECTION_SYSTEM_PROMPT = """\
You are optimizing the embedding text for a single taxonomy node so that
a ColBERT-based MaxSim retrieval system attracts the *right* database
columns to this node and not others.

You will receive:
* The node's current definition + common names.
* The node's children (if any).
* Concrete evidence of OVER-ATTRACTION (columns the system routed here
  but should have gone elsewhere) and CORRECT ATTRACTION (columns
  where this node was the right answer).

Your task is to:

1. **Diagnose** in 2-4 sentences: what tokens/phrases in the current
   definition are causing the over-attraction?  Be specific — name the
   exact tokens.
2. **Propose** a new definition that:
   - Leads with the *discriminating* concept.  Drops boilerplate
     ("Unstructured, undefined, user-defined, or aggregated data which
     may contain one or more X").  Generic high-frequency tokens that
     don't differentiate this node MUST go.
   - When the node is an UMBRELLA (has children), explicitly states
     umbrella semantics: "Used when a column clearly belongs to this
     family but the specific child subtype cannot be determined."
   - Cites the children inline so the embedding text carries the
     family signature.
   - Stays under 60 words.  ColBERT token budget is real; concise wins.
3. **List the targeted edits**: each as `"<old phrase>" → "<new phrase>": <which misclassification this fixes>`.

Output STRICT JSON with this shape:
{
  "diagnosis": "...",
  "new_definition": "...",
  "new_common_names": "comma, separated, list",
  "edits": [
    {"from": "...", "to": "...", "fixes": "..."}
  ]
}

No markdown, no preamble.  Just the JSON object.
"""


def build_reflection_prompt(
    node: dict,
    children: list[dict],
    trace: dict,
) -> str:
    """Compose the per-node reflection prompt body."""
    lines = []
    lines.append(f"# Node: {node['id']} [{node['annotation']}]\n")
    lines.append(f"## Current definition\n")
    lines.append(f"> {node['definition']}\n")
    if node.get("common_names"):
        lines.append(f"## Current common_names\n")
        lines.append(f"> {node['common_names']}\n")

    if children:
        lines.append(f"\n## Children ({len(children)})\n")
        for child in children:
            lines.append(
                f"- `{child['id']}` [{child['annotation']}] — "
                f"{child['definition'][:120]}"
            )

    lines.append(f"\n## Attraction statistics\n")
    lines.append(
        f"- Times this node was cosine top-1: **{trace['total_top1_count']}**"
    )
    lines.append(
        f"  - Of those, OVER-attracted (wrong subtree): "
        f"**{len(trace['over_attracted'])}**"
    )
    lines.append(
        f"  - Of those, correctly attracted (right subtree): "
        f"**{len(trace['correctly_attracted'])}**"
    )
    lines.append(
        f"- Times this node (or descendant) was the reference: "
        f"**{trace['total_ref_count']}**"
    )
    lines.append(
        f"  - Of those, UNDER-attracted (cosine went elsewhere): "
        f"**{len(trace['under_attracted'])}**"
    )

    if trace["over_attracted"]:
        lines.append(f"\n## OVER-attraction evidence (first 10)\n")
        lines.append("Columns this node attracted that should have gone elsewhere:\n")
        ref_counter = Counter(s["reference"] for s in trace["over_attracted"])
        lines.append("Distribution of *correct* destinations (most common):")
        for code, n in ref_counter.most_common(5):
            lines.append(f"- `{code}` — {n} columns")
        lines.append("\nSpecific examples:")
        for sample in trace["over_attracted"][:10]:
            lines.append(
                f"- `{sample['column']}` should have gone to "
                f"`{sample['reference']}`.  Embedding text: "
                f"\"{sample['embedding_text']}\""
            )

    if trace["under_attracted"]:
        lines.append(f"\n## UNDER-attraction evidence (first 5)\n")
        lines.append("Columns this node should have attracted but cosine went elsewhere:")
        for sample in trace["under_attracted"][:5]:
            lines.append(
                f"- `{sample['column']}` ref=`{sample['reference']}` "
                f"cosine_top1=`{sample['cosine_top1']}`.  Embedding text: "
                f"\"{sample['embedding_text']}\""
            )

    if trace["correctly_attracted"]:
        lines.append(f"\n## Correct-attraction examples (first 3, for context)\n")
        for sample in trace["correctly_attracted"][:3]:
            lines.append(
                f"- `{sample['column']}` ref=`{sample['reference']}`.  "
                f"Embedding text: \"{sample['embedding_text']}\""
            )

    return "\n".join(lines)


# ── Phase 1c: LLM call ────────────────────────────────────────────


ANTHROPIC_FALLBACK_MODEL = "claude-opus-4-7"


def resolve_model(cfg, override: str | None = None) -> str:
    """Pick the model id with CLI override > cfg.agent_model > fallback.

    Guards against the case where ATELIER_AGENT_MODEL is exported as an
    empty string in the Session pod env, which HOCON's ``${?VAR}``
    substitution treats as a real override and clobbers the config
    default of ``claude-opus-4-7``.
    """
    if override:
        return override
    candidate = (cfg.agent_model or "").strip()
    return candidate or ANTHROPIC_FALLBACK_MODEL


def call_llm(system: str, user: str, cfg, *, model: str) -> dict:
    """Call the configured LLM and parse a strict-JSON response."""
    from atelier.agents.client import (
        _build_anthropic_client,
        _build_bedrock_client,
    )

    is_bedrock = model.startswith("arn:") or "anthropic." in model
    client = (
        _build_bedrock_client(cfg, timeout=120.0)
        if is_bedrock
        else _build_anthropic_client(cfg, timeout=120.0)
    )

    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        system=[{"type": "text", "text": system,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(
        block.text for block in resp.content if hasattr(block, "text")
    ).strip()
    # Strip code fences if present
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`")
    return json.loads(text)


# ── Phase 1d: persistence ─────────────────────────────────────────


def next_cohort_version(out_root: Path, cohort_name: str) -> str:
    """Find the next vN suffix for this cohort name."""
    existing = sorted(out_root.glob(f"cohort_{cohort_name}_v*"))
    if not existing:
        return "v0"
    last = existing[-1].name
    n = int(last.rsplit("_v", 1)[1])
    return f"v{n + 1}"


def write_per_node_artifact(
    out_dir: Path,
    node: dict,
    reflection_prompt: str,
    proposal: dict | None,
) -> None:
    """Persist the per-node markdown artifact (reviewable)."""
    code = node["id"]
    safe = code.replace(".", "_")
    path = out_dir / "nodes" / f"{safe}.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [reflection_prompt, "\n---\n"]
    if proposal:
        lines.append("## Proposed rewrite\n")
        lines.append(f"**Diagnosis:** {proposal.get('diagnosis', '<missing>')}\n")
        lines.append(f"**New definition:**")
        lines.append(f"> {proposal.get('new_definition', '<missing>')}\n")
        if proposal.get("new_common_names"):
            lines.append(f"**New common_names:** {proposal['new_common_names']}\n")
        edits = proposal.get("edits", [])
        if edits:
            lines.append("**Targeted edits:**")
            for e in edits:
                lines.append(
                    f"- `{e.get('from', '?')}` → `{e.get('to', '?')}` — "
                    f"fixes: {e.get('fixes', '?')}"
                )
            lines.append("")
    else:
        lines.append("## Proposal\n")
        lines.append("*Dry-run mode — re-run with `--llm` to populate.*\n")

    path.write_text("\n".join(lines))


# ── CLI ───────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dir",
        help="Path to build/results/<run_id> for trace gathering",
    )
    parser.add_argument(
        "--llm", action="store_true",
        help="Actually call the LLM; default is dry-run (prompts + traces only)",
    )
    parser.add_argument(
        "--only", default=None,
        help="Scope to a single node code (e.g. 1.1.2.2.1)",
    )
    parser.add_argument(
        "--cohort-name", default=DEFAULT_COHORT_NAME,
        help=f"Cohort label (default: {DEFAULT_COHORT_NAME})",
    )
    parser.add_argument(
        "--out-root", default="build/enrichment_evolution",
        help="Output root directory (default: build/enrichment_evolution)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override model id (defaults to cfg.agent_model, or "
             f"{ANTHROPIC_FALLBACK_MODEL} if cfg's value is empty)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run_dir = Path(args.run_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading classifications from {run_dir}...")
    rows = load_classifications(run_dir)
    print(f"  {len(rows)} classifications")

    print("Pulling annotations from hive-poc/default.annotations...")
    if args.only:
        annotations = fetch_annotations(cohort_codes=[args.only])
        cohort_nodes = [a for a in annotations if a["id"] == args.only]
    else:
        annotations = fetch_annotations()
        cohort_nodes = annotations
    cohort_codes = {a["id"] for a in cohort_nodes}
    children_by_parent: dict[str, list[dict]] = defaultdict(list)
    for a in annotations:
        for parent_code in cohort_codes:
            if a["id"].startswith(parent_code + ".") and a["id"] != parent_code:
                children_by_parent[parent_code].append(a)
    print(f"  cohort size: {len(cohort_codes)} node(s)")

    print("Gathering traces...")
    traces = gather_traces(rows, cohort_codes)

    version = next_cohort_version(out_root, args.cohort_name)
    out_dir = out_root / f"cohort_{args.cohort_name}_{version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    candidates_json = {
        "run_id": run_dir.name,
        "cohort_name": args.cohort_name,
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_called": args.llm,
        "candidates": {},
    }

    cfg = None
    model = None
    if args.llm:
        from atelier.config import load_config
        cfg = load_config()
        model = resolve_model(cfg, args.model)
        candidates_json["model"] = model
        print(f"  LLM model: {model}")

    for node in sorted(cohort_nodes, key=lambda n: n["id"]):
        code = node["id"]
        children = sorted(children_by_parent.get(code, []),
                          key=lambda n: n["id"])
        trace = traces.get(code, {
            "over_attracted": [], "correctly_attracted": [],
            "under_attracted": [], "total_top1_count": 0,
            "total_ref_count": 0,
        })

        reflection = build_reflection_prompt(node, children, trace)
        proposal = None
        if args.llm:
            print(f"  Reflecting on {code}...")
            try:
                proposal = call_llm(
                    REFLECTION_SYSTEM_PROMPT, reflection, cfg,
                    model=model,
                )
            except Exception as exc:
                logger.error("LLM call for %s failed: %s", code, exc)
                proposal = {"error": f"{type(exc).__name__}: {exc}"}

        write_per_node_artifact(out_dir, node, reflection, proposal)

        candidates_json["candidates"][code] = {
            "annotation": node["annotation"],
            "current_definition": node["definition"],
            "current_common_names": node.get("common_names", ""),
            "children_codes": [c["id"] for c in children],
            "trace_summary": {
                "total_top1": trace["total_top1_count"],
                "total_ref": trace["total_ref_count"],
                "over_attracted_n": len(trace["over_attracted"]),
                "correctly_attracted_n": len(trace["correctly_attracted"]),
                "under_attracted_n": len(trace["under_attracted"]),
            },
            "proposals": [proposal] if proposal else [],
        }

    (out_dir / "candidates.json").write_text(
        json.dumps(candidates_json, indent=2, default=str),
    )

    # Aggregate summary
    summary_lines = [
        f"# Cohort `{args.cohort_name}` {version} — reflective rewrite",
        "",
        f"- Source run: `{run_dir.name}`",
        f"- Cohort size: {len(cohort_codes)} node(s)",
        f"- LLM called: {args.llm}",
    ]
    if args.llm and model is not None:
        summary_lines.append(f"- Model: `{model}`")
    summary_lines.append("")
    summary_lines.append("| code | annotation | top1 | over | correct | ref | under |")
    summary_lines.append("|---|---|---:|---:|---:|---:|---:|")
    for code in sorted(cohort_codes):
        t = traces.get(code, {})
        node = next(n for n in cohort_nodes if n["id"] == code)
        summary_lines.append(
            f"| `{code}` | {node['annotation']} | "
            f"{t.get('total_top1_count', 0)} | "
            f"{len(t.get('over_attracted', []))} | "
            f"{len(t.get('correctly_attracted', []))} | "
            f"{t.get('total_ref_count', 0)} | "
            f"{len(t.get('under_attracted', []))} |"
        )
    summary_lines.append("")
    summary_lines.append(
        f"Per-node artifacts in `nodes/`.  Review proposals, then apply "
        f"selected rewrites via your enrichment-edit pathway."
    )
    (out_dir / "summary.md").write_text("\n".join(summary_lines))

    print(f"\nWrote {out_dir / 'candidates.json'}")
    print(f"Wrote {out_dir / 'summary.md'}")
    print(f"Per-node artifacts: {out_dir / 'nodes'}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
