#!/usr/bin/env python
"""Semantic-optimization loop — APO-shaped, single-branch critic, frozen reference.

Phase 2-4 of the GEPA-inspired methodology started in
``scripts/enrichment_evolution.py`` (which covers Phase 0+1, trace +
single-pass propose).  This script adds the iterative loop:

  Phase 2 — apply LLM-proposed enrichment edits to live Qdrant in place
            (preserving point_ids)
  Phase 3 — re-score against the frozen reference using failure-row
            queries from classifications.json (per-cluster rescue@1/3/5
            + regression check)
  Phase 4 — iterate, with implicit per-row attempt counter for
            stagnation detection.  All rows stay in the metric
            denominator at all times.

Discipline-preserving choices:

* The critic LLM proposes **enrichment edits only** — it cannot
  recommend reference changes.  Reference-incoherence surfacing is
  procedural: rows that exhaust per-row attempts without rescue are
  listed in a hand-off artifact at termination, *only if* the overall
  cluster objective is also missed.
* All rows stay in the rescue-rate denominator throughout.  The
  attempt counter gates further critic proposals for a given row but
  never removes it from the score.  This prevents the optimizer from
  gaming its objective by labeling difficult rows as "skip."
* Reference data (``agent_mediated.json``) is read as a frozen oracle;
  the loop hashes it at start and end and refuses to terminate if
  the hashes differ.

Usage:
  python scripts/semantic_optimize.py \\
    --run-dir build/results/46229e17 \\
    --target 1.3.2 \\
    --max-iterations 8 --max-attempts 5 --rescue-target 75 --llm

  # Smoke (no LLM call, no Qdrant mutation — just measurement)
  python scripts/semantic_optimize.py --run-dir build/results/46229e17 \\
    --target 1.3.2 --dry-run

  # Adversarial check (force all rows attempt-exhausted on iter 1)
  python scripts/semantic_optimize.py --run-dir build/results/46229e17 \\
    --target 1.3.2 --adversarial

Output: build/semantic_optimization/cohort_<name>_v<N>/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

COLLECTION = "hive-poc_hive-poc-default-annotations_019e5636-f3d8-7231-a1b0-4bdf0ea3a4ce"
QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_OUT_ROOT = "build/semantic_optimization"
REFERENCE_PATH = Path("build/data/agent_mediated/agent_mediated.json")
ANTHROPIC_FALLBACK_MODEL = "claude-opus-4-7"


# ──────────────────────────────────────────────────────────────────────
# Data shapes
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FailureRow:
    """One row that wants to land at target_code but currently doesn't."""
    table: str
    column: str
    reference_code: str
    embedding_text: str
    pre_cosine_top1: str | None
    pre_cosine_top5: list[tuple[str, float]]

    @property
    def key(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass
class Cluster:
    """A target code + its in-budget failure rows."""
    target_code: str
    target_label: str
    rows: list[FailureRow]


@dataclass
class IterationResult:
    iteration: int
    rescue_at_1: int
    rescue_at_3: int
    rescue_at_5: int
    total: int
    regressions: int
    edited_nodes: list[str]
    proposal_hash: str | None
    per_row_top1_after: dict[str, str | None]
    elapsed_s: float


@dataclass
class LoopResult:
    target_code: str
    target_label: str
    iterations: list[IterationResult]
    final_rescue_at_5: int
    objective_met: bool
    resisting_rows: list[FailureRow]
    per_row_attempts: dict[str, int]
    pre_state_hash: str
    post_state_hash: str
    reference_hash_pre: str
    reference_hash_post: str


# ──────────────────────────────────────────────────────────────────────
# Qdrant helpers
# ──────────────────────────────────────────────────────────────────────

def qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=QDRANT_URL)


def get_payload_by_code(code: str) -> dict | None:
    """Read live Qdrant payload (with vector) by `code` field."""
    import requests
    r = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
        json={"limit": 1, "with_payload": True, "with_vector": True,
              "filter": {"must": [{"key": "code", "match": {"value": code}}]}},
        timeout=10,
    )
    r.raise_for_status()
    pts = r.json().get("result", {}).get("points", [])
    return pts[0] if pts else None


def upsert_point_in_place(point_id: str, vectors: list[list[float]], payload: dict) -> None:
    """PUT the point back with same id (preserves position; replaces vector + payload)."""
    import requests
    r = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points",
        json={"points": [{"id": point_id, "vector": {"colbert": vectors}, "payload": payload}]},
        timeout=30,
    )
    r.raise_for_status()


# ──────────────────────────────────────────────────────────────────────
# Proposal merge — defensive, only known fields can be modified
# ──────────────────────────────────────────────────────────────────────

ALLOWED_PROPOSAL_FIELDS = {
    "description", "name_hints_add", "name_hints_remove",
    "value_patterns_add", "value_patterns_remove_expr",
    "prototype_values_add", "prototype_values_remove",
    "anti_examples_add",
}


def _norm_hint(h) -> str | None:
    """Normalize a name_hint to a plain string. LLM sometimes returns dicts."""
    if isinstance(h, str):
        return h
    if isinstance(h, dict):
        return str(h.get("hint") or h.get("value") or h.get("name") or "")
    return str(h) if h is not None else None


def _norm_value(v) -> str | None:
    """Normalize a prototype_value to a plain string."""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return str(v.get("value") or v.get("hint") or "")
    return str(v) if v is not None else None


def merge_proposal_into_payload(payload: dict, proposal: dict) -> dict:
    """Apply LLM proposal to a copy of payload.  Defensive: reject unknown keys."""
    new = dict(payload)
    edits = proposal.get("edits") or {}
    unknown = set(edits.keys()) - ALLOWED_PROPOSAL_FIELDS
    if unknown:
        logger.warning(f"proposal contains unknown fields, ignoring: {unknown}")

    if "description" in edits and edits["description"]:
        new["description"] = edits["description"]

    if "name_hints_add" in edits and edits["name_hints_add"]:
        current = [_norm_hint(h) for h in (new.get("name_hints") or [])]
        current = [h for h in current if h]
        adds = [_norm_hint(h) for h in edits["name_hints_add"]]
        adds = [h for h in adds if h]
        new["name_hints"] = list(dict.fromkeys(current + adds))[:40]

    if "name_hints_remove" in edits and edits["name_hints_remove"]:
        removes = {_norm_hint(h) for h in edits["name_hints_remove"]}
        new["name_hints"] = [h for h in (new.get("name_hints") or [])
                              if _norm_hint(h) not in removes]

    if "value_patterns_add" in edits and edits["value_patterns_add"]:
        new["value_patterns"] = list(new.get("value_patterns") or []) + list(edits["value_patterns_add"])

    if "value_patterns_remove_expr" in edits and edits["value_patterns_remove_expr"]:
        removes = set(edits["value_patterns_remove_expr"])
        new["value_patterns"] = [
            p for p in (new.get("value_patterns") or [])
            if not (isinstance(p, dict) and p.get("expr") in removes)
        ]

    if "prototype_values_add" in edits and edits["prototype_values_add"]:
        current = [_norm_value(v) for v in (new.get("prototype_values") or [])]
        current = [v for v in current if v]
        adds = [_norm_value(v) for v in edits["prototype_values_add"]]
        adds = [v for v in adds if v]
        new["prototype_values"] = list(dict.fromkeys(current + adds))[:20]

    if "prototype_values_remove" in edits and edits["prototype_values_remove"]:
        removes = {_norm_value(v) for v in edits["prototype_values_remove"]}
        new["prototype_values"] = [v for v in (new.get("prototype_values") or [])
                                    if _norm_value(v) not in removes]

    if "anti_examples_add" in edits and edits["anti_examples_add"]:
        new["anti_examples"] = list(new.get("anti_examples") or []) + list(edits["anti_examples_add"])

    # Append-only operator_edits log
    new["operator_edits"] = list(new.get("operator_edits") or []) + [{
        "at": datetime.now(timezone.utc).isoformat(),
        "by": "semantic_optimize.py",
        "kind": "iterative_loop",
        "summary": (proposal.get("diagnosis") or "")[:200],
    }]

    return new


# ──────────────────────────────────────────────────────────────────────
# Compose-and-upsert (reuses qdrant_writer.compose_annotation_text)
# ──────────────────────────────────────────────────────────────────────

def apply_proposal_to_qdrant(
    target_code: str,
    point_id: str,
    new_payload: dict,
    encoder,
) -> None:
    """Re-embed new payload via compose_annotation_text + colbert encoder, then upsert."""
    from atelier.enrichment.qdrant_writer import compose_annotation_text
    doc_text = compose_annotation_text(new_payload)
    vecs = encoder.encode_single(doc_text)
    upsert_point_in_place(point_id, vecs.tolist(), new_payload)


# ──────────────────────────────────────────────────────────────────────
# Phase 3: rescue measurement
# ──────────────────────────────────────────────────────────────────────

def is_subtree_match(code: str | None, target: str) -> bool:
    if not code:
        return False
    return code == target or code.startswith(target + ".")


def measure_cluster_rescue(
    cluster: Cluster,
    encoder,
    client,
) -> tuple[dict[str, int], dict[str, str | None], dict[str, list[tuple[str, float]]]]:
    """Run cosine queries for every row in cluster.

    Returns:
      counts:     {"r1": int, "r3": int, "r5": int, "total": int}
      top1_map:   {row_key: top1_code | None}
      top5_map:   {row_key: [(code, score), ...top5]}
    """
    from atelier.enrichment.qdrant_writer import COLBERT_VECTOR_NAME
    counts = {"r1": 0, "r3": 0, "r5": 0, "total": 0}
    top1_map: dict[str, str | None] = {}
    top5_map: dict[str, list[tuple[str, float]]] = {}
    for row in cluster.rows:
        counts["total"] += 1
        if not row.embedding_text:
            top1_map[row.key] = None
            top5_map[row.key] = []
            continue
        vecs = encoder.encode_single(row.embedding_text)
        res = client.query_points(
            collection_name=COLLECTION, query=vecs.tolist(),
            using=COLBERT_VECTOR_NAME, limit=5, with_payload=True,
        )
        top5 = [((p.payload.get("code", "") or "").rstrip("*"), round(p.score, 4)) for p in res.points]
        top1_map[row.key] = top5[0][0] if top5 else None
        top5_map[row.key] = top5
        if top5 and is_subtree_match(top5[0][0], cluster.target_code):
            counts["r1"] += 1
        if any(is_subtree_match(c, cluster.target_code) for c, _ in top5[:3]):
            counts["r3"] += 1
        if any(is_subtree_match(c, cluster.target_code) for c, _ in top5[:5]):
            counts["r5"] += 1
    return counts, top1_map, top5_map


def check_regressions(
    regression_sample: list[dict],
    encoder,
    client,
) -> tuple[int, int, int]:
    """Sample N already-correct rows; verify post-state still ranks ref-correct.

    Returns: (pre_correct, post_correct, regressed_count).
    `regressed_count` = rows that were correct pre-edit but wrong post-edit.
    """
    from atelier.enrichment.qdrant_writer import COLBERT_VECTOR_NAME
    pre_ok = post_ok = regressed = 0
    for r in regression_sample:
        ref = r.get("reference_code")
        emb = r.get("embedding_text", "")
        if not ref or not emb:
            continue
        pre_top1_attr = r.get("cosine_attribution", {}).get("top_k", [])
        pre_top1 = (pre_top1_attr[0].get("code", "").rstrip("*") if pre_top1_attr else None)
        pre_match = is_subtree_match(pre_top1, ref)
        vecs = encoder.encode_single(emb)
        res = client.query_points(
            collection_name=COLLECTION, query=vecs.tolist(),
            using=COLBERT_VECTOR_NAME, limit=1, with_payload=True,
        )
        post_top1 = ((res.points[0].payload.get("code") or "").rstrip("*") if res.points else None)
        post_match = is_subtree_match(post_top1, ref)
        if pre_match:
            pre_ok += 1
        if post_match:
            post_ok += 1
        if pre_match and not post_match:
            regressed += 1
    return pre_ok, post_ok, regressed


def measure_neighbor_protection(
    neighbor_codes: list[str],
    classifications: list[dict],
    encoder,
    client,
) -> dict[str, int]:
    """For each neighbor code, count rows where ref ∈ subtree(neighbor)
    AND current cosine top-1 ∈ subtree(neighbor) — i.e., rows the neighbor
    currently rightfully wins.

    Edits to the target's enrichment could damage these neighbor-wins;
    this measurement feeds the cross-cluster guard inside the iteration
    loop.  Per [[feedback-hierarchical-svm-only]] and the
    domain-adaptation framing in classification.md, the cosine channel's
    job is per-cluster, but its DST-independence guarantee depends on
    not stealing other clusters' rightful wins.
    """
    from atelier.enrichment.qdrant_writer import COLBERT_VECTOR_NAME
    out: dict[str, int] = {}
    for ncode in neighbor_codes:
        kept = 0
        for c in classifications:
            ref = c.get("reference_code")
            emb = c.get("embedding_text", "")
            if not ref or not emb or not is_subtree_match(ref, ncode):
                continue
            vecs = encoder.encode_single(emb)
            res = client.query_points(
                collection_name=COLLECTION, query=vecs.tolist(),
                using=COLBERT_VECTOR_NAME, limit=1, with_payload=True,
            )
            post_top1 = ((res.points[0].payload.get("code") or "").rstrip("*") if res.points else None)
            if is_subtree_match(post_top1, ncode):
                kept += 1
        out[ncode] = kept
    return out


# ──────────────────────────────────────────────────────────────────────
# Critic prompt (single-branch — enrichment edits only)
# ──────────────────────────────────────────────────────────────────────

CRITIC_SYSTEM_PROMPT = """You are an enrichment-payload critic for a ColBERT-based late-interaction cosine evidence channel inside the Atelier classification pipeline.  Your job is to propose edits to one taxonomy node's enrichment payload so that the node's MaxSim score against a specific set of failure-row queries IMPROVES — without breaking unrelated rows.

You are working at the TOKEN level.  ColBERT MaxSim sums per-query-token best matches across the document's tokens.  No semantic understanding of clauses, scope, or intent — every token is a positive vote.

EMPIRICALLY VALIDATED PRINCIPLES (these have been measured to matter):

1. NO NEGATION SEMANTICS.  Phrases like "Not for X" or "Excludes X" or "rather than Y" make X and Y POSITIVE TOKENS in the embedding.  Move all exclusion semantics to the structured `anti_examples` field.  Never put negative prose in `description`.

2. TOKEN SURFACE IS MASS.  Descriptions ≤100 chars get out-competed by fat hijackers (400+ chars) regardless of semantic correctness.  Aim for 250-500 chars of positive-anchor prose.  Quality > quantity, but starvation is fatal.

3. SIBLING OVERLAP IS A CURATION DEFECT.  If two nodes have overlapping name_hints or value_pattern regexes, both attract the same queries.  Each pattern should be owned by exactly one node.  If you see overlap with a competing node listed below, propose edits that disambiguate.

4. NAME_HINTS ARE HIGH-VALUE TOKENS.  Query strings often contain column names verbatim.  If failure rows have column names like `listing_status`, `order_status`, etc., putting those exact strings in `name_hints` is a high-leverage edit.

5. QUERY-SURFACE ALIGNMENT.  The query text is `column_name | column_type | sample_values | cardinality=N | ...`.  Your edits should anchor to tokens that appear in the failure-row queries verbatim.

CONSTRAINTS:
- You may modify ONLY: description, name_hints, value_patterns, prototype_values, anti_examples.
- You CANNOT recommend reference changes; the reference is a frozen oracle.  If a row "feels wrong" to you, propose your best enrichment edit anyway — the system has separate procedural mechanisms for handling rows that resist optimization.
- You MUST propose at least one edit per call.  "Cannot be done" is not an output.

OUTPUT FORMAT — strict JSON, no prose outside the JSON:

{
  "diagnosis": "2-3 sentences naming specific tokens/phrases causing the failures",
  "edits": {
    "description": "new description text (optional)",
    "name_hints_add": ["..."],  // optional - appends to existing
    "name_hints_remove": ["..."],  // optional - removes by exact match
    "value_patterns_add": [{"kind":"regex","expr":"..."}],  // optional
    "value_patterns_remove_expr": ["..."],  // optional - removes by exact expr
    "prototype_values_add": ["..."],  // optional
    "prototype_values_remove": ["..."],  // optional
    "anti_examples_add": [{"value":"...","confusable_tag":"...","reason":"..."}]  // optional
  },
  "rationale": "1-2 sentences on why these edits should help"
}
"""


def build_critic_user_prompt(
    target_payload: dict,
    target_code: str,
    hijacker_payloads: list[tuple[str, dict, int]],  # [(code, payload, vote_count), ...]
    sample_failure_rows: list[FailureRow],
    prior_attempts: list[dict],
    iteration: int,
    max_iterations: int,
    pre_rescue_at_5_pct: float,
    current_rescue_at_5_pct: float,
) -> str:
    """Assemble the user message for the critic."""
    lines = []
    lines.append(f"# Target node: `{target_code}` ({target_payload.get('label')})")
    lines.append(f"Mnemonic: {target_payload.get('mnemonic')}")
    lines.append(f"Iteration {iteration} of {max_iterations}.")
    lines.append(f"Pre-loop rescue@5: {pre_rescue_at_5_pct:.1f}%.  Current rescue@5: {current_rescue_at_5_pct:.1f}%.")
    lines.append("")
    lines.append("## Current target payload\n")
    lines.append(f"**description** ({len(target_payload.get('description','') or '')} chars):")
    lines.append(f"> {target_payload.get('description','')}")
    lines.append("")
    lines.append(f"**name_hints** ({len(target_payload.get('name_hints') or [])}):")
    lines.append(f"> {target_payload.get('name_hints')}")
    lines.append("")
    lines.append(f"**value_patterns** ({len(target_payload.get('value_patterns') or [])}):")
    for p in (target_payload.get('value_patterns') or [])[:8]:
        lines.append(f"> - {p}")
    lines.append("")
    lines.append(f"**prototype_values**: {target_payload.get('prototype_values')}")
    lines.append("")

    if hijacker_payloads:
        lines.append("## Competing nodes currently winning these queries\n")
        for hcode, hpayload, votes in hijacker_payloads:
            lines.append(f"### `{hcode}` ({hpayload.get('mnemonic')}: {hpayload.get('label')}) — winning top-1 on {votes} of {len(sample_failure_rows)} rows below")
            lines.append(f"  description ({len(hpayload.get('description','') or '')} chars):")
            lines.append(f"  > {(hpayload.get('description','') or '')[:400]}")
            lines.append(f"  name_hints: {(hpayload.get('name_hints') or [])[:10]}")
            lines.append("")
        lines.append("**Cross-cluster impact warning**: each competing node above has OTHER rows in the corpus where it correctly wins top-1 (not shown here).  Your edits to the target could inadvertently steal those rightful wins.  A net-positive guard will REJECT your proposal if target gains are exceeded by losses on competing nodes.  Favor edits that *disambiguate* (move shared tokens out of competing-node payloads via the value-pattern/name-hint surface, or anchor target with discriminating tokens) over edits that just add bulk to the target.")
        lines.append("")

    lines.append("## Sample of failure-row queries (verbatim embedding_text)\n")
    lines.append(f"These rows should rank `{target_code}` in cosine top-5 but currently don't.")
    lines.append("")
    for row in sample_failure_rows[:10]:
        lines.append(f"- **{row.key}** (reference={row.reference_code}, current top-1={row.pre_cosine_top1 or '?'})")
        lines.append(f"  query: `{row.embedding_text[:200]}`")
    lines.append("")

    if prior_attempts:
        lines.append("## Prior iteration attempts on this target\n")
        for i, attempt in enumerate(prior_attempts[-3:], 1):
            lines.append(f"### Attempt {attempt.get('iteration', '?')}: diagnosis = \"{(attempt.get('diagnosis') or '')[:140]}\"")
            edits = attempt.get('edits') or {}
            edit_keys = [k for k in edits.keys() if edits.get(k)]
            lines.append(f"  edits made: {edit_keys}")
            lines.append(f"  rescue@5 after: {attempt.get('rescue_at_5_pct','?'):.1f}%")
            lines.append("")
        lines.append("Reflect on what worked vs what didn't.  Propose a NEW edit, not a repeat.")
        lines.append("")

    lines.append("## Your task\n")
    lines.append(f"Propose enrichment edits to `{target_code}` to lift its MaxSim against the failure-row queries.")
    lines.append("Output the strict JSON object specified in the system message.")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Phase 4: iterate
# ──────────────────────────────────────────────────────────────────────

def reference_hash() -> str:
    """sha256 of agent_mediated.json for immutability check."""
    if not REFERENCE_PATH.exists():
        return ""
    return hashlib.sha256(REFERENCE_PATH.read_bytes()).hexdigest()


def state_hash(target_code: str) -> str:
    """sha256 of the live Qdrant payload for the target — proxies Qdrant state."""
    pt = get_payload_by_code(target_code)
    if pt is None:
        return ""
    p = pt.get("payload", {})
    blob = json.dumps({
        "description": p.get("description"),
        "name_hints": p.get("name_hints"),
        "value_patterns": p.get("value_patterns"),
        "prototype_values": p.get("prototype_values"),
        "anti_examples": p.get("anti_examples"),
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def build_cluster(target_code: str, target_label: str, classifications: list[dict]) -> Cluster:
    """All rows where reference_code is in target subtree but cosine top-1 isn't."""
    rows = []
    for c in classifications:
        ref = c.get("reference_code")
        if not is_subtree_match(ref, target_code):
            continue
        tk = (c.get("cosine_attribution") or {}).get("top_k") or []
        top1 = (tk[0].get("code", "").rstrip("*") if tk else None)
        if is_subtree_match(top1, target_code):
            continue  # already correct under cosine — not a failure
        rows.append(FailureRow(
            table=c.get("table_name", ""),
            column=c.get("column_name", ""),
            reference_code=ref,
            embedding_text=c.get("embedding_text", "") or "",
            pre_cosine_top1=top1,
            pre_cosine_top5=[(x.get("code", "").rstrip("*"), x.get("maxsim_score", 0))
                              for x in tk[:5]],
        ))
    return Cluster(target_code=target_code, target_label=target_label, rows=rows)


def call_critic(
    target_payload: dict,
    target_code: str,
    hijacker_payloads: list[tuple[str, dict, int]],
    sample_failure_rows: list[FailureRow],
    prior_attempts: list[dict],
    iteration: int,
    max_iterations: int,
    pre_rescue_pct: float,
    current_rescue_pct: float,
    cfg,
    model: str,
) -> dict:
    """Single critic call.  Reuses scripts.enrichment_evolution.call_llm."""
    from enrichment_evolution import call_llm
    user = build_critic_user_prompt(
        target_payload, target_code, hijacker_payloads, sample_failure_rows,
        prior_attempts, iteration, max_iterations, pre_rescue_pct, current_rescue_pct,
    )
    return call_llm(CRITIC_SYSTEM_PROMPT, user, cfg, model=model)


def iterate_with_attempt_counter(
    cluster: Cluster,
    *,
    encoder,
    client,
    cfg,
    model: str,
    classifications: list[dict],
    max_iterations: int = 8,
    max_attempts_per_row: int = 5,
    rescue_r1_pct: int = 82,
    rescue_r3_pct: int = 90,
    rescue_r5_pct: int = 94,
    no_improvement_window: int = 3,
    max_neighbors: int = 3,
    regression_sample: list[dict],
    out_dir: Path,
    dry_run: bool = False,
    adversarial: bool = False,
) -> LoopResult:
    """Outer loop.  All rows stay in metric denominator at all times.

    Per-row attempt counter gates which rows are surfaced to the critic
    (we don't propose more edits for rows that've hit max attempts),
    but exhausted rows still count in the rescue-rate calculation.
    """
    reference_pre = reference_hash()
    pre_state = state_hash(cluster.target_code)

    per_row_attempts: dict[str, int] = {r.key: 0 for r in cluster.rows}
    iterations_log: list[IterationResult] = []
    prior_attempts_for_critic: list[dict] = []
    iterations_log_file = out_dir / "iterations.jsonl"
    iterations_log_file.parent.mkdir(parents=True, exist_ok=True)
    log_fh = iterations_log_file.open("w")

    # Initial measurement (iteration 0 — pre-edit baseline)
    target_pt = get_payload_by_code(cluster.target_code)
    target_payload = target_pt["payload"] if target_pt else {}
    target_point_id = target_pt["id"] if target_pt else None

    counts0, top1_map0, top5_map0 = measure_cluster_rescue(cluster, encoder, client)
    pre_pct = (counts0["r5"] / counts0["total"] * 100) if counts0["total"] else 0
    iter0 = IterationResult(
        iteration=0, rescue_at_1=counts0["r1"], rescue_at_3=counts0["r3"],
        rescue_at_5=counts0["r5"], total=counts0["total"], regressions=0,
        edited_nodes=[], proposal_hash=None,
        per_row_top1_after=dict(top1_map0), elapsed_s=0.0,
    )
    iterations_log.append(iter0)
    log_fh.write(json.dumps(asdict(iter0)) + "\n")
    log_fh.flush()
    print(f"\n  iter 0 (baseline): rescue@1={counts0['r1']}/{counts0['total']} "
          f"rescue@3={counts0['r3']}/{counts0['total']} rescue@5={counts0['r5']}/{counts0['total']}")

    if adversarial:
        # Force per-row attempts exhausted on iter 1 to verify denominator stays intact
        for k in per_row_attempts:
            per_row_attempts[k] = max_attempts_per_row
        print(f"  [ADVERSARIAL] all {len(per_row_attempts)} rows marked attempt-exhausted")

    for it in range(1, max_iterations + 1):
        t0 = time.time()
        # Stop: ALL THREE thresholds met (compressed-range objective)
        latest = iterations_log[-1]
        latest_r1_pct = (latest.rescue_at_1 / latest.total * 100) if latest.total else 0
        latest_r3_pct = (latest.rescue_at_3 / latest.total * 100) if latest.total else 0
        latest_r5_pct = (latest.rescue_at_5 / latest.total * 100) if latest.total else 0
        if (latest_r1_pct >= rescue_r1_pct
                and latest_r3_pct >= rescue_r3_pct
                and latest_r5_pct >= rescue_r5_pct):
            print(f"  STOPPED: thresholds met (@1={latest_r1_pct:.1f}≥{rescue_r1_pct} "
                  f"@3={latest_r3_pct:.1f}≥{rescue_r3_pct} @5={latest_r5_pct:.1f}≥{rescue_r5_pct})")
            break
        # Stop: stagnation — no improvement on rescue@1 OR rescue@5 in window
        if len(iterations_log) >= no_improvement_window + 1:
            recent_r1 = [r.rescue_at_1 for r in iterations_log[-(no_improvement_window + 1):]]
            recent_r5 = [r.rescue_at_5 for r in iterations_log[-(no_improvement_window + 1):]]
            if max(recent_r1) <= recent_r1[0] and max(recent_r5) <= recent_r5[0]:
                print(f"  STOPPED: no improvement in last {no_improvement_window} iterations")
                break

        # Which rows still have attempts remaining?
        in_budget = [r for r in cluster.rows if per_row_attempts[r.key] < max_attempts_per_row]
        if not in_budget:
            print(f"  STOPPED: all rows attempt-exhausted")
            break

        # Identify top hijackers from latest measurement
        latest_top1 = iterations_log[-1].per_row_top1_after
        hijacker_votes = Counter(
            latest_top1[r.key] for r in in_budget
            if not is_subtree_match(latest_top1.get(r.key), cluster.target_code)
            and latest_top1.get(r.key)
        )
        top_hijackers: list[tuple[str, dict, int]] = []
        for hcode, votes in hijacker_votes.most_common(max_neighbors):
            hpt = get_payload_by_code(hcode)
            if hpt:
                top_hijackers.append((hcode, hpt["payload"], votes))

        # Sample failure rows for the critic — prefer in-budget rows that are NOT yet rescued
        # Always include some still-failing rows; if none, include any in-budget ones
        unrescued_in_budget = [r for r in in_budget
                               if not is_subtree_match(latest_top1.get(r.key), cluster.target_code)]
        sample_rows = unrescued_in_budget[:10] or in_budget[:10]

        if not sample_rows:
            print(f"  STOPPED: no rows to surface to critic")
            break

        # Refresh target payload from Qdrant (in case prior iter mutated it)
        target_pt = get_payload_by_code(cluster.target_code)
        target_payload = target_pt["payload"]
        target_point_id = target_pt["id"]

        if dry_run:
            print(f"  iter {it}: DRY RUN — would call critic on {len(sample_rows)} sample rows, {len(top_hijackers)} hijackers")
            edited_nodes = []
            proposal_hash = None
            counts = counts0
        else:
            print(f"  iter {it}: calling critic (sample={len(sample_rows)} rows, "
                  f"hijackers={[(h,v) for h,_,v in top_hijackers]})")
            try:
                proposal = call_critic(
                    target_payload, cluster.target_code, top_hijackers,
                    sample_rows, prior_attempts_for_critic, it, max_iterations,
                    pre_pct, latest_r5_pct, cfg, model,
                )
            except Exception as exc:
                print(f"  iter {it}: critic call FAILED: {exc}")
                break

            # Hash proposal for log
            proposal_blob = json.dumps(proposal, sort_keys=True, ensure_ascii=False)
            proposal_hash = hashlib.sha256(proposal_blob.encode()).hexdigest()[:16]
            (out_dir / "proposals").mkdir(exist_ok=True)
            (out_dir / "proposals" / f"iter_{it:02d}_{proposal_hash}.json").write_text(
                json.dumps(proposal, indent=2)
            )

            # Snapshot pre-iter state for possible revert
            pre_iter_payload = dict(target_payload)
            pre_iter_vec_field = target_pt["vector"]
            pre_iter_vectors = (pre_iter_vec_field.get("colbert")
                                if isinstance(pre_iter_vec_field, dict)
                                else pre_iter_vec_field)
            pre_r1, pre_r3, pre_r5 = latest.rescue_at_1, latest.rescue_at_3, latest.rescue_at_5

            # Cross-cluster baseline: measure each hijacker's currently-rightful wins
            # BEFORE we apply the proposal.  After measuring post-edit, we'll
            # compute neighbor_loss and feed it into the net-positive guard.
            neighbor_codes = [hcode for hcode, _, _ in top_hijackers]
            pre_neighbor = measure_neighbor_protection(
                neighbor_codes, classifications, encoder, client,
            ) if neighbor_codes else {}

            # Apply proposal
            new_payload = merge_proposal_into_payload(target_payload, proposal)
            apply_proposal_to_qdrant(cluster.target_code, target_point_id, new_payload, encoder)

            # Re-measure target + neighbors
            counts, top1_map, top5_map = measure_cluster_rescue(cluster, encoder, client)
            post_neighbor = measure_neighbor_protection(
                neighbor_codes, classifications, encoder, client,
            ) if neighbor_codes else {}

            # Net-positive guard.  An edit must lift the target *at least as
            # much* as it costs neighbors.  Cross-cluster interference is
            # assumed possible by default; rejection happens whenever the
            # neighbor cost exceeds the target gain on rescue@1.
            target_gain_r1 = counts["r1"] - pre_r1
            neighbor_loss = sum(
                max(0, pre_neighbor.get(c, 0) - post_neighbor.get(c, 0))
                for c in neighbor_codes
            )
            net_r1 = target_gain_r1 - neighbor_loss
            improved_anywhere = (counts["r1"] > pre_r1
                                  or counts["r3"] > pre_r3
                                  or counts["r5"] > pre_r5)
            keep = net_r1 > 0 or (improved_anywhere and neighbor_loss == 0)

            if not keep:
                reason = (f"net_r1={net_r1:+d} (target_gain_r1={target_gain_r1:+d}, "
                          f"neighbor_loss={neighbor_loss})")
                if not improved_anywhere:
                    reason += f"; no target metric improved: r1 {pre_r1}→{counts['r1']}, " \
                              f"r3 {pre_r3}→{counts['r3']}, r5 {pre_r5}→{counts['r5']}"
                print(f"  iter {it}: edit REJECTED ({reason}); reverting")
                upsert_point_in_place(target_point_id, pre_iter_vectors, pre_iter_payload)
                counts, top1_map, top5_map = measure_cluster_rescue(cluster, encoder, client)
                edited_nodes = []
                improved = False
            else:
                if neighbor_loss > 0:
                    print(f"  iter {it}: net-positive trade kept "
                          f"(target_gain_r1={target_gain_r1:+d}, neighbor_loss={neighbor_loss}, net={net_r1:+d})")
                edited_nodes = [cluster.target_code]
                improved = True

            # Log the attempt (kept or reverted) so the critic learns from it
            prior_attempts_for_critic.append({
                "iteration": it,
                "diagnosis": proposal.get("diagnosis", ""),
                "edits": proposal.get("edits", {}),
                "rescue_at_5_pct": latest_r5_pct,
                "kept": improved,
            })

        # Increment attempt counter for in-budget rows (these are the rows the critic considered)
        for r in in_budget:
            per_row_attempts[r.key] += 1

        # Compute regressions
        if not dry_run:
            _, _, regressed = check_regressions(regression_sample, encoder, client)
            iter_res = IterationResult(
                iteration=it, rescue_at_1=counts["r1"], rescue_at_3=counts["r3"],
                rescue_at_5=counts["r5"], total=counts["total"], regressions=regressed,
                edited_nodes=edited_nodes, proposal_hash=proposal_hash,
                per_row_top1_after=dict(top1_map), elapsed_s=time.time() - t0,
            )
        else:
            iter_res = IterationResult(
                iteration=it, rescue_at_1=counts["r1"], rescue_at_3=counts["r3"],
                rescue_at_5=counts["r5"], total=counts["total"], regressions=0,
                edited_nodes=[], proposal_hash=None,
                per_row_top1_after=dict(top1_map0), elapsed_s=time.time() - t0,
            )
        iterations_log.append(iter_res)
        log_fh.write(json.dumps(asdict(iter_res)) + "\n")
        log_fh.flush()
        print(f"  iter {it}: rescue@1={iter_res.rescue_at_1}/{iter_res.total} "
              f"rescue@3={iter_res.rescue_at_3}/{iter_res.total} rescue@5={iter_res.rescue_at_5}/{iter_res.total} "
              f"regr={iter_res.regressions} elapsed={iter_res.elapsed_s:.1f}s")

    log_fh.close()

    post_state = state_hash(cluster.target_code)
    reference_post = reference_hash()
    final = iterations_log[-1]
    if final.total:
        final_r1_pct = final.rescue_at_1 / final.total * 100
        final_r3_pct = final.rescue_at_3 / final.total * 100
        final_r5_pct = final.rescue_at_5 / final.total * 100
        objective_met = (final_r1_pct >= rescue_r1_pct
                         and final_r3_pct >= rescue_r3_pct
                         and final_r5_pct >= rescue_r5_pct)
    else:
        objective_met = False

    # Resisting rows = max-attempt rows that are STILL not rescued
    final_top1 = final.per_row_top1_after
    resisting = [
        r for r in cluster.rows
        if per_row_attempts[r.key] >= max_attempts_per_row
        and not is_subtree_match(final_top1.get(r.key), cluster.target_code)
    ]

    return LoopResult(
        target_code=cluster.target_code,
        target_label=cluster.target_label,
        iterations=iterations_log,
        final_rescue_at_5=final.rescue_at_5,
        objective_met=objective_met,
        resisting_rows=resisting,
        per_row_attempts=per_row_attempts,
        pre_state_hash=pre_state,
        post_state_hash=post_state,
        reference_hash_pre=reference_pre,
        reference_hash_post=reference_post,
    )


# ──────────────────────────────────────────────────────────────────────
# Artifacts
# ──────────────────────────────────────────────────────────────────────

def write_summary(result: LoopResult, out_dir: Path) -> None:
    lines = [f"# Semantic-optimization loop summary — target `{result.target_code}`", ""]
    lines.append(f"- target: `{result.target_code}` ({result.target_label})")
    lines.append(f"- objective met: **{result.objective_met}**")
    lines.append(f"- final rescue@5: **{result.final_rescue_at_5} / {result.iterations[-1].total}**")
    lines.append(f"- iterations run: {len(result.iterations) - 1}")
    lines.append(f"- resisting rows: {len(result.resisting_rows)} / {result.iterations[-1].total}")
    lines.append(f"- reference hash pre/post: {result.reference_hash_pre[:8]}…/{result.reference_hash_post[:8]}…")
    lines.append(f"- reference immutability check: **{'PASS' if result.reference_hash_pre == result.reference_hash_post else 'FAIL'}**")
    lines.append("")
    lines.append("## Trajectory\n")
    lines.append("| iter | r@1 | r@3 | r@5 | total | regressions | elapsed_s |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in result.iterations:
        lines.append(f"| {r.iteration} | {r.rescue_at_1} | {r.rescue_at_3} | {r.rescue_at_5} | {r.total} | {r.regressions} | {r.elapsed_s:.1f} |")
    lines.append("")
    if result.resisting_rows:
        lines.append(f"## Resisting rows ({len(result.resisting_rows)})\n")
        for r in result.resisting_rows[:20]:
            lines.append(f"- `{r.key}` (ref={r.reference_code}, attempts={result.per_row_attempts[r.key]})")
    (out_dir / "summary.md").write_text("\n".join(lines))


def write_ml_uplift_report(result: LoopResult, out_dir: Path) -> None:
    first = result.iterations[0]
    final = result.iterations[-1]
    delta_r1 = final.rescue_at_1 - first.rescue_at_1
    delta_r3 = final.rescue_at_3 - first.rescue_at_3
    delta_r5 = final.rescue_at_5 - first.rescue_at_5
    lines = [f"# ML uplift report — target `{result.target_code}`", ""]
    lines.append(f"_Gates the pipeline-restart decision._  Generated {datetime.now(timezone.utc).isoformat()}.")
    lines.append("")
    lines.append("## Cosine channel rescue rates\n")
    lines.append("| metric | pre-loop | post-loop | Δ | post % |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| rescue@1 | {first.rescue_at_1} | {final.rescue_at_1} | {delta_r1:+d} | {final.rescue_at_1/final.total*100:.1f}% |")
    lines.append(f"| rescue@3 | {first.rescue_at_3} | {final.rescue_at_3} | {delta_r3:+d} | {final.rescue_at_3/final.total*100:.1f}% |")
    lines.append(f"| rescue@5 | {first.rescue_at_5} | {final.rescue_at_5} | {delta_r5:+d} | {final.rescue_at_5/final.total*100:.1f}% |")
    lines.append("")
    cumulative_regressions = sum(r.regressions for r in result.iterations)
    lines.append(f"## Regression check\n")
    lines.append(f"- cumulative regressions across iterations: {cumulative_regressions}")
    lines.append(f"- final-iteration regression count: {final.regressions}")
    lines.append("")
    lines.append(f"## Reference immutability\n")
    lines.append(f"- hash pre:  `{result.reference_hash_pre}`")
    lines.append(f"- hash post: `{result.reference_hash_post}`")
    lines.append(f"- match: **{'PASS' if result.reference_hash_pre == result.reference_hash_post else 'FAIL'}**")
    lines.append("")
    lines.append("## Projected DST impact (qualitative)\n")
    lines.append("Cosine top-K is one of six DST evidence channels.  A measurable")
    lines.append("rescue@5 uplift on the target cluster shifts cosine mass toward the")
    lines.append("reference for those rows.  Whether this translates to fused-prediction")
    lines.append("corrections depends on:")
    lines.append("- LLM channel's own pick on the same rows (LLM mass dominates fusion)")
    lines.append("- catboost_fit_to_llm echo of LLM")
    lines.append("- DST discount configuration")
    lines.append("")
    lines.append("Top-1 uplift is more directly DST-relevant than top-5 uplift, since")
    lines.append("DST mass concentrates on the top-K cosine results.  rescue@1 ≥ 30%")
    lines.append("is a reasonable threshold for confidence that DST fusion will see")
    lines.append("meaningful cosine-channel uplift.")
    (out_dir / "ml_uplift.md").write_text("\n".join(lines))


def write_handoff_artifact(result: LoopResult, classifications: list[dict], out_dir: Path) -> Path | None:
    """Conditional: only write if objective missed AND resisting_rows non-empty."""
    if result.objective_met or not result.resisting_rows:
        return None

    # Build a quick lookup for original classification row by (table, column)
    by_key = {f"{r.get('table_name')}.{r.get('column_name')}": r for r in classifications}

    lines = ["# Reference-review candidates — surfaced by semantic-optimization loop", ""]
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat()}._")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- target objective: cluster `{result.target_code}` rescue@5 ≥ threshold")
    lines.append(f"- objective met: **NO** (final rescue@5 = {result.final_rescue_at_5}/{result.iterations[-1].total})")
    lines.append(f"- resisting rows (attempts exhausted, never rescued): **{len(result.resisting_rows)}**")
    lines.append("")
    lines.append("## Per-row candidates\n")
    for r in result.resisting_rows:
        orig = by_key.get(r.key, {})
        lines.append(f"### Row: `{r.key}`")
        lines.append(f"- reference: `{r.reference_code}`")
        lines.append(f"- pipeline prediction: `{orig.get('predicted_code','?')}`  (confidence: {orig.get('confidence','?')})")
        lines.append(f"- attempts: {result.per_row_attempts[r.key]} (max reached)")
        final_top1 = result.iterations[-1].per_row_top1_after.get(r.key)
        lines.append(f"- final cosine top-1: `{final_top1 or 'none'}`")
        lines.append(f"- embedding_text: `{r.embedding_text[:160]}`")
        lines.append("")
    lines.append("## Recommendation\n")
    lines.append("Hand these rows to `/curate-agent-mediated` for reference review.")
    lines.append("The semantic-optimization loop exhausted its enrichment-edit budget")
    lines.append("without rescuing them — they require judgment about whether the")
    lines.append("reference assignment is coherent with the column's semantics.")
    path = out_dir / "handoff_to_curate_agent_mediated.md"
    path.write_text("\n".join(lines))
    return path


# ──────────────────────────────────────────────────────────────────────
# Cohort versioning
# ──────────────────────────────────────────────────────────────────────

def next_cohort_version(out_root: Path, cohort_name: str) -> str:
    existing = sorted(out_root.glob(f"cohort_{cohort_name}_v*"))
    if not existing:
        return "v0"
    last = existing[-1].name
    n = int(last.rsplit("_v", 1)[1])
    return f"v{n + 1}"


# ──────────────────────────────────────────────────────────────────────
# Sampling helpers
# ──────────────────────────────────────────────────────────────────────

def sample_regression_rows(classifications: list[dict], cluster: Cluster, n: int = 20) -> list[dict]:
    """N rows where matches_reference=True AND not in the cluster's failure set."""
    failure_keys = {r.key for r in cluster.rows}
    correct = [c for c in classifications
               if c.get("matches_reference") and c.get("reference_code")
               and c.get("embedding_text")
               and f"{c.get('table_name')}.{c.get('column_name')}" not in failure_keys]
    # Stratify by reference_code
    seen = set()
    out = []
    for c in correct:
        if c["reference_code"] not in seen:
            seen.add(c["reference_code"])
            out.append(c)
            if len(out) >= n:
                break
    return out


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run-dir", default="build/results/46229e17",
                    help="run dir for failure-row + reference source")
    ap.add_argument("--target", required=True,
                    help="target code to optimize toward (e.g. 1.3.2)")
    ap.add_argument("--target-label", default=None,
                    help="human-readable label for the target")
    ap.add_argument("--max-iterations", type=int, default=8)
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--rescue-r1", type=int, default=82,
                    help="objective: rescue@1 percentage threshold (must meet all of r1/r3/r5)")
    ap.add_argument("--rescue-r3", type=int, default=90,
                    help="objective: rescue@3 percentage threshold")
    ap.add_argument("--rescue-r5", type=int, default=94,
                    help="objective: rescue@5 percentage threshold")
    ap.add_argument("--no-improvement-window", type=int, default=3)
    ap.add_argument("--max-neighbors", type=int, default=3,
                    help="Top-N hijackers treated as cross-cluster guard set (default 3). "
                         "Per-iter, the top-N categories winning the failure rows have their "
                         "rightful wins measured pre/post; edits that cost neighbors more "
                         "than they gain target are rejected.  Set to 0 to disable.")
    ap.add_argument("--cohort-name", default=None,
                    help="cohort label (default: target_code with dots→underscores)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--model", default=None,
                    help="LLM model id (defaults to cfg.agent_model or claude-opus-4-7)")
    ap.add_argument("--dry-run", action="store_true",
                    help="measurement only; no LLM call, no Qdrant mutation")
    ap.add_argument("--adversarial", action="store_true",
                    help="adversarial check: mark all rows attempt-exhausted on iter 1")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_dir = Path(args.run_dir)
    classifications = json.loads((run_dir / "classifications.json").read_text())
    print(f"Loaded {len(classifications)} classifications from {run_dir}")

    target_label = args.target_label or args.target
    cluster = build_cluster(args.target, target_label, classifications)
    print(f"Target `{args.target}` ({target_label}): {len(cluster.rows)} failure rows")

    if not cluster.rows:
        print("No failure rows for this target — nothing to optimize.")
        return 0

    cohort_name = args.cohort_name or args.target.replace(".", "_")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    version = next_cohort_version(out_root, cohort_name)
    out_dir = out_root / f"cohort_{cohort_name}_{version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # Encoder + Qdrant client
    from atelier.classify.colbert_encoder import get_encoder
    encoder = get_encoder()
    client = qdrant_client()

    # Config + model
    cfg = None
    model = ANTHROPIC_FALLBACK_MODEL
    if not args.dry_run:
        from atelier.config import load_config
        from enrichment_evolution import resolve_model
        cfg = load_config()
        model = resolve_model(cfg, args.model)
        print(f"LLM model: {model}")

    # Regression sample
    reg_sample = sample_regression_rows(classifications, cluster, n=20)
    print(f"Regression sample: {len(reg_sample)} rows")

    # Save the original target payload as rollback artifact
    pt = get_payload_by_code(args.target)
    if pt:
        rollback = {"target_code": args.target, "point_id": pt["id"],
                    "payload": pt["payload"], "vectors": pt["vector"]}
        (out_dir / "rollback.json").write_text(json.dumps(rollback, indent=2))

    # Run the loop
    result = iterate_with_attempt_counter(
        cluster, encoder=encoder, client=client, cfg=cfg, model=model,
        classifications=classifications,
        max_iterations=args.max_iterations,
        max_attempts_per_row=args.max_attempts,
        rescue_r1_pct=args.rescue_r1,
        rescue_r3_pct=args.rescue_r3,
        rescue_r5_pct=args.rescue_r5,
        no_improvement_window=args.no_improvement_window,
        max_neighbors=args.max_neighbors,
        regression_sample=reg_sample, out_dir=out_dir,
        dry_run=args.dry_run, adversarial=args.adversarial,
    )

    # Artifacts
    write_summary(result, out_dir)
    write_ml_uplift_report(result, out_dir)
    handoff_path = write_handoff_artifact(result, classifications, out_dir)

    # Final report
    print()
    print(f"=== Done.  rescue@5 final = {result.final_rescue_at_5}/{result.iterations[-1].total}, "
          f"objective_met = {result.objective_met} ===")
    if handoff_path:
        print(f"Hand-off artifact: {handoff_path}")
    if result.reference_hash_pre != result.reference_hash_post:
        print(f"*** REFERENCE IMMUTABILITY VIOLATION *** pre={result.reference_hash_pre[:8]} post={result.reference_hash_post[:8]}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
