"""Cautious-code review — agent-mediated backoff for over-specified predictions.

.. warning:: **PROVEN HARMFUL — DO NOT ENABLE IN PRODUCTION.**

   Empirical validation against run ``ce4f3777`` (2026-05-20, 920
   reference columns) demonstrated that cautious review **destroys**
   classification accuracy:

   - **Reroute** (238 columns): 76.1% miss rate.  Hurt 99 columns
     where the LLM was already correct; helped only 19.
   - **Backoff** (80 columns): 78.8% miss rate.  Hurt 50 columns
     where the leaf prediction was correct; helped only 3.
   - **Overall**: fused + review accuracy = **62.5%** vs LLM-only
     accuracy = **76.1%** — cautious review cost **13.6 percentage
     points** of strict accuracy.

   Root cause: when evidence sources conflict (mean K = 0.70 in that
   run), beliefs are low, the bel_threshold fires on nearly every
   column, and the review LLM (Sonnet 4.5 on Bedrock) systematically
   overrides correct predictions with wrong ones.  The cascade is
   self-reinforcing — degraded evidence → high K → low belief →
   mass review → mass damage.

   The ``bel_threshold`` default is set to **0.0** (unreachable) so
   that the review never fires unless an operator explicitly opts in.
   If you are experimenting with re-enabling this logic, be aware
   that every prior attempt to tune it (R2a stability guard, R2c
   shortlist-permissive, R3 opaque-sibling exclusion) failed to
   overcome the fundamental problem: a second LLM call on low-
   confidence evidence introduces more error than it corrects.

When the DST fusion commits to a deeper code than the cautious-belief
threshold supports, the over-specified code may be wrong.  This module
asks Claude (via the same LLM backend the classify pipeline uses, or
optionally direct Anthropic for reasoning-budget access) to consider
whether to back off to the cautious code given the column's values +
sibling context.  The agent's decision and rationale are recorded for
audit: ``predicted_code`` is updated in place when backoff is
recommended, the original is preserved as ``predicted_code_pre_review``,
and ``review_decision`` / ``review_rationale`` are stamped on the
classification dict and into ``cautious_review.json`` beside the run
parquet.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are reviewing a column-classification decision from an "
    "evidence-fusion pipeline (Atelier).  The pipeline assigns each "
    "column a hierarchical taxonomy code from belief-mass fusion across "
    "six evidence sources (name match, value patterns, cosine similarity, "
    "an LLM sweep, CatBoost, SVM).  The fused decision can fail in two "
    "structurally different ways, and a correct review must distinguish "
    "them:\n"
    "\n"
    "  (1) RIGHT SUBTREE, WRONG DEPTH — the predicted code is on the "
    "correct branch but commits too deeply (e.g. 'Shipping City' when "
    "evidence only supports 'City').  Resolution: backoff to the "
    "cautious (parent-or-shallower) code on the same chain.\n"
    "\n"
    "  (2) WRONG SUBTREE — the prediction lands on the wrong branch "
    "entirely (e.g. 'Personal Data' for a column of monetary amounts "
    "that should be 'Financial Data').  Backoff up the wrong chain "
    "compounds the error: 'Personal Data' → 'Personally Identifiable "
    "Data' is shallower but still in the wrong subtree.  Resolution: "
    "REROUTE to a code on the correct chain — preferably one whose "
    "depth matches what the value evidence actually supports.\n"
    "\n"
    "You will be given a shortlist of candidate codes drawn from the "
    "fusion's cross-subtree belief (codes anywhere in the hierarchy "
    "with non-trivial mass).  This is the closed set of valid "
    "destinations; you may not invent codes outside it.\n"
    "\n"
    "Three possible decisions:\n"
    "  - keep    — the predicted code is genuinely supported; do not "
    "swap.\n"
    "  - backoff — same subtree, over-specified; swap to the marked "
    "cautious_code (no separate ``code`` needed; cautious_code is the "
    "canonical safe parent on the same chain).\n"
    "  - reroute — wrong subtree (or right subtree at a depth not on "
    "the cautious_code chain); swap to a specific code from the "
    "shortlist.  REQUIRES a ``code`` field whose value matches one of "
    "the listed shortlist codes exactly.\n"
    "\n"
    "Multi-objective principle: choose the right subtree first, then "
    "the optimal depth within it.  Shallower-but-correct beats "
    "deeper-but-wrong; deeper-and-correct beats shallower-and-correct.\n"
    "\n"
    "Reply with strict JSON matching the schema "
    '{"decision": "keep" | "backoff" | "reroute", '
    '"code": "<required when decision=reroute, must be from shortlist>", '
    '"rationale": "<1-3 sentences citing specific evidence>", '
    '"confidence": <0.0-1.0>}.  No other prose.'
)


_MAX_SHORTLIST_SIZE = 10


def _format_belief_path(path: list[dict]) -> str:
    """Render a belief path leaf→root for the prompt."""
    if not path:
        return "(no belief path available)"
    lines = []
    for entry in path:
        code = entry.get("code", "")
        label = entry.get("label", "")
        bel = entry.get("bel", 0.0)
        pl = entry.get("pl", 0.0)
        lines.append(f"  {code} ({label}): Bel={bel:.3f}, Pl={pl:.3f}")
    return "\n".join(lines)


def _build_shortlist(column: dict) -> list[dict]:
    """Compose the closed set of valid reroute targets for the LLM.

    Sources, in order of priority:

    1. ``cross_subtree_belief`` — the fusion's own structured view of
       codes anywhere in the hierarchy with non-trivial mass.  Already
       carries label / bel / pl / depth / kind for each entry and
       includes the top-per-subtree alternatives that ``belief_path``
       (confined to the predicted leaf's ancestor chain) cannot show.
    2. ``predicted_code`` — fallback inclusion if the cross-subtree
       roll-up didn't already include it.  Should be redundant but
       guards against a fusion variant that drops the predicted code
       from the cross-subtree view.
    3. ``cautious_code`` — same fallback rationale for the cautious
       backoff target.

    Deduplicated by code, sorted most-specific first then by
    descending belief, capped at ``_MAX_SHORTLIST_SIZE`` rows.  The
    cap protects prompt length on dense hierarchies; in practice
    cross-subtree rarely exceeds 8 codes already.
    """
    rows: list[dict] = []
    seen: set[str] = set()

    for r in column.get("cross_subtree_belief") or []:
        code = (r.get("code") or "").strip()
        if not code or code in seen:
            continue
        rows.append({
            "code": code,
            "label": r.get("label") or code,
            "bel": float(r.get("bel", 0.0)),
            "pl": float(r.get("pl", 0.0)),
            "depth": int(r.get("depth", 0)),
            "kind": r.get("kind") or "",
        })
        seen.add(code)

    pred = (column.get("predicted_code") or "").strip()
    if pred and pred not in seen:
        rows.append({
            "code": pred,
            "label": column.get("predicted_label") or pred,
            "bel": float(column.get("belief", 0.0)),
            "pl": float(column.get("plausibility", 0.0)),
            "depth": pred.count("."),
            "kind": "leaf",
        })
        seen.add(pred)

    cautious = (column.get("cautious_code") or "").strip()
    if cautious and cautious not in seen:
        rows.append({
            "code": cautious,
            "label": cautious,
            "bel": 0.0,
            "pl": 0.0,
            "depth": cautious.count("."),
            "kind": "internal",
        })
        seen.add(cautious)

    rows.sort(key=lambda r: (-r["depth"], -r["bel"]))
    return rows[:_MAX_SHORTLIST_SIZE]


def _format_shortlist(
    shortlist: list[dict],
    *,
    predicted_code: str,
    cautious_code: str,
) -> str:
    """Render the shortlist as a fixed-width table with markers."""
    if not shortlist:
        return "(no alternatives — predicted code is the only candidate)"
    lines = ["  code                   Bel    Pl     depth  kind      label"]
    for r in shortlist:
        marker = ""
        if r["code"] == predicted_code and r["code"] == cautious_code:
            marker = " ← predicted_code & cautious_code"
        elif r["code"] == predicted_code:
            marker = " ← predicted_code"
        elif r["code"] == cautious_code:
            marker = " ← cautious_code"
        lines.append(
            f"  {r['code']:<22} {r['bel']:.3f}  {r['pl']:.3f}  "
            f"{r['depth']:>3}    {r['kind']:<8}  {r['label']}{marker}"
        )
    return "\n".join(lines)


def _build_review_prompt(
    column: dict,
    sibling_predictions: list[dict],
    shortlist: list[dict],
) -> str:
    """Build the per-column review prompt."""
    predicted_code = column.get("predicted_code", "") or ""
    cautious_code = column.get("cautious_code", "") or ""

    lines = [
        f"Column: {column.get('table_name','')}.{column.get('column_name','')}",
        f"Type: {column.get('column_type', 'unknown')}",
        "",
        "Current prediction (from DST fusion):",
        f"  predicted_code:  {predicted_code}",
        f"  predicted_label: {column.get('predicted_label', '')}",
        f"  belief:          {float(column.get('belief', 0)):.3f}",
        f"  plausibility:    {float(column.get('plausibility', 0)):.3f}",
        f"  uncertainty gap: {float(column.get('uncertainty', 0)):.3f}",
        "",
        "Cautious-code alternative (deepest code where Bel >= 0.70):",
        f"  cautious_code: {cautious_code}",
        "",
        "Alternatives shortlist — the closed set of valid reroute "
        "targets.  Sorted most-specific first.  ``decision=reroute`` "
        "must pick exactly one ``code`` from this list:",
        _format_shortlist(
            shortlist,
            predicted_code=predicted_code,
            cautious_code=cautious_code,
        ),
        "",
        "Belief path (leaf → root; Bel increases ascending the hierarchy):",
        _format_belief_path(column.get("belief_path", [])),
    ]

    sigs = column.get("pattern_signals") or {}
    if sigs:
        lines.append("")
        lines.append(f"Pattern signals: {sigs}")

    embedding_text = column.get("embedding_text", "")
    if embedding_text:
        lines.append("")
        excerpt = embedding_text[:500]
        if len(embedding_text) > 500:
            excerpt += "..."
        lines.append(f"Value evidence (excerpt): {excerpt}")

    if sibling_predictions:
        lines.append("")
        lines.append(
            f"Sibling columns in {column.get('table_name','')} "
            f"(at most 8 shown):"
        )
        for sib in sibling_predictions[:8]:
            lines.append(
                f"  {sib.get('column_name','')} → {sib.get('predicted_code','')} "
                f"({sib.get('predicted_label','')})"
            )

    lines.extend([
        "",
        "Decide:",
        "  - keep — predicted_code is correct (right subtree, right depth).",
        "  - backoff — same subtree, over-specified; use cautious_code.",
        "  - reroute — different subtree (or different depth not on the "
        "cautious chain); pick a ``code`` from the shortlist.",
        "",
        "Apply the multi-objective principle: right subtree first, "
        "then optimal depth.  If the column's evidence (values, "
        "patterns, ontology priors) points at a different subtree from "
        "predicted_code, prefer reroute over backoff — backing off up "
        "the wrong chain compounds the error.",
    ])
    return "\n".join(lines)


def _extract_json_object(text: str) -> str:
    """Locate and isolate the first balanced JSON object in ``text``.

    R2b (audit_2026-05-06_a): handles two failure modes the previous
    greedy regex (``r"\\{[\\s\\S]*\\}"``) could not:

    1. Bedrock Sonnet wraps responses in `````json … ````` fences with
       trailing prose.  Strip the fence pair (or any opening fence)
       so the JSON body sits in plain text.
    2. Trailing prose / pseudo-JSON commentary after the real object
       caused ``json.loads`` to raise ``Extra data`` even though the
       initial object parses cleanly.  Walk braces with depth to
       extract the *first* balanced object and stop.

    Returns the raw JSON substring (still possibly invalid — the
    caller does the actual ``json.loads``).  Raises ``ValueError`` if
    no balanced object is found.
    """
    body = text.strip()
    fence = re.match(
        r"^\s*```(?:json|JSON)?\s*\n([\s\S]*?)\n\s*```",
        body,
    )
    if fence:
        body = fence.group(1)
    else:
        # Bedrock sometimes emits an opening fence without a matching
        # closing one (truncated response).  Trim that prefix anyway.
        prefix = re.match(r"^\s*```(?:json|JSON)?\s*\n", body)
        if prefix:
            body = body[prefix.end():]

    start = body.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    depth = 0
    in_str = False
    escape = False
    end = -1
    for i in range(start, len(body)):
        ch = body[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise ValueError(
            f"unbalanced JSON object in response: {text[:200]!r}"
        )
    return body[start:end]


def _parse_decision(
    text: str,
    valid_codes: set[str] | None = None,
    *,
    fallback_codes: set[str] | None = None,
) -> dict[str, Any]:
    """Extract a structured decision from agent output.

    Validates:
    - JSON parseable
    - ``decision`` is one of ``keep`` / ``backoff`` / ``reroute``
    - When ``decision == "reroute"``, ``code`` is non-empty and
      (if ``valid_codes`` is supplied) appears in the shortlist set.

      R2c (audit_2026-05-06_a): when ``fallback_codes`` is supplied
      AND the chosen code is outside ``valid_codes`` but inside
      ``fallback_codes``, we accept the code as a *shortlist-extended*
      reroute.  This closes 8/11 errored decisions in 8d67b1ed where
      the LLM correctly identified a code outside the 5-entry
      shortlist but inside the runtime taxonomy.  Hallucinated codes
      (outside both sets) remain rejected.
    """
    raw = _extract_json_object(text)
    try:
        decision = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON ({e}) in: {raw[:200]!r}")
    if decision.get("decision") not in ("keep", "backoff", "reroute"):
        raise ValueError(
            f"decision must be one of 'keep' / 'backoff' / 'reroute', "
            f"got {decision.get('decision')!r}"
        )
    if decision["decision"] == "reroute":
        code = (decision.get("code") or "").strip()
        if not code:
            raise ValueError(
                "decision=reroute requires a non-empty 'code' field"
            )
        if valid_codes is not None and code not in valid_codes:
            if fallback_codes is not None and code in fallback_codes:
                # R2c: accept as shortlist-extended reroute.  Mark it
                # so downstream audit can distinguish from in-shortlist
                # reroutes (which already had cross_subtree_belief
                # evidence behind them).
                decision["shortlist_extended"] = True
            else:
                raise ValueError(
                    f"reroute code {code!r} is not in the shortlist or "
                    f"runtime taxonomy; hallucinated codes outside the "
                    f"valid set are rejected (shortlist has "
                    f"{len(valid_codes)} entries"
                    + (
                        f", taxonomy has {len(fallback_codes)} entries"
                        if fallback_codes is not None else ""
                    )
                    + ")"
                )
        decision["code"] = code
    decision.setdefault("rationale", "(none)")
    decision.setdefault("confidence", None)
    decision.setdefault("shortlist_extended", False)
    return decision


def _resolve_client(cfg, backend_choice: str):
    """Build the Anthropic-compatible client + model for the review.

    Three modes:
      - ``default`` — reuse whichever provider classify is using
        (Bedrock when ``classify_subagent_model`` is an ARN, else
        direct Anthropic).
      - ``anthropic_direct`` — force direct Anthropic API (for
        Opus 4.7 reasoning-budget access; requires
        ``ANTHROPIC_API_KEY``).
      - ``bedrock`` — force Bedrock.
    Returns ``(client, model_id)``.
    """
    import anthropic
    import httpx
    from atelier.config import is_bedrock_model

    timeout = httpx.Timeout(connect=15.0, read=180.0, write=10.0, pool=5.0)

    if backend_choice == "anthropic_direct":
        if not cfg.has_anthropic:
            raise ValueError(
                "cautious_review backend=anthropic_direct requires "
                "ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN"
            )
        client = anthropic.Anthropic(**cfg.anthropic_client_kwargs(), timeout=timeout)
        # Direct Anthropic prefers cfg.agent_model (Opus 4.7 family)
        model = cfg.agent_model
        return client, model

    # default or bedrock — figure out which is configured.  A corporate
    # gateway (base_url + token, no AWS creds) hosts Bedrock-style model
    # IDs over the anthropic protocol, so those route direct in that case.
    classify_model = cfg.classify_subagent_model or cfg.agent_model
    gateway_hosts_bedrock_ids = bool(
        cfg.anthropic_base_url and cfg.has_anthropic and not cfg.has_bedrock
    )
    if backend_choice == "bedrock" or (
        backend_choice == "default"
        and is_bedrock_model(classify_model)
        and not gateway_hosts_bedrock_ids
    ):
        from atelier.agents.client import _build_bedrock_client
        client = _build_bedrock_client(cfg, timeout=180.0)
        return client, classify_model

    # Default with non-Bedrock model → Anthropic direct
    if not cfg.has_anthropic:
        raise ValueError(
            "cautious_review default backend chose Anthropic but "
            "neither ANTHROPIC_API_KEY nor ANTHROPIC_AUTH_TOKEN is set"
        )
    client = anthropic.Anthropic(**cfg.anthropic_client_kwargs(), timeout=timeout)
    return client, classify_model


def _invoke(
    client,
    model: str,
    prompt: str,
    *,
    valid_codes: set[str] | None = None,
    fallback_codes: set[str] | None = None,
) -> dict[str, Any]:
    """Invoke the review agent and parse the JSON decision.

    ``valid_codes`` is the closed set of shortlist codes.
    ``fallback_codes`` (R2c, audit_2026-05-06_a) extends the validity
    check to the full runtime taxonomy: when the LLM picks a code
    outside the shortlist but inside the taxonomy, accept it as a
    shortlist-extended reroute rather than rejecting as hallucination.

    Note: Opus 4.7+ deprecated the ``temperature`` parameter, so we
    don't pass it.  The model is deterministic enough at default
    settings for the keep / backoff / reroute decision.
    """
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    blocks = getattr(resp, "content", None) or []
    text = ""
    for b in blocks:
        if getattr(b, "type", "") == "text":
            text += getattr(b, "text", "")
    if not text:
        raise ValueError("agent returned no text content")
    return _parse_decision(
        text, valid_codes=valid_codes, fallback_codes=fallback_codes,
    )


def review_classifications(
    classifications: list[dict[str, Any]],
    cfg,
    *,
    category_set=None,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Run the Cautious-Code Review pass on a fused classification set.

    .. warning:: **PROVEN HARMFUL — this function should not be called
       in production.**  See module docstring for the empirical evidence
       (run ce4f3777: −13.6pp accuracy).  The default config disables
       this path (enabled=False, bel_threshold=0.0).  If you are here
       because you re-enabled it, you should have a validated fix for
       the fundamental problem first.

    Mutates ``classifications`` in place when the agent recommends a
    swap.  Returns an audit dict for ``cautious_review.json``.
    """
    if not getattr(cfg, "classify_cautious_review_enabled", False):
        return {"enabled": False}

    bel_threshold = float(
        getattr(cfg, "classify_cautious_review_bel_threshold", 0.0)
    )

    logger.warning(
        "Cautious review is ENABLED (proven harmful — see "
        "cautious_review.py docstring).  bel_threshold=%.2f.  "
        "This will review and likely damage predictions.  "
        "Disable via classify.cautious_review.enabled=false.",
        bel_threshold,
    )
    backend_choice = getattr(cfg, "classify_cautious_review_backend", "default")

    # Identify candidates: cautious differs from predicted AND belief is
    # below threshold (the deep code isn't strongly supported).
    candidates: list[dict] = []
    for c in classifications:
        predicted = c.get("predicted_code") or ""
        cautious = c.get("cautious_code") or ""
        bel = float(c.get("belief", 1.0))
        if predicted and cautious and predicted != cautious and bel < bel_threshold:
            candidates.append(c)

    if not candidates:
        logger.info(
            "Cautious review: 0 candidates (bel_threshold=%.2f) — no review needed",
            bel_threshold,
        )
        return {
            "enabled": True,
            "bel_threshold": bel_threshold,
            "backend_choice": backend_choice,
            "candidates": 0,
            "decisions": [],
        }

    # Build sibling map for context
    by_table: dict[str, list[dict]] = {}
    for c in classifications:
        by_table.setdefault(c.get("table_name", ""), []).append(c)

    try:
        client, model = _resolve_client(cfg, backend_choice)
    except Exception as exc:
        logger.warning(
            "Cautious review skipped — could not resolve LLM client: %s", exc,
        )
        return {
            "enabled": True,
            "bel_threshold": bel_threshold,
            "backend_choice": backend_choice,
            "candidates": len(candidates),
            "skipped": True,
            "skip_reason": str(exc),
            "decisions": [],
        }

    logger.info(
        "Cautious review: %d candidates (bel_threshold=%.2f, model=%s)",
        len(candidates), bel_threshold, model,
    )

    def _swap_predicted(col: dict, new_code: str) -> None:
        """Swap predicted_code → new_code, refreshing label / annotation
        / reference-match from the category_set when available.  Used
        identically for backoff (new_code = cautious_code) and reroute
        (new_code = LLM-chosen shortlist code)."""
        col["predicted_code_pre_review"] = col.get("predicted_code", "")
        col["predicted_code"] = new_code
        if category_set is not None:
            new_cat = (
                getattr(category_set, "all_by_code", {}).get(new_code)
                or getattr(category_set, "by_code", {}).get(new_code)
            )
            if new_cat is not None:
                col["predicted_label"] = getattr(new_cat, "label", new_code)
                col["predicted_annotation"] = getattr(new_cat, "abbrev", "") or ""
        ref = col.get("reference_code")
        if ref:
            col["matches_reference"] = (col["predicted_code"] == ref)

    # R2c: full-taxonomy fallback set for the parser's permissive mode.
    # Built once per run from the category_set; covers leaf + internal
    # codes the LLM might select outside the cross_subtree_belief
    # shortlist.
    permissive = bool(getattr(
        cfg, "classify_cautious_review_shortlist_permissive", True,
    ))
    fallback_codes_full: set[str] | None = None
    if permissive and category_set is not None:
        fallback_codes_full = set(getattr(category_set, "by_code", {}).keys())
        all_by_code = getattr(category_set, "all_by_code", None)
        if all_by_code:
            fallback_codes_full |= set(all_by_code.keys())

    # R3: sibling-exclusion regex.  Drops opaque-named siblings from
    # the reviewer's prompt context to break the col_04 sibling-
    # context-poisoning class.
    exclude_opaque = bool(getattr(
        cfg, "classify_cautious_review_exclude_opaque_siblings", True,
    ))
    opaque_re = re.compile(
        r"^(?:attr|code|col|data|field|item|key|ref|val|var|dim)_?\d+(?:_\d+)*$",
        re.IGNORECASE,
    )

    # R2a: stability-guard config — when fusion + LLM converged with
    # high confidence, reject reroute as likely LLM-fluctuation noise.
    stability_guard_on = bool(getattr(
        cfg, "classify_cautious_review_stability_guard_enabled", True,
    ))
    stability_guard_conf = float(getattr(
        cfg, "classify_cautious_review_stability_guard_llm_conf", 0.80,
    ))

    decisions: list[dict] = []
    for i, col in enumerate(candidates):
        siblings_all = [
            s for s in by_table.get(col.get("table_name", ""), [])
            if s.get("column_name") != col.get("column_name")
        ]
        if exclude_opaque:
            siblings = [
                s for s in siblings_all
                if not opaque_re.match(str(s.get("column_name", "")))
            ]
            # Floor: when filtering would drop ALL siblings (e.g. a
            # table of pure col_NN columns), keep at least 3 to give
            # the reviewer some context.  Better-than-nothing semantics;
            # the reviewer's prompt warns about opaque names so it
            # doesn't trust them as authoritative.
            if siblings_all and not siblings:
                siblings = siblings_all[:3]
        else:
            siblings = siblings_all
        shortlist = _build_shortlist(col)
        valid_codes = {r["code"] for r in shortlist}
        prompt = _build_review_prompt(col, siblings, shortlist)

        if progress_callback:
            try:
                progress_callback({
                    "phase": "cautious_review",
                    "candidate_index": i,
                    "candidates_total": len(candidates),
                    "column": (
                        f"{col.get('table_name','')}."
                        f"{col.get('column_name','')}"
                    ),
                })
            except Exception:
                pass

        try:
            decision = _invoke(
                client, model, prompt,
                valid_codes=valid_codes,
                fallback_codes=fallback_codes_full,
            )
        except Exception as exc:
            logger.warning(
                "Cautious review failed for %s.%s: %s — keeping predicted_code",
                col.get("table_name", ""), col.get("column_name", ""), exc,
            )
            col["review_decision"] = "error"
            col["review_rationale"] = f"agent invocation failed: {exc}"[:300]
            decisions.append({
                "column": col.get("column_name", ""),
                "table": col.get("table_name", ""),
                "decision": "error",
                "rationale": col["review_rationale"],
                "predicted_code": col.get("predicted_code", ""),
                "cautious_code": col.get("cautious_code", ""),
            })
            continue

        # R2a stability guard — fires *only* on cross-subtree reroutes
        # where fusion + LLM already converged with high confidence on
        # the pre-review code.  Within-subtree reroutes (depth
        # corrections — pre and chosen share a top-level root) are
        # legitimate and pass through.  This narrowing matters: paper-
        # trading against 8d67b1ed showed several reroutes that
        # superficially looked like noise (LLM and fusion agreed) were
        # actually depth corrections like
        # ``1.1.1.8.2 → 1.1.1.8`` (PIN → Authentication parent).  Only
        # genuine cross-subtree movements (the gaming_profiles.handle
        # failure class) get blocked.
        guard_fired = False
        if (
            stability_guard_on
            and decision["decision"] == "reroute"
            and (col.get("predicted_code") or "") == (col.get("llm_code") or "")
            and float(col.get("llm_confidence", 0.0) or 0.0) >= stability_guard_conf
            and (col.get("predicted_code") or "")
        ):
            pre = col.get("predicted_code", "")
            chosen = (decision.get("code") or "").strip()
            # Cross-subtree iff neither is an ancestor of the other.
            # Within-subtree depth corrections (one is a proper prefix
            # of the other, separated by ".") pass through:
            #   1.1.1.8.2 → 1.1.1.8 — backoff, allow.
            #   1.1.1.8   → 1.1.1.8.2 — over-deepening, allow.
            #   0.1       → 1.1.2.x — true cross-subtree, fire.
            #   1.1.2.3   → 1.1.1.4.4 — sideways move, fire.
            def _is_ancestor(a: str, b: str) -> bool:
                return bool(a) and bool(b) and (a == b or b.startswith(a + "."))
            cross_subtree = bool(pre) and bool(chosen) and not (
                _is_ancestor(pre, chosen) or _is_ancestor(chosen, pre)
            )
            if cross_subtree:
                guard_fired = True
                original_rationale = decision.get("rationale", "")
                decision["decision"] = "keep"
                decision["rationale"] = (
                    f"[R2a stability guard fired: pre-review code "
                    f"{pre!r} matches llm_code at "
                    f"conf={col.get('llm_confidence', 0.0):.2f} ≥ "
                    f"{stability_guard_conf:.2f}; cross-subtree reroute "
                    f"to {chosen!r} (no ancestor relationship) rejected "
                    f"as likely LLM-fluctuation noise. Original reroute "
                    f"rationale: {original_rationale[:200]}]"
                )

        col["review_decision"] = decision["decision"]
        col["review_rationale"] = decision["rationale"][:500]

        chosen_code: str | None = None
        if decision["decision"] == "backoff":
            chosen_code = col.get("cautious_code", "") or ""
            if chosen_code:
                _swap_predicted(col, chosen_code)
        elif decision["decision"] == "reroute":
            chosen_code = decision.get("code") or ""
            # _parse_decision rejects empty codes; in shortlist-permissive
            # mode it accepts codes inside the runtime taxonomy but
            # outside the shortlist — those carry decision["shortlist_extended"]=True.
            if chosen_code:
                _swap_predicted(col, chosen_code)
                col["review_chosen_code"] = chosen_code

        decisions.append({
            "column": col.get("column_name", ""),
            "table": col.get("table_name", ""),
            "decision": decision["decision"],
            "rationale": decision.get("rationale", ""),
            "confidence": decision.get("confidence"),
            "predicted_code": col.get("predicted_code", ""),
            "cautious_code": col.get("cautious_code", ""),
            "chosen_code": chosen_code,
            "predicted_code_pre_review": col.get("predicted_code_pre_review"),
            "shortlist_size": len(shortlist),
            "shortlist_extended": bool(decision.get("shortlist_extended", False)),
            "stability_guard_fired": guard_fired,
            "siblings_after_filter": len(siblings),
            "siblings_unfiltered": len(siblings_all),
        })

    backed_off = sum(1 for d in decisions if d.get("decision") == "backoff")
    rerouted = sum(1 for d in decisions if d.get("decision") == "reroute")
    kept = sum(1 for d in decisions if d.get("decision") == "keep")
    errored = sum(1 for d in decisions if d.get("decision") == "error")
    shortlist_extended_n = sum(1 for d in decisions if d.get("shortlist_extended"))
    stability_guard_fired_n = sum(1 for d in decisions if d.get("stability_guard_fired"))

    logger.info(
        "Cautious review complete: %d/%d backed off, %d rerouted, %d kept, %d errored "
        "(R2a stability_guard fired: %d, R2c shortlist-extended: %d)",
        backed_off, len(decisions), rerouted, kept, errored,
        stability_guard_fired_n, shortlist_extended_n,
    )

    return {
        "enabled": True,
        "bel_threshold": bel_threshold,
        "backend_choice": backend_choice,
        "model": model,
        "candidates": len(candidates),
        "backed_off": backed_off,
        "rerouted": rerouted,
        "kept": kept,
        "errored": errored,
        "shortlist_extended": shortlist_extended_n,
        "stability_guard_fired": stability_guard_fired_n,
        "decisions": decisions,
    }
