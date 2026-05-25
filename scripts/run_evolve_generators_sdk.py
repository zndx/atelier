#!/usr/bin/env python3
"""scripts/run_evolve_generators_sdk.py — Phase B of the corpus expansion plan.

Spawns the Agent SDK against the /evolve-generators skill to author
synthetic data generators for the gap list produced by
``scripts/audit_generator_coverage.py``.  After the agent run, this
script:

  1. Validates each proposal JSON under
     ``build/svm_corpus_v2/proposals/`` through AST + sandbox + output-
     shape gates (imported from ``svm_generator_experiment``).
  2. Persists accepted generators incrementally to
     ``build/lib/generated/generators_v1.py``.
  3. Reports per-code acceptance vs rejection.

Resume-safe at the proposal-file level (agent skips authored codes)
AND at the persistence level (this script preserves prior accepted
generators in generators_v1.py and merges).

Model + credential resolution follows the same pattern as
``scripts/run_curate_agent_sdk.py`` — honors ``ATELIER_AGENT_MODEL``
(Bedrock ARN on CAI App pod) via ``_build_sdk_env``.

Usage:
  python scripts/run_evolve_generators_sdk.py
  python scripts/run_evolve_generators_sdk.py --skip-agent  # validate-only
  python scripts/run_evolve_generators_sdk.py --no-persist  # dry run
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path("/home/cdsw")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

SKILL_PATH = PROJECT_ROOT / ".claude/commands/evolve-generators.md"
AUDIT_JSON = Path("build/svm_corpus_v2/coverage_audit.json")
PROPOSALS_DIR = Path("build/svm_corpus_v2/proposals")
GENERATED_PKG = Path("build/lib/generated")
GENERATORS_V1 = GENERATED_PKG / "generators_v1.py"
ACCEPTANCE_LOG = Path("build/svm_corpus_v2/acceptance_log.json")

log = logging.getLogger("run_evolve_generators_sdk")


def sanitize_code(code: str) -> str:
    """Convert 1.1.1.9.3.1 → 1_1_1_9_3_1 (matches skill's convention)."""
    return code.replace(".", "_")


# ──────────────────────────────────────────────────────────────────────
# Agent SDK invocation
# ──────────────────────────────────────────────────────────────────────

async def _run_agent() -> int:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )
    from atelier.agents.client import _build_sdk_env
    from atelier.config import load_config
    from atelier.model_compat import requires_adaptive_thinking
    from atelier.terminal_selection import active_model_ref

    cfg = load_config()
    model = active_model_ref(cfg) or os.environ.get("ATELIER_AGENT_MODEL", "")
    if not model:
        print("ERROR: no agent model resolved (set ATELIER_AGENT_MODEL or "
              "configure agents.model)", file=sys.stderr)
        return 2

    env = _build_sdk_env(cfg, selected_model=model)

    if not SKILL_PATH.exists():
        print(f"ERROR: skill file not found at {SKILL_PATH}", file=sys.stderr)
        return 2

    prompt = (
        SKILL_PATH.read_text()
        + "\n\n---\n\n"
        + "Execute this skill end-to-end.  Read coverage_audit.json, identify "
        + "the gap list, and author proposals for as many gap codes as you can "
        + "in this session (target: 30+ per session; the wrapper will re-invoke "
        + "you if more remain).  Skip any code whose proposal file already exists. "
        + "When you finish, emit a summary listing codes you processed."
    )

    thinking_kwargs: dict = {}
    if requires_adaptive_thinking(model):
        thinking_kwargs["max_thinking_tokens"] = 0
        thinking_kwargs["effort"] = "high"

    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="bypassPermissions",
        model=model,
        max_turns=None,
        cwd=str(PROJECT_ROOT),
        setting_sources=["project"],
        env=env,
        **thinking_kwargs,
    )

    print(f"=== run_evolve_generators_sdk ===")
    print(f"  model: {model}")
    print(f"  skill: {SKILL_PATH}")
    print(f"  cwd  : {PROJECT_ROOT}")
    print()

    rc = 0
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(block.text)
        elif isinstance(msg, ResultMessage):
            if getattr(msg, "is_error", False):
                print("=== agent ERROR ===", file=sys.stderr)
                rc = 1
            else:
                cost = getattr(msg, "total_cost_usd", None)
                turns = getattr(msg, "num_turns", None)
                print(f"=== agent done (turns={turns}, cost_usd={cost}) ===")
    return rc


# ──────────────────────────────────────────────────────────────────────
# Proposal validation (reuses svm_generator_experiment helpers)
# ──────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────
# Reference-shape divergence — catches categorical-prior override
# ──────────────────────────────────────────────────────────────────────

# Empirically calibrated 2026-05-25 from the 94-code survey of the first
# convergence cycle's authored proposals (15 severe + 23 concerning out
# of 94 with reference data).  Severity bands:
#   ≤ 0.2  good distribution match
#   0.2-0.5  minor variation
#   0.5-1.0  concerning shape mismatch (often character-class wrong)
#   > 1.0   severe override (agent invented its own interpretation)
DIVERGENCE_REJECT_THRESHOLD = 1.0   # severe overrides rejected outright
DIVERGENCE_WARN_THRESHOLD = 0.5     # concerning shape mismatches logged


def _shape_signature(values: list[str]) -> dict | None:
    """Compact character-class + length signature for a value set."""
    if not values:
        return None
    total = 0
    digit = upper = lower = symbol = space = 0
    lengths = []
    for v in values:
        s = str(v)
        lengths.append(len(s))
        for c in s:
            total += 1
            if c.isdigit(): digit += 1
            elif c.isupper(): upper += 1
            elif c.islower(): lower += 1
            elif c.isspace(): space += 1
            else: symbol += 1
    total = max(total, 1)
    return {
        "digit_frac": digit / total,
        "upper_frac": upper / total,
        "lower_frac": lower / total,
        "symbol_frac": symbol / total,
        "space_frac": space / total,
        "len_mean": sum(lengths) / max(len(lengths), 1),
    }


def _shape_divergence(sig_a: dict, sig_b: dict) -> float | None:
    """L1 over character-class fractions + relative length-mean delta.

    Symmetric, in [0, ~3].  0 = identical distributions.
    """
    if not sig_a or not sig_b:
        return None
    char_keys = ["digit_frac", "upper_frac", "lower_frac", "symbol_frac", "space_frac"]
    char_dist = sum(abs(sig_a[k] - sig_b[k]) for k in char_keys)
    len_a = sig_a["len_mean"]
    len_b = sig_b["len_mean"]
    len_dist = abs(len_a - len_b) / max(max(len_a, len_b), 1)
    return char_dist + len_dist


def _reference_values_for_code(code: str) -> list[str]:
    """Pull actual hive-poc reference values for the code, from the
    coverage audit's `reference_value_samples` field.  Returns flat
    list across reference columns, or empty list if no reference data.
    """
    if not AUDIT_JSON.exists():
        return []
    try:
        audit = json.loads(AUDIT_JSON.read_text())
    except Exception:
        return []
    for node in audit.get("per_node", []):
        if node.get("code") == code:
            vals = []
            for col in node.get("reference_value_samples", []):
                vals.extend(col.get("values", []))
            return vals
    return []


def validate_proposal(proposal: dict) -> tuple[bool, str, list[dict]]:
    """Validate a per-code proposal through AST + sandbox + output-shape
    + reference-shape-divergence.

    Returns (ok, reason_if_fail, accepted_generators).  A proposal is
    accepted if at least one of its generator variants passes all gates;
    failed variants are dropped from the accepted list.

    The reference-shape-divergence gate catches the categorical-prior-
    override pattern: when the agent invents its own interpretation of
    a code's semantics rather than matching the reference value
    distribution.  Generators with divergence > 1.0 from reference are
    rejected; 0.5-1.0 are logged as concerning but accepted (the
    metrology + refinement loop will catch them via accuracy regression
    if they're truly wrong).
    """
    from svm_generator_experiment import (
        validate_ast, sandbox_exec, validate_output_shape,
    )

    code = proposal.get("code")
    if not code:
        return False, "missing 'code' field", []

    generators = proposal.get("generators", [])
    if not generators or not isinstance(generators, list):
        return False, "missing/empty 'generators' list", []

    # Reference values for divergence check (empty list if no reference
    # data — the gate is then skipped and the agent's authorship is
    # accepted on AST + sandbox alone).
    ref_values = _reference_values_for_code(code)
    ref_sig = _shape_signature(ref_values) if ref_values else None

    accepted: list[dict] = []
    rejection_reasons: list[str] = []
    for gen_spec in generators:
        fn_name = gen_spec.get("function_name", "")
        gen_code = gen_spec.get("code", "")
        if not fn_name or not gen_code:
            rejection_reasons.append(f"{fn_name or '<unnamed>'}: missing function_name or code")
            continue

        ok_ast, reason_ast = validate_ast(gen_code)
        if not ok_ast:
            rejection_reasons.append(f"{fn_name}: AST rejected — {reason_ast}")
            continue

        ok_sb, reason_sb, outputs = sandbox_exec(gen_code, fn_name, n=50)
        if not ok_sb:
            rejection_reasons.append(f"{fn_name}: sandbox rejected — {reason_sb}")
            continue

        # Output-shape gate without source values (use n>0 + diversity check
        # only; length-distribution gate doesn't apply for gap codes with no
        # source row).
        ok_shape, reason_shape = validate_output_shape(outputs, source_values=[])
        if not ok_shape:
            rejection_reasons.append(f"{fn_name}: shape rejected — {reason_shape}")
            continue

        # Reference-shape-divergence gate (only when reference values exist)
        divergence: float | None = None
        if ref_sig is not None:
            gen_sig = _shape_signature(outputs)
            divergence = _shape_divergence(ref_sig, gen_sig)
            if divergence is not None and divergence > DIVERGENCE_REJECT_THRESHOLD:
                rejection_reasons.append(
                    f"{fn_name}: shape-divergence {divergence:.2f} > "
                    f"{DIVERGENCE_REJECT_THRESHOLD} — outputs don't match "
                    f"reference distribution.  Reference shape "
                    f"(char-class fractions): "
                    f"d={ref_sig['digit_frac']:.2f} u={ref_sig['upper_frac']:.2f} "
                    f"l={ref_sig['lower_frac']:.2f} s={ref_sig['symbol_frac']:.2f}, "
                    f"len_mean={ref_sig['len_mean']:.1f}; generator: "
                    f"d={gen_sig['digit_frac']:.2f} u={gen_sig['upper_frac']:.2f} "
                    f"l={gen_sig['lower_frac']:.2f} s={gen_sig['symbol_frac']:.2f}, "
                    f"len_mean={gen_sig['len_mean']:.1f}.  Categorical-prior "
                    f"override suspected — re-author with explicit attention "
                    f"to the reference_value_samples distribution."
                )
                continue
            if divergence is not None and divergence > DIVERGENCE_WARN_THRESHOLD:
                log.warning(
                    "  %s: shape-divergence %.2f in warn band (%.1f-%.1f) — "
                    "accepted but flagged for refinement-loop attention",
                    fn_name, divergence,
                    DIVERGENCE_WARN_THRESHOLD, DIVERGENCE_REJECT_THRESHOLD,
                )

        accepted.append({
            "code": code,
            "function_name": fn_name,
            "variant_label": gen_spec.get("variant_label", ""),
            "code_body": gen_code,
            "reference_shape_divergence": (round(divergence, 4)
                                            if divergence is not None else None),
        })

    if not accepted:
        return False, "; ".join(rejection_reasons) or "no variants accepted", []
    return True, "", accepted


# ──────────────────────────────────────────────────────────────────────
# Persistence to build/lib/generated/generators_v1.py
# ──────────────────────────────────────────────────────────────────────

def load_existing_v1() -> dict:
    """Load existing generators_v1.py state by re-importing its dicts.

    Returns {generator_code_blocks: list[str], generators_by_code,
    name_variants_by_code, table_context_by_code}.  Used for resume
    safety + incremental merging.
    """
    if not GENERATORS_V1.exists():
        return {
            "generator_code_blocks": [],
            "generators_by_code": {},
            "name_variants_by_code": {},
            "table_context_by_code": {},
            "fn_names_seen": set(),
        }
    # Re-parse the file to extract its dicts.  Cheap + avoids module
    # re-import side effects.
    src = GENERATORS_V1.read_text()
    out = {
        "generator_code_blocks": [],
        "generators_by_code": {},
        "name_variants_by_code": {},
        "table_context_by_code": {},
        "fn_names_seen": set(),
    }
    # Extract all `def generate_*(rng):` blocks.  Simple but works for
    # the format this script writes.
    import re as _re
    blocks = _re.findall(
        r"(def generate_\w+\(rng[^)]*\)[^\n]*:\n(?:    .*\n|\n)+?)(?=\ndef |\n\w|\Z)",
        src,
    )
    for blk in blocks:
        out["generator_code_blocks"].append(blk.strip() + "\n")
        m = _re.match(r"def (\w+)\(", blk)
        if m:
            out["fn_names_seen"].add(m.group(1))
    # Extract dicts via exec in a sandboxed namespace.
    ns = {}
    try:
        # Only run the dict-assignment portion; skip function defs to avoid
        # name resolution issues.  Cheap approach: exec the whole file in
        # a namespace where the function names defer to identity-stubs.
        exec(compile(src, str(GENERATORS_V1), "exec"), ns)
        out["generators_by_code"] = dict(ns.get("GENERATORS_BY_CODE", {}))
        out["name_variants_by_code"] = dict(ns.get("NAME_VARIANTS_BY_CODE", {}))
        out["table_context_by_code"] = dict(ns.get("TABLE_CONTEXT_BY_CODE", {}))
    except Exception as exc:
        log.warning("Could not re-import existing generators_v1.py: %s", exc)
    return out


def persist_v1(
    new_acceptances: list[tuple[dict, list[dict]]],
    existing: dict,
) -> None:
    """Merge new accepted proposals into generators_v1.py."""
    GENERATED_PKG.mkdir(parents=True, exist_ok=True)

    header = f'''"""Opus-evolved synthetic data generators (Phase B output).

Auto-generated by scripts/run_evolve_generators_sdk.py.
Last updated: {datetime.now(timezone.utc).isoformat()}.

Three dicts:
- GENERATORS_BY_CODE: dict[code, list[Callable[[random.Random], str]]]
- NAME_VARIANTS_BY_CODE: dict[code, list[str]]
- TABLE_CONTEXT_BY_CODE: dict[code, list[{{table_names, siblings}}]]
"""
import random
import string
import re


'''
    # Merge code blocks (deduplicate by function name)
    code_blocks = list(existing["generator_code_blocks"])
    fn_names_seen = set(existing["fn_names_seen"])
    generators_by_code: dict[str, list[str]] = {}
    for code, fns in existing["generators_by_code"].items():
        if isinstance(fns, list):
            generators_by_code[code] = list(fns)
        else:
            generators_by_code[code] = [str(fns)]  # backward-compat single-fn
    name_variants_by_code = dict(existing["name_variants_by_code"])
    table_context_by_code = dict(existing["table_context_by_code"])

    for proposal, accepted in new_acceptances:
        code = proposal["code"]
        # Reset the per-code state with new proposal's data (latest wins)
        fn_list_for_code: list[str] = []
        for gen in accepted:
            if gen["function_name"] not in fn_names_seen:
                code_blocks.append(gen["code_body"].strip() + "\n")
                fn_names_seen.add(gen["function_name"])
            fn_list_for_code.append(gen["function_name"])
        generators_by_code[code] = fn_list_for_code
        if proposal.get("name_variants"):
            name_variants_by_code[code] = list(proposal["name_variants"])
        if proposal.get("table_context_templates"):
            table_context_by_code[code] = list(proposal["table_context_templates"])

    # Build the module body
    parts = [header]
    for blk in code_blocks:
        parts.append(blk + "\n")

    parts.append("\nGENERATORS_BY_CODE: dict[str, list[callable]] = {\n")
    for code, fn_list in sorted(generators_by_code.items()):
        joined = ", ".join(fn_list) if fn_list else ""
        parts.append(f'    "{code}": [{joined}],\n')
    parts.append("}\n\n")

    parts.append("NAME_VARIANTS_BY_CODE: dict[str, list[str]] = {\n")
    for code, names in sorted(name_variants_by_code.items()):
        parts.append(f'    "{code}": {json.dumps(names)},\n')
    parts.append("}\n\n")

    parts.append("TABLE_CONTEXT_BY_CODE: dict[str, list[dict]] = {\n")
    for code, templates in sorted(table_context_by_code.items()):
        parts.append(f'    "{code}": {json.dumps(templates)},\n')
    parts.append("}\n\n")

    all_fns = sorted(fn_names_seen)
    parts.append('__all__ = ["GENERATORS_BY_CODE", "NAME_VARIANTS_BY_CODE", '
                  '"TABLE_CONTEXT_BY_CODE"] + [\n')
    for fn in all_fns:
        parts.append(f'    "{fn}",\n')
    parts.append("]\n")

    GENERATORS_V1.write_text("".join(parts))


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-agent", action="store_true",
                    help="Don't spawn agent; only validate + persist existing proposals")
    ap.add_argument("--no-persist", action="store_true",
                    help="Dry run — validate but don't write generators_v1.py")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not AUDIT_JSON.exists():
        log.error("Coverage audit missing at %s. Run "
                  "scripts/audit_generator_coverage.py first.", AUDIT_JSON)
        return 2

    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_agent:
        log.info("=== Phase B.1: spawn Agent SDK ===")
        try:
            rc = asyncio.run(_run_agent())
        except KeyboardInterrupt:
            log.warning("agent interrupted")
            rc = 130
        if rc != 0:
            log.error("agent run exited rc=%d; proceeding with validation of "
                      "any proposals written before exit", rc)

    log.info("=== Phase B.2: validate proposals ===")
    proposal_files = sorted(PROPOSALS_DIR.glob("*.json"))
    log.info("  found %d proposal files", len(proposal_files))

    new_acceptances: list[tuple[dict, list[dict]]] = []
    rejected: list[dict] = []
    accepted_count = 0
    rejected_count = 0
    for pf in proposal_files:
        try:
            proposal = json.loads(pf.read_text())
        except Exception as exc:
            rejected.append({"file": str(pf), "reason": f"JSON parse error: {exc}"})
            rejected_count += 1
            continue
        ok, reason, accepted_gens = validate_proposal(proposal)
        if ok:
            new_acceptances.append((proposal, accepted_gens))
            accepted_count += 1
            log.info("  ✓ %s — %d generators accepted",
                     proposal.get("code"), len(accepted_gens))
        else:
            rejected.append({
                "file": str(pf),
                "code": proposal.get("code"),
                "reason": reason,
            })
            rejected_count += 1
            log.warning("  ✗ %s — %s", proposal.get("code"), reason[:200])

    log.info("Validation summary: accepted=%d  rejected=%d",
             accepted_count, rejected_count)

    if not args.no_persist and new_acceptances:
        log.info("=== Phase B.3: persist to generators_v1.py ===")
        existing = load_existing_v1()
        persist_v1(new_acceptances, existing)
        log.info("Wrote %s", GENERATORS_V1)

    ACCEPTANCE_LOG.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_proposals": len(proposal_files),
        "n_accepted": accepted_count,
        "n_rejected": rejected_count,
        "rejected": rejected,
        "accepted_codes": [p["code"] for p, _ in new_acceptances],
    }, indent=2))
    log.info("Wrote %s", ACCEPTANCE_LOG)

    print()
    print(f"=== Phase B complete ===")
    print(f"  proposals processed: {len(proposal_files)}")
    print(f"  accepted: {accepted_count}")
    print(f"  rejected: {rejected_count}")
    if not args.no_persist:
        print(f"  generators_v1.py: {GENERATORS_V1}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
