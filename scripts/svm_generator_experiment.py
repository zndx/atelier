"""SVM generator-code experiment — helper library (legacy main retired).

This module is now a **helper-only library** of Agent-SDK + AST + sandbox
+ persistence + critic-prompt functions used by the dual-gate pipeline
in ``scripts/run_corpus_expansion_pipeline.sh``.  The legacy ``main()``
that ran an offline LinearSVC training + comparison is REMOVED to
prevent silent regression to the pre-factorized-NHSVM behavior.

Active call sites:
  - ``scripts/run_evolve_generators_sdk.py`` — imports
    ``validate_ast``, ``sandbox_exec``, ``validate_output_shape``
  - ``scripts/update_reference_from_xlsx.py`` — defines its own
    ``build_user_prompt`` (not imported from here)

Functions preserved as a library:
  - ``parse_query``, ``load_confusables_map``, ``load_confusable_samples``
  - ``build_user_prompt``, ``_confusables_block``, ``call_generator_critic``
  - ``validate_ast``, ``sandbox_exec``, ``validate_output_shape``
  - ``persist_generators``

The current SVM-stage entry point is::

    just optimize svm   # → bash scripts/run_corpus_expansion_pipeline.sh

The factorized NHSVM head training happens in
``scripts/reflect_nhsvm_eval_shap_v2.py`` (Phase D); the agent
authoring loop runs from ``scripts/run_evolve_generators_sdk.py``;
deployment exit gates are
``scripts/refine_generators_from_failures.py`` (Gate A — TARGET_ACCURACY)
and ``scripts/svm_cosine_uplift_gate.py`` (Gate B — DEPLOYMENT_READY).

Do NOT add a new ``main()`` here.  Per
``feedback_no_silent_dst_degradation``, any reactivation of legacy
SVM training paths must be explicit operator action with a fresh
review, not a silent script invocation.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import logging
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/home/cdsw/src")
sys.path.insert(0, "/home/cdsw/scripts")

logger = logging.getLogger(__name__)

LIVE_CORPUS = Path("/home/cdsw/build/data/svm_training/corpus")
EXPERIMENT_ROOT = Path("/home/cdsw/build/svm_generator_experiment")
GENERATED_PKG = Path("/home/cdsw/build/lib/generated")
RUN_DIR = Path("/home/cdsw/build/results/46229e17")
REVIEW_SUBMISSION = Path("/home/cdsw/build/semantic_optimization/review_submission.json")


# ──────────────────────────────────────────────────────────────────────
# Phase 0: candidate selection
# ──────────────────────────────────────────────────────────────────────

def load_candidates(max_codes: int = 9) -> list[dict]:
    """CAPABILITY_FLOOR rows grouped by reference_code (one entry per code)."""
    out_by_code: dict[str, dict] = {}
    if REVIEW_SUBMISSION.exists():
        review = json.loads(REVIEW_SUBMISSION.read_text())
        for r in review:
            if r.get("category") != "CAPABILITY_FLOOR" or not r.get("reference_code"):
                continue
            code = r["reference_code"]
            if code in out_by_code:
                # accumulate sample rows so the critic has multiple shape examples
                out_by_code[code].setdefault("aux_rows", []).append(r)
                continue
            out_by_code[code] = {
                "reference_code": code,
                "target_label": r.get("target_label"),
                "target_mnemonic": r.get("target_mnemonic"),
                "primary_row": r,
                "aux_rows": [],
            }
    return list(out_by_code.values())[:max_codes]


def parse_query(text: str) -> dict:
    parts = [p.strip() for p in (text or "").split("|")]
    out = {"column_name_humanized": "", "column_type": "",
           "sample_values": [], "cardinality": None}
    if parts:
        out["column_name_humanized"] = parts[0]
    if len(parts) > 1:
        out["column_type"] = parts[1]
    if len(parts) > 2:
        out["sample_values"] = [v.strip() for v in parts[2].split(",") if v.strip()][:10]
    for p in parts[3:]:
        if p.startswith("cardinality="):
            try:
                out["cardinality"] = int(p.split("=", 1)[1])
            except ValueError:
                pass
    return out


# ──────────────────────────────────────────────────────────────────────
# Phase 1: Opus generates Python generator function code
# ──────────────────────────────────────────────────────────────────────

GENERATOR_SYSTEM_PROMPT = """You produce two artifacts for an NHSVM classifier's synthetic training corpus:

1. A Python GENERATOR FUNCTION that produces VALUES for the target class.
2. A list of plausible COLUMN NAMES that real-world columns of this class would have.

The SVM's training text per column is `column_name | column_type | val1, val2, val3, val4, val5` — so BOTH the column name tokens AND the value tokens drive learning.  Without realistic column names, the SVM cannot generalize from the synth corpus to real-world columns.

## Generator function

A `Callable[[random.Random], str]` that takes a random.Random and returns ONE string value.  Calling it many times must produce a DIVERSE distribution.

CONSTRAINTS:
- Only stdlib imports allowed: random, string, re.
- No file I/O, no network, no eval/exec/compile/open/__import__.
- Always return a NON-EMPTY string.
- Use the `rng` parameter for all randomness; no globals.
- Function name: `generate_<sanitized_mnemonic_lowercase>` in snake_case.
- 5-25 line body.
- Generate VARIATION: different lengths, prefixes, format variants — not 100 near-identical strings.
- Do NOT copy the source row's verbatim values.

## Name variants

A list of 15-20 plausible column names that real-world columns of this class would have in production database schemas.  These will be used to LABEL the synthetic columns in the training corpus.

CONSTRAINTS:
- snake_case identifiers (lowercase, underscore-separated, alphanumeric)
- No table prefixes (NOT `users.user_id` — just `user_id`)
- Mix semantic and opaque names: about 75% semantic (e.g., `crypto_ref`, `hash_id`, `api_key_digest`) and 25% opaque (`val_42`, `col_07`, `dat_3a`)
- Names must NOT match the source column name verbatim (look at the source's column name and AVOID it)
- Variety in length and structure — single words, two-word combos, three-word combos with abbreviations

## OUTPUT FORMAT — strict JSON, no prose outside JSON:

{
  "function_name": "generate_<sanitized_mnemonic>",
  "code": "def generate_<name>(rng: random.Random) -> str:\\n    ...\\n    return value\\n",
  "name_variants": ["semantic_name_1", "another_semantic", "col_42", "..."],
  "rationale": "1-2 sentences explaining the shape your generator targets"
}

The "code" field must be a complete, self-contained function definition string (NOT a code fence, NOT a partial snippet — the literal Python source ready to be parsed by ast.parse()).
"""


def load_confusables_map() -> dict[str, list[dict]]:
    """Load the per-target confusables map (target_code -> [{ref, key}]).

    These are rows that the PRIOR iteration's generator-trained SVM
    hijacked away from their correct ref class.  The critic uses this
    to refine output patterns away from confusable shapes.
    """
    p = Path("/tmp/confusables_map.json")
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def load_confusable_samples(refs: list[str]) -> dict[str, str]:
    """For each confusable ref code, fetch a sample embedding_text."""
    if not refs: return {}
    cls = json.loads((RUN_DIR / "classifications.json").read_text())
    out = {}
    refs_set = set(refs)
    for r in cls:
        ref = r.get("reference_code")
        if ref in refs_set and ref not in out and r.get("embedding_text"):
            out[ref] = r["embedding_text"][:200]
            if len(out) == len(refs_set):
                break
    return out


def build_user_prompt(group: dict, confusables: list[dict] | None = None,
                      confusable_samples: dict[str, str] | None = None) -> str:
    primary = group["primary_row"]
    parsed = parse_query(primary.get("embedding_text", ""))
    aux_lines = []
    for aux in (group.get("aux_rows") or [])[:3]:
        ap = parse_query(aux.get("embedding_text", ""))
        aux_lines.append(f"  - additional source: `{aux.get('key')}` values: `{ap['sample_values'][:6]}`")
    aux_block = "\n".join(aux_lines) if aux_lines else "  (none)"
    return f"""Generate a synthetic-data generator function for class `{group['reference_code']}` ({group.get('target_label') or group.get('target_mnemonic')}).

## Source failure row (primary)
- column: `{primary.get('key')}`
- reference_code: `{group['reference_code']}`
- reference label/mnemonic: {group.get('target_label')} / {group.get('target_mnemonic')}
- column_type: `{parsed['column_type']}`
- sample_values (from source, do NOT copy verbatim): `{parsed['sample_values'][:8]}`
- cardinality: {parsed['cardinality']}

## Additional shape examples
{aux_block}

## Source raw embedding text
`{primary.get('embedding_text','')[:300]}`
{_confusables_block(confusables, confusable_samples)}
Write a generator function for class `{group['reference_code']}`.  Output strict JSON per system message."""


def _confusables_block(confusables: list[dict] | None,
                       confusable_samples: dict[str, str] | None) -> str:
    """Append a 'do not overlap with' block when prior regressions are known."""
    if not confusables:
        return ""
    lines = ["", "## Confusables — your generator's PRIOR iteration hijacked these classes:",
             "Your refined output MUST visibly differ from the value-shapes and column-name token patterns below.  "
             "Adjust formats (lengths, delimiter chars, prefixes) so char-n-gram TF-IDF features distinguish your class.", ""]
    for entry in (confusables or []):
        ref = entry.get("ref")
        key = entry.get("key")
        sample = (confusable_samples or {}).get(ref, "(no sample available)")
        lines.append(f"- confusable ref `{ref}` — sample column `{key}`:")
        lines.append(f"  `{sample}`")
    lines.append("")
    lines.append("CRITICAL: avoid value shapes/lengths/delimiters and name-variant token patterns that overlap with the above. "
                 "If your previous output used dashes-and-digits patterns that collide with date/IP/timestamp formats, "
                 "switch to a clearly distinguishing format (different delimiters, different overall length, country/region prefixes, etc.).")
    return "\n".join(lines)


def call_generator_critic(group: dict, cfg, model: str,
                           confusables: list[dict] | None = None,
                           confusable_samples: dict[str, str] | None = None) -> dict:
    from enrichment_evolution import call_llm
    user = build_user_prompt(group, confusables, confusable_samples)
    return call_llm(GENERATOR_SYSTEM_PROMPT, user, cfg, model=model)


# ──────────────────────────────────────────────────────────────────────
# Phase 2: validate generator code
# ──────────────────────────────────────────────────────────────────────

ALLOWED_IMPORTS = {"random", "string", "re"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "open", "__import__"}


def validate_ast(code: str) -> tuple[bool, str]:
    """AST safety inspection."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    return False, f"forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module not in ALLOWED_IMPORTS:
                return False, f"forbidden import-from: {node.module}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
                return False, f"forbidden call: {node.func.id}"
            if isinstance(node.func, ast.Attribute) and node.func.attr in FORBIDDEN_CALLS:
                return False, f"forbidden attr call: {node.func.attr}"
    return True, "ok"


SANDBOX_RUNNER = """
import sys, json, random, importlib.util, traceback
code_path = sys.argv[1]
func_name = sys.argv[2]
n = int(sys.argv[3])
spec = importlib.util.spec_from_file_location("_gen_test", code_path)
module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(module)
except Exception as e:
    print(json.dumps({"ok": False, "reason": f"exec failed: {e}"}))
    sys.exit(0)
fn = getattr(module, func_name, None)
if not callable(fn):
    print(json.dumps({"ok": False, "reason": f"function {func_name} not found or not callable"}))
    sys.exit(0)
rng = random.Random(42)
outputs = []
for _ in range(n):
    try:
        v = fn(rng)
    except Exception as e:
        print(json.dumps({"ok": False, "reason": f"call raised: {type(e).__name__}: {e}"}))
        sys.exit(0)
    if not isinstance(v, str) or not v:
        print(json.dumps({"ok": False, "reason": f"non-string or empty output: {v!r}"}))
        sys.exit(0)
    outputs.append(v)
print(json.dumps({"ok": True, "outputs": outputs}))
"""


def sandbox_exec(code: str, func_name: str, n: int = 50) -> tuple[bool, str, list[str]]:
    """Run the generator N times in a subprocess; verify outputs."""
    tmpfile = EXPERIMENT_ROOT / "_tmp_gen.py"
    tmpfile.parent.mkdir(parents=True, exist_ok=True)
    # Prepend stdlib imports so type annotations like `random.Random` resolve.
    # Opus's prompt allows these imports; we provide them defensively.
    wrapped = "import random\nimport string\nimport re\n\n" + code
    tmpfile.write_text(wrapped)

    runner_path = EXPERIMENT_ROOT / "_tmp_runner.py"
    runner_path.write_text(SANDBOX_RUNNER)

    try:
        result = subprocess.run(
            [sys.executable, str(runner_path), str(tmpfile), func_name, str(n)],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        return False, "sandbox timeout", []
    finally:
        tmpfile.unlink(missing_ok=True)
        runner_path.unlink(missing_ok=True)

    try:
        out = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return False, f"sandbox stdout malformed: {result.stdout[:200]!r} stderr: {result.stderr[:200]!r}", []
    if not out.get("ok"):
        return False, out.get("reason", "unknown sandbox failure"), []
    return True, "ok", out.get("outputs", [])


def validate_output_shape(outputs: list[str], source_values: list[str]) -> tuple[bool, str]:
    """Compare candidate distribution to source values."""
    if not outputs:
        return False, "no outputs produced"
    # All non-empty
    if any(not v for v in outputs):
        return False, "some outputs are empty"
    # Diversity check: no more than 50% identical outputs
    counts = Counter(outputs)
    most_common_count = counts.most_common(1)[0][1]
    if most_common_count > len(outputs) // 2:
        return False, f"too repetitive ({most_common_count}/{len(outputs)} are identical)"
    # Length distribution check vs source
    if source_values:
        src_lens = [len(v) for v in source_values if v]
        out_lens = [len(v) for v in outputs]
        if src_lens:
            src_mean = sum(src_lens) / len(src_lens)
            out_mean = sum(out_lens) / len(out_lens)
            # Allow 0.3x-3x ratio (very generous to accommodate format variation)
            if src_mean > 0 and not (0.3 <= out_mean / src_mean <= 3.5):
                return False, f"length mismatch: source mean {src_mean:.1f}, output mean {out_mean:.1f}"
    return True, "ok"


# ──────────────────────────────────────────────────────────────────────
# Phase 3: persist to build/lib/generated/
# ──────────────────────────────────────────────────────────────────────

def persist_generators(accepted: list[dict]) -> Path:
    """Write build/lib/generated/{__init__.py, generators_v0.py, README.md}."""
    GENERATED_PKG.mkdir(parents=True, exist_ok=True)

    # Build the module content
    header = f'''"""Opus-generated synthetic data generators.

Auto-generated by scripts/svm_generator_experiment.py
on {datetime.now(timezone.utc).isoformat()}.

This module is auto-generated.  Do not edit by hand.
Each function generates one synthetic training value per call;
intended to be used via a GeneratorRegistry plugged into
src/atelier/classify/synth.py:generate_synth_tables().
"""
import random
import string
import re


'''
    body_parts = [header]
    code_to_fn: dict[str, str] = {}
    for entry in accepted:
        body_parts.append(entry["code"].strip() + "\n\n")
        code_to_fn[entry["reference_code"]] = entry["function_name"]

    # GENERATORS_BY_CODE dict
    body_parts.append("\nGENERATORS_BY_CODE: dict[str, callable] = {\n")
    for code, fn in code_to_fn.items():
        body_parts.append(f'    "{code}": {fn},\n')
    body_parts.append("}\n\n")

    # NAME_VARIANTS_BY_CODE dict — drives column-name selection during corpus augmentation
    body_parts.append("NAME_VARIANTS_BY_CODE: dict[str, list[str]] = {\n")
    for entry in accepted:
        names = entry.get("name_variants", [])
        body_parts.append(f'    "{entry["reference_code"]}": {json.dumps(names)},\n')
    body_parts.append("}\n\n")

    body_parts.append("__all__ = [\"GENERATORS_BY_CODE\", \"NAME_VARIANTS_BY_CODE\"] + [\n")
    for fn in code_to_fn.values():
        body_parts.append(f'    "{fn}",\n')
    body_parts.append("]\n")

    gens_path = GENERATED_PKG / "generators_v0.py"
    gens_path.write_text("".join(body_parts))

    init_path = GENERATED_PKG / "__init__.py"
    init_path.write_text(
        '"""Generated synthetic-data generators (auto-produced)."""\n'
        "from .generators_v0 import GENERATORS_BY_CODE\n"
        '__all__ = ["GENERATORS_BY_CODE"]\n'
    )

    readme_path = GENERATED_PKG / "README.md"
    readme_path.write_text(
        "# build/lib/generated/\n\n"
        "Auto-generated synthetic-data generator functions.  "
        "Do not edit by hand.  "
        "Generated by `scripts/svm_generator_experiment.py`.\n\n"
        f"Most recent generation: {datetime.now(timezone.utc).isoformat()}\n\n"
        f"Functions: {len(code_to_fn)}\n\n"
        "Codes covered:\n"
        + "\n".join(f"- `{code}`" for code in code_to_fn) + "\n"
    )
    return gens_path


# Legacy main() + offline LinearSVC training removed 2026-05-25.
# See module docstring for the current entry points.
