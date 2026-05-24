"""SVM generator-code experiment.

Opus generates Python generator FUNCTIONS (not raw data) for cosine-floor
failure rows.  Generators are persisted under build/lib/generated/ (a
gitignored, importable package), merged into a GeneratorRegistry, used
to build an augmented synth corpus, and the candidate SVM trained on
that corpus is compared head-to-head against a baseline trained on the
same framework WITHOUT the new generators.

Live artifacts NEVER touched:
  - build/data/svm_training/corpus/   (canonical synth corpus)
  - any svm.pkl under build/results/  (pipeline-run SVMs)
  - any source code under src/

Usage:
  python scripts/svm_generator_experiment.py [--max-codes 9] [--samples-per-code 100]
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


def load_llm_generators(path: Path) -> tuple[dict[str, callable], dict[str, list[str]]]:
    spec = importlib.util.spec_from_file_location("_llm_generators_v0", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GENERATORS_BY_CODE, getattr(module, "NAME_VARIANTS_BY_CODE", {})


# ──────────────────────────────────────────────────────────────────────
# Phase 4-5: build corpora + train SVMs
# ──────────────────────────────────────────────────────────────────────

def copy_live_corpus(out_dir: Path):
    """Copy the live (pipeline) corpus exactly — this is our common base."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for f in LIVE_CORPUS.glob("synth_*.csv"):
        shutil.copy2(f, out_dir / f.name)
    shutil.copy2(LIVE_CORPUS / "reference_labels.json", out_dir / "reference_labels.json")


def augment_corpus_with_generators(
    out_dir: Path,
    llm_generators: dict[str, callable],
    name_variants: dict[str, list[str]],
    *,
    columns_per_code: int = 30,
    rows_per_column: int = 100,
    seed: int = 4242,
) -> int:
    """Append new procedurally-generated columns to a corpus dir.

    For each LLM-generated function: produce ``columns_per_code`` synthetic
    columns named via the LLM-emitted name_variants (cycled and suffixed
    if more columns than variants).  Each column has ``rows_per_column``
    values from running the generator.  Append as a new CSV file
    (synth_999_proxies.csv) and update reference_labels.json.

    Returns the number of NEW columns added.
    """
    rng = random.Random(seed)
    ref_path = out_dir / "reference_labels.json"
    ref_data: dict[str, str] = json.loads(ref_path.read_text())

    col_data: dict[str, list[str]] = {}
    for code, gen_fn in llm_generators.items():
        variants = name_variants.get(code, [])
        # Fallback: if somehow no variants, use sanitized code as opaque name
        if not variants:
            sanitized = code.replace(".", "_")
            variants = [f"col_{sanitized}_{i}" for i in range(columns_per_code)]
        for k in range(columns_per_code):
            base = variants[k % len(variants)]
            # If we'd recycle a name, suffix with _vN to keep names unique
            if k < len(variants):
                col_name = base
            else:
                col_name = f"{base}_v{k // len(variants):02d}"
            # Defensive: avoid collisions with existing reference_labels
            while col_name in ref_data or col_name in col_data:
                col_name = f"{col_name}_x"
            sub_rng = random.Random(rng.randint(0, 2**31))
            values: list[str] = []
            for _ in range(rows_per_column):
                try:
                    v = gen_fn(sub_rng)
                    if not isinstance(v, str):
                        v = str(v)
                    values.append(v)
                except Exception as exc:
                    values.append(f"genfail_{exc}")
            col_data[col_name] = values
            ref_data[col_name] = code

    # Write a new CSV file: header = column names, rows = values per column
    out_csv = out_dir / "synth_999_proxies.csv"
    if col_data:
        max_rows = max(len(v) for v in col_data.values())
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(col_data.keys()))
            for i in range(max_rows):
                w.writerow([col_data[name][i] if i < len(col_data[name]) else ""
                            for name in col_data.keys()])

    # Update reference_labels.json
    ref_path.write_text(json.dumps(ref_data, indent=2))
    return len(col_data)


def build_category_set_from_full_vocab():
    """Build a HierarchicalCategorySet from the FULL pipeline vocabulary.

    The synth corpus only emits leaf-coded columns, so its
    reference_labels.json covers a subset of the true vocab — internal
    nodes (umbrellas) never appear as labels.  NHSVM requires the full
    hierarchy for Kronecker expansion to operate over every node, and
    every node is a first-class tagging target (see
    feedback_dynamic_annotations.md, feedback_hierarchical_svm_only.md).
    Source the vocab from the live pipeline's svm.classes.json, not the
    synth corpus's labels.
    """
    from atelier.classify.taxonomy import _build_category_set_from_records
    classes_path = RUN_DIR / "svm.classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(f"Cannot find pipeline vocab at {classes_path}")
    payload = json.loads(classes_path.read_text())
    # Format may be either a plain list or a dict with "classes" key
    if isinstance(payload, dict):
        codes = payload.get("classes") or payload.get("codes") or []
    else:
        codes = payload
    codes = sorted({str(c) for c in codes if c})
    records = [{"id": c, "ontology": c, "annotation": c} for c in codes]
    return _build_category_set_from_records(records, hierarchical=True), codes


def train_svm_offline(corpus_dir: Path, output_path: Path, category_set,
                       *, svd_components: int = 1000):
    """Train SVM with a custom SVD-components budget.

    Bypasses train_svm so we can pass a non-default SVMConfig.  Default
    is 1000 components (~75-80% variance) vs the library default of 200
    (~49% variance).  Trades model size for fidelity.
    """
    from atelier.classify.ml_train import _load_synth_data, _infer_column_type
    from atelier.classify.svm_classifier import SVMClassifier, SVMConfig, build_svm_text
    columns, reference_labels = _load_synth_data(corpus_dir)
    texts: list[str] = []
    labels: list[str] = []
    for col_name, values in columns.items():
        code = reference_labels.get(col_name)
        if not code: continue
        col_type = _infer_column_type(values[:10])
        text = build_svm_text(col_name, column_type=col_type, sample_values=values[:5])
        texts.append(text)
        labels.append(code)
    cfg = SVMConfig(nhsvm_svd_components=svd_components)
    classifier = SVMClassifier(config=cfg, category_set=category_set, hierarchical=True)
    t0 = time.time()
    classifier.fit(texts, labels)
    classifier.save(output_path)
    elapsed = time.time() - t0
    print(f"  trained {output_path.name} in {elapsed:.1f}s (SVD={svd_components})")
    return output_path


# ──────────────────────────────────────────────────────────────────────
# Phase 6: evaluate
# ──────────────────────────────────────────────────────────────────────

def is_subtree(code, target):
    if not code or not target: return False
    return code == target or code.startswith(target + ".")


def svm_top1(svm, text):
    try:
        proba = svm.predict_proba_single(text)
        if not proba:
            return None
        return max(proba.items(), key=lambda kv: kv[1])[0]
    except Exception:
        return None


def evaluate(baseline, candidate, candidates: list[dict], regression_sample: list[dict]):
    from atelier.classify.svm_classifier import build_svm_text

    # Target rescue: each candidate group's primary_row + aux_rows
    target_results = []
    seen_keys = set()
    for group in candidates:
        rows = [group["primary_row"]] + (group.get("aux_rows") or [])
        for r in rows:
            key = r.get("key")
            if key in seen_keys:
                continue
            seen_keys.add(key)
            parsed = parse_query(r.get("embedding_text", ""))
            text = build_svm_text(
                parsed.get("column_name_humanized") or key.split(".", 1)[-1],
                parsed.get("column_type"),
                parsed.get("sample_values", [])[:5],
            )
            bl = svm_top1(baseline, text)
            cd = svm_top1(candidate, text)
            ref = r.get("reference_code")
            target_results.append({
                "key": key,
                "ref": ref,
                "baseline_top1": bl,
                "candidate_top1": cd,
                "baseline_correct": is_subtree(bl, ref),
                "candidate_correct": is_subtree(cd, ref),
            })

    # Regression check on held-out non-floor rows where baseline is correct
    regression_results = []
    for r in regression_sample:
        parsed = parse_query(r.get("embedding_text", ""))
        text = build_svm_text(
            parsed.get("column_name_humanized") or r.get("column_name", ""),
            parsed.get("column_type"),
            parsed.get("sample_values", [])[:5],
        )
        bl = svm_top1(baseline, text)
        cd = svm_top1(candidate, text)
        ref = r.get("reference_code")
        regression_results.append({
            "key": f"{r.get('table_name')}.{r.get('column_name')}",
            "ref": ref,
            "baseline_top1": bl,
            "candidate_top1": cd,
            "baseline_correct": is_subtree(bl, ref),
            "candidate_correct": is_subtree(cd, ref),
        })
    return target_results, regression_results


def write_report(target, regression, accepted, rejected, candidates):
    base_correct = sum(1 for r in target if r["baseline_correct"])
    cand_correct = sum(1 for r in target if r["candidate_correct"])
    reg_pre = sum(1 for r in regression if r["baseline_correct"])
    reg_post = sum(1 for r in regression if r["candidate_correct"])
    broken = sum(1 for r in regression if r["baseline_correct"] and not r["candidate_correct"])
    rescued_post_only = sum(1 for r in target if r["candidate_correct"] and not r["baseline_correct"])

    rescue_lift = cand_correct - base_correct
    regression_pct = (broken / len(regression) * 100) if regression else 0

    if rescue_lift >= 1 and broken <= max(3, len(regression) // 10):
        verdict = "VALIDATED — mechanism plumbed and proves out; iterate from here"
    elif rescue_lift >= 1:
        verdict = "PARTIAL — lift achieved but regression budget exceeded"
    elif rescue_lift == 0 and broken == 0:
        verdict = "INERT — no lift, no regression; generators didn't change behavior"
    else:
        verdict = "WRONG — generators hurt performance"

    lines = ["# SVM generator-code experiment — metrics report", ""]
    lines.append(f"_Generated {datetime.now(timezone.utc).isoformat()}._")
    lines.append("")
    lines.append("## Verdict")
    lines.append(f"**{verdict}**")
    lines.append("")
    lines.append("## Target rescue (CAPABILITY_FLOOR rows)")
    lines.append(f"- baseline correct: **{base_correct} / {len(target)}**")
    lines.append(f"- candidate correct: **{cand_correct} / {len(target)}**")
    lines.append(f"- net lift: **{rescue_lift:+d}**")
    lines.append(f"- rescued by candidate only: **{rescued_post_only}**")
    lines.append("")
    lines.append("## Regression check (rows baseline gets right)")
    lines.append(f"- sample size: {len(regression)}")
    lines.append(f"- baseline correct: {reg_pre}/{len(regression)}")
    lines.append(f"- candidate correct: {reg_post}/{len(regression)}")
    lines.append(f"- regressions (was correct, now wrong): **{broken} ({regression_pct:.1f}%)**")
    lines.append("")
    lines.append("## Generator generation summary")
    lines.append(f"- candidates considered: {len(candidates)}")
    lines.append(f"- generators accepted: {len(accepted)}")
    lines.append(f"- generators rejected: {len(rejected)}")
    if rejected:
        reasons = Counter(r["reason"][:50] for r in rejected)
        lines.append("- rejection reasons:")
        for reason, n in reasons.most_common():
            lines.append(f"  - `{reason}`: {n}")
    lines.append("")
    lines.append("## Per-row target results")
    lines.append("")
    lines.append("| row | ref | baseline | candidate | rescued? |")
    lines.append("|---|---|---|---|---|")
    for r in target:
        if r["candidate_correct"] and not r["baseline_correct"]:
            mark = "✓"
        elif r["candidate_correct"] and r["baseline_correct"]:
            mark = "—"
        elif r["baseline_correct"] and not r["candidate_correct"]:
            mark = "✗ regressed"
        else:
            mark = "✗"
        lines.append(f"| `{r['key']}` | `{r['ref']}` | `{r['baseline_top1']}` | `{r['candidate_top1']}` | {mark} |")
    lines.append("")
    if broken > 0:
        lines.append("## Regressed rows")
        lines.append("")
        lines.append("| row | ref | baseline | candidate |")
        lines.append("|---|---|---|---|")
        for r in regression:
            if r["baseline_correct"] and not r["candidate_correct"]:
                lines.append(f"| `{r['key']}` | `{r['ref']}` | `{r['baseline_top1']}` | `{r['candidate_top1']}` |")
    (EXPERIMENT_ROOT / "metrics.md").write_text("\n".join(lines))
    return verdict, rescue_lift, broken


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-codes", type=int, default=9)
    ap.add_argument("--samples-per-code", type=int, default=50,
                    help="number of generator calls in sandbox validation")
    ap.add_argument("--svd-components", type=int, default=1000,
                    help="NHSVM SVD components (default 1000 vs lib default 200)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_ROOT / "proposals").mkdir(exist_ok=True)

    # Hash live corpus for integrity check
    live_hash = hashlib.sha256()
    for f in sorted(LIVE_CORPUS.glob("synth_*.csv")):
        live_hash.update(f.read_bytes())
    live_ref_before = (LIVE_CORPUS / "reference_labels.json").read_bytes()
    canonical_before = live_hash.hexdigest()

    print("=== Phase 0: candidate selection ===")
    candidates = load_candidates(max_codes=args.max_codes)
    print(f"  {len(candidates)} candidate codes (grouped from CAPABILITY_FLOOR rows)")
    for c in candidates:
        print(f"    {c['reference_code']:14s} ({c.get('target_mnemonic','?')}) primary={c['primary_row'].get('key')}")
    if not candidates:
        print("  No candidates.  Aborting.")
        return 1

    print("\n=== Phase 1: generate generator code via Bedrock Opus ===")
    from atelier.config import load_config
    from enrichment_evolution import resolve_model
    cfg = load_config()
    model = resolve_model(cfg, None)
    print(f"  LLM model: {model}")

    # Load confusables map (from prior iteration's regressions) — used to refine
    # critic prompts so generators avoid overlapping with previously-hijacked refs.
    confusables_map = load_confusables_map()
    if confusables_map:
        all_refs = sorted({c["ref"] for entries in confusables_map.values() for c in entries})
        confusable_samples = load_confusable_samples(all_refs)
        print(f"  loaded confusables map: {len(confusables_map)} targets w/ confusion history; "
              f"{len(all_refs)} distinct confusable refs")
    else:
        confusable_samples = {}

    accepted: list[dict] = []
    rejected: list[dict] = []
    for i, group in enumerate(candidates, 1):
        code = group["reference_code"]
        # Look up confusables FOR THIS TARGET CODE
        target_confusables = confusables_map.get(code, [])
        if target_confusables:
            print(f"  [{i}/{len(candidates)}] code={code} -> generating function "
                  f"(with {len(target_confusables)} prior confusables)")
        else:
            print(f"  [{i}/{len(candidates)}] code={code} -> generating function")
        try:
            resp = call_generator_critic(group, cfg, model,
                                         confusables=target_confusables,
                                         confusable_samples=confusable_samples)
        except Exception as exc:
            print(f"    LLM call failed: {exc}")
            rejected.append({"code": code, "reason": f"llm-call-failed: {exc}"})
            continue
        # Save raw proposal
        prop_path = EXPERIMENT_ROOT / "proposals" / f"{code.replace('.', '_')}.json"
        prop_path.write_text(json.dumps(resp, indent=2))

        func_name = resp.get("function_name", "")
        code_src = resp.get("code", "")
        if not func_name or not code_src:
            rejected.append({"code": code, "reason": "missing function_name or code"})
            continue
        if not re.match(r"^generate_[a-z][a-z0-9_]*$", func_name):
            rejected.append({"code": code, "reason": f"bad function_name: {func_name}"})
            continue

        # Phase 2 validation
        ok, reason = validate_ast(code_src)
        if not ok:
            rejected.append({"code": code, "reason": f"ast: {reason}"})
            continue
        ok, reason, outputs = sandbox_exec(code_src, func_name, n=args.samples_per_code)
        if not ok:
            rejected.append({"code": code, "reason": f"sandbox: {reason}"})
            continue
        parsed = parse_query(group["primary_row"].get("embedding_text", ""))
        ok, reason = validate_output_shape(outputs, parsed.get("sample_values", []))
        if not ok:
            rejected.append({"code": code, "reason": f"shape: {reason}"})
            continue

        # Validate name_variants
        name_variants_raw = resp.get("name_variants", [])
        if not isinstance(name_variants_raw, list) or len(name_variants_raw) < 5:
            rejected.append({"code": code, "reason": f"too few name_variants ({len(name_variants_raw) if isinstance(name_variants_raw, list) else 'not-list'})"})
            continue
        name_variants: list[str] = []
        source_col = group["primary_row"].get("key", "").split(".", 1)[-1].lower()
        for nv in name_variants_raw:
            if not isinstance(nv, str): continue
            nv = nv.strip().lower()
            if not re.match(r"^[a-z][a-z0-9_]*$", nv): continue
            if nv == source_col: continue  # never copy source verbatim
            if nv in name_variants: continue  # dedupe
            name_variants.append(nv)
        if len(name_variants) < 5:
            rejected.append({"code": code, "reason": f"name_variants: only {len(name_variants)} passed validation"})
            continue

        accepted.append({
            "reference_code": code,
            "function_name": func_name,
            "code": code_src,
            "name_variants": name_variants[:25],
            "rationale": resp.get("rationale", ""),
            "sample_outputs": outputs[:5],
        })
        print(f"    ✓ accepted (vals: {outputs[:2]}; names: {name_variants[:3]})")

    (EXPERIMENT_ROOT / "validation_log.json").write_text(json.dumps({
        "accepted": accepted, "rejected": rejected,
    }, indent=2))
    print(f"\n  accepted: {len(accepted)}, rejected: {len(rejected)}")

    if not accepted:
        print("  No generators accepted; aborting before corpus build.")
        return 1

    print("\n=== Phase 3: persist to build/lib/generated/ ===")
    gens_path = persist_generators(accepted)
    print(f"  wrote {gens_path}")
    llm_gens, name_variants = load_llm_generators(gens_path)
    print(f"  loaded {len(llm_gens)} generators + {len(name_variants)} name-variant lists from disk")

    print("\n=== Phase 4: build corpora (baseline=live copy, candidate=live + generators) ===")
    baseline_corpus = EXPERIMENT_ROOT / "baseline_corpus"
    candidate_corpus = EXPERIMENT_ROOT / "candidate_corpus"

    # Baseline = exact copy of the live (pipeline) corpus
    print(f"  copying live corpus to baseline_corpus...")
    copy_live_corpus(baseline_corpus)
    # Candidate = copy + generator-produced columns
    print(f"  copying live corpus to candidate_corpus + augmenting with generators...")
    copy_live_corpus(candidate_corpus)
    n_added = augment_corpus_with_generators(candidate_corpus, llm_gens, name_variants,
                                              columns_per_code=30, rows_per_column=100)
    print(f"  candidate corpus: added {n_added} new columns across {len(llm_gens)} target codes")

    # Build category_set from the FULL pipeline vocabulary (296 dotted-decimal
    # codes), not just from the synth corpus's 219.  NHSVM needs the full
    # hierarchy for proper Kronecker expansion.
    cat_set, all_codes = build_category_set_from_full_vocab()
    n_cats = len(getattr(cat_set, "all_categories", cat_set.categories))
    print(f"  category_set built from pipeline vocab: {n_cats} categories ({len(all_codes)} codes)")
    # Sanity check: are our 9 target codes in the full vocab?
    target_codes = {g['reference_code'] for g in candidates}
    in_vocab = target_codes & set(all_codes)
    missing = target_codes - set(all_codes)
    print(f"  target codes in vocab: {len(in_vocab)}/{len(target_codes)}")
    if missing:
        print(f"    NOT in vocab: {sorted(missing)}")

    print("\n=== Phase 5: train SVMs (NHSVM, SVD=1000 for higher fidelity) ===")
    baseline_svm_path = EXPERIMENT_ROOT / "baseline_svm"
    candidate_svm_path = EXPERIMENT_ROOT / "candidate_svm"
    train_svm_offline(baseline_corpus, baseline_svm_path, cat_set, svd_components=args.svd_components)
    train_svm_offline(candidate_corpus, candidate_svm_path, cat_set, svd_components=args.svd_components)

    print("\n=== Phase 6: evaluate ===")
    from atelier.classify.svm_classifier import SVMClassifier
    baseline = SVMClassifier.load(baseline_svm_path)
    candidate = SVMClassifier.load(candidate_svm_path)

    # Regression sample: rows from classifications.json where reference is in
    # at least one of our target codes' subtrees OR a peer subtree, and where
    # the baseline SVM gets it right.  Stratify across diverse reference codes.
    cls = json.loads((RUN_DIR / "classifications.json").read_text())
    seen_refs = set()
    regression_sample = []
    target_codes = {c["reference_code"] for c in candidates}
    # Skip rows in our target candidate set
    target_keys = set()
    for c in candidates:
        target_keys.add(c["primary_row"].get("key"))
        for ax in (c.get("aux_rows") or []):
            target_keys.add(ax.get("key"))
    for r in cls:
        key = f"{r.get('table_name')}.{r.get('column_name')}"
        if key in target_keys:
            continue
        ref = r.get("reference_code")
        if not ref or not r.get("embedding_text"):
            continue
        if ref in seen_refs:
            continue
        # We want rows where baseline_svm is correct (apples-to-apples regression)
        parsed = parse_query(r.get("embedding_text", ""))
        from atelier.classify.svm_classifier import build_svm_text
        text = build_svm_text(
            parsed.get("column_name_humanized") or r.get("column_name", ""),
            parsed.get("column_type"),
            parsed.get("sample_values", [])[:5],
        )
        bl_top1 = svm_top1(baseline, text)
        if is_subtree(bl_top1, ref):
            regression_sample.append(r)
            seen_refs.add(ref)
            if len(regression_sample) >= 30:
                break
    print(f"  regression sample: {len(regression_sample)} rows (baseline-correct)")

    target_results, regression_results = evaluate(baseline, candidate, candidates, regression_sample)

    print("\n=== Phase 7: report ===")
    verdict, lift, broken = write_report(target_results, regression_results, accepted, rejected, candidates)

    # Integrity check
    live_hash_after = hashlib.sha256()
    for f in sorted(LIVE_CORPUS.glob("synth_*.csv")):
        live_hash_after.update(f.read_bytes())
    live_ref_after = (LIVE_CORPUS / "reference_labels.json").read_bytes()
    canonical_after = live_hash_after.hexdigest()
    print()
    print(f"  canonical corpus integrity: {canonical_after == canonical_before}  ({canonical_before[:12]} → {canonical_after[:12]})")
    print(f"  canonical ref_labels integrity: {live_ref_after == live_ref_before}")
    print()
    print(f"=== {verdict} ===")
    print(f"  rescue lift: {lift:+d}")
    print(f"  regressions: {broken}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
