#!/usr/bin/env python3
"""scripts/generate_corpus_v2.py — Phase C of the corpus expansion plan.

Produces ~200 synthetic columns per taxonomy node (~57k total rows
across 287 nodes) for training the factorized NHSVM head.  Per column:

  - 5 sample values from a generator (v1 → hand-coded ICE → template
    → inferred, in priority order)
  - A column name from per-code NAME_VARIANTS_BY_CODE (v1) merged
    with the 8-style rotation from generate_synth_source.py
  - 4 sibling column names from per-code TABLE_CONTEXT_BY_CODE (v1,
    70% of the time) or cross-domain mixing (30%)
  - A synthetic table name from the same template (or
    cross-domain default)
  - A column type inferred from a 10-sample probe of the generator

Output: ``build/data/svm_training/corpus_v2/synth_rows.jsonl`` — one
JSON record per row matching the ``Row`` dataclass shape that
``reflect_nhsvm.load_rows`` produces.  Phase D loads these directly.

Resume-safe: if the output exists and `manifest.json` matches the
expected per-code targets, skips work for codes already at target.

Usage:
  python scripts/generate_corpus_v2.py
  python scripts/generate_corpus_v2.py --rows-per-node 200
  python scripts/generate_corpus_v2.py --rows-per-node 100  # smaller smoke run
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
import re
import string
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from reflect_nhsvm import build_category_set
from atelier.classify.synth_registry import GeneratorRegistry
from atelier.classify.enrichment_loader import load_enrichment_payloads

log = logging.getLogger("generate_corpus_v2")

OUTPUT_DIR = Path("build/data/svm_training/corpus_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SYNTH_ROWS = OUTPUT_DIR / "synth_rows.jsonl"
MANIFEST = OUTPUT_DIR / "manifest.json"
DEFAULT_PAYLOADS = Path("build/data/svm_training/enrichment_payloads.json")
GENERATORS_V1 = Path("build/lib/generated/generators_v1.py")

# Cross-domain fallback table names and sibling pools — used when a
# code lacks per-code TABLE_CONTEXT_BY_CODE entries.
FALLBACK_TABLE_NAMES = [
    "users", "customers", "accounts", "transactions", "events",
    "orders", "products", "audit_log", "sessions", "subscriptions",
    "device_registry", "communication_log", "payment_history",
    "permission_grants", "system_metadata", "incident_reports",
]
FALLBACK_SIBLINGS = [
    "user_id", "created_at", "updated_at", "status", "type",
    "amount", "currency", "country_code", "ip_address", "session_id",
    "first_name", "last_name", "email", "phone", "address",
    "device_id", "transaction_id", "reference_id", "name", "description",
    "is_active", "version", "metadata", "source", "channel",
]


# ──────────────────────────────────────────────────────────────────────
# Load generators_v1 if present
# ──────────────────────────────────────────────────────────────────────

def load_generators_v1() -> dict:
    """Load v1 generators + name + table-context dicts.

    Returns {} if generators_v1.py doesn't exist; otherwise returns
    {generators_by_code, name_variants_by_code, table_context_by_code}.
    """
    if not GENERATORS_V1.exists():
        log.info("  generators_v1.py not present — will use only ICE + template + inferred")
        return {
            "generators_by_code": {},
            "name_variants_by_code": {},
            "table_context_by_code": {},
        }
    spec = importlib.util.spec_from_file_location("generators_v1", GENERATORS_V1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    n_gens = len(getattr(mod, "GENERATORS_BY_CODE", {}))
    n_names = len(getattr(mod, "NAME_VARIANTS_BY_CODE", {}))
    n_tables = len(getattr(mod, "TABLE_CONTEXT_BY_CODE", {}))
    log.info("  loaded generators_v1.py: %d codes have generators, %d have name variants, %d have table contexts",
             n_gens, n_names, n_tables)
    return {
        "generators_by_code": dict(getattr(mod, "GENERATORS_BY_CODE", {})),
        "name_variants_by_code": dict(getattr(mod, "NAME_VARIANTS_BY_CODE", {})),
        "table_context_by_code": dict(getattr(mod, "TABLE_CONTEXT_BY_CODE", {})),
    }


# ──────────────────────────────────────────────────────────────────────
# Type inference (matches what svm_generator_experiment does at line 554)
# ──────────────────────────────────────────────────────────────────────

def infer_column_type(samples: list[str]) -> str:
    """Infer a plausible SQL type from generator output samples."""
    if not samples:
        return "string"
    numeric_count = 0
    date_like = 0
    bool_like = 0
    max_len = 0
    for v in samples:
        v_str = str(v).strip()
        max_len = max(max_len, len(v_str))
        # Numeric check
        try:
            float(v_str.replace(",", ""))
            numeric_count += 1
        except (ValueError, OverflowError):
            pass
        # Date check (loose)
        if re.match(r"^\d{4}-\d{2}-\d{2}", v_str):
            date_like += 1
        # Boolean check
        if v_str.lower() in {"true", "false", "yes", "no", "1", "0"}:
            bool_like += 1
    n = len(samples)
    if numeric_count >= n * 0.8:
        # Distinguish int vs float
        has_decimal = any("." in str(v) for v in samples)
        return "double" if has_decimal else "bigint"
    if date_like >= n * 0.8:
        return "timestamp"
    if bool_like >= n * 0.8:
        return "boolean"
    if max_len > 200:
        return "text"
    return "string"


# ──────────────────────────────────────────────────────────────────────
# Column name generation — merges v1 names with style rotations
# ──────────────────────────────────────────────────────────────────────

def _snake_case(s: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", s.lower()).strip("_")


def _camel_case(snake: str) -> str:
    parts = snake.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


OPAQUE_PREFIXES = ["val_", "col_", "dat_", "fld_", "attr_", "f", "c", "v"]


def synth_column_name(
    cat,
    rng: random.Random,
    *,
    v1_names: list[str] | None = None,
    variant_idx: int = 0,
    opaque: bool = False,
) -> str:
    """Produce a per-column name.

    Pool priority:
      1. v1_names from NAME_VARIANTS_BY_CODE[code]  (Opus-authored)
      2. 8-style rotation from label/abbrev/common_names
      3. Opaque (val_42, col_07) as ~25% of names per SHAP-priority spec
    """
    if opaque:
        prefix = rng.choice(OPAQUE_PREFIXES)
        style = rng.randint(0, 3)
        if style == 0:
            return f"{prefix}{rng.randint(1, 999)}"
        elif style == 1:
            return f"{prefix}{''.join(rng.choices(string.ascii_lowercase, k=rng.randint(2, 4)))}"
        elif style == 2:
            return f"{prefix}{rng.choice(string.ascii_lowercase)}{rng.randint(1, 99)}"
        else:
            return f"{''.join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 6)))}{rng.randint(1, 99)}"

    # 60% chance to draw from v1_names pool when available
    if v1_names and rng.random() < 0.6:
        return rng.choice(v1_names)

    # Otherwise style rotation from label/abbrev
    label = getattr(cat, "label", "") or ""
    abbrev = getattr(cat, "abbrev", "") or ""
    # If label is just a code-like string (digits/dots), prefer abbrev —
    # this happens with the Hive-format taxonomy where label == code.
    if not label or re.match(r"^[\d.]+$", label):
        label = abbrev
    base = _snake_case(label) or _snake_case(abbrev) or f"col_{cat.code.replace('.', '_')}"

    style = variant_idx % 8
    if style == 0:
        return base
    elif style == 1:
        return _camel_case(base)
    elif style == 2:
        return base.upper()
    elif style == 3 and abbrev:
        return abbrev.strip().lower()
    elif style == 4:
        prefix = rng.choice(["user_", "customer_", "src_", "raw_", "dim_", "fact_"])
        return prefix + base
    elif style == 5:
        suffix = rng.choice(["_val", "_field", "_data", "_info", "_raw", "_v2"])
        return base + suffix
    elif style == 6:
        # Two-word combo
        prefix = rng.choice(["primary", "secondary", "alt", "backup", "legacy", "current"])
        return f"{prefix}_{base}"
    else:
        # Numeric discriminator
        return f"{base}_{variant_idx % 10 + 1}" if variant_idx else base


# ──────────────────────────────────────────────────────────────────────
# Table-context selection
# ──────────────────────────────────────────────────────────────────────

def pick_table_context(
    rng: random.Random,
    *,
    code_templates: list[dict] | None,
    fallback_table_names: list[str],
    fallback_siblings: list[str],
) -> tuple[str, list[str]]:
    """Return (table_name, sibling_columns).

    70% from per-code templates (v1) when available, 30% cross-domain.
    """
    if code_templates and rng.random() < 0.7:
        template = rng.choice(code_templates)
        table_names = template.get("table_names", []) or fallback_table_names
        siblings_pool = template.get("siblings", []) or fallback_siblings
        table_name = rng.choice(table_names)
        # Pick 4 siblings from the pool
        if len(siblings_pool) >= 4:
            siblings = rng.sample(siblings_pool, 4)
        else:
            siblings = list(siblings_pool)
            while len(siblings) < 4:
                siblings.append(rng.choice(fallback_siblings))
        return table_name, siblings

    table_name = rng.choice(fallback_table_names)
    siblings = rng.sample(fallback_siblings, 4)
    return table_name, siblings


# ──────────────────────────────────────────────────────────────────────
# Main generation loop
# ──────────────────────────────────────────────────────────────────────

def generate_corpus(
    *,
    rows_per_node: int,
    payloads_path: Path,
    seed: int = 42,
) -> dict:
    """Generate ~rows_per_node columns per taxonomy node."""
    log.info("=== Phase C: corpus generation at scale ===")
    log.info("Loading category set...")
    cat_set = build_category_set()
    cats_by_code = {c.code: c for c in cat_set.all_categories}
    log.info("  %d nodes total", len(cats_by_code))

    log.info("Loading enrichment payloads from %s...", payloads_path)
    payloads = load_enrichment_payloads(json_path=payloads_path)

    log.info("Building registry (hand-coded ICE + template + inferred)...")
    base_registry = GeneratorRegistry.from_enrichment_payloads(payloads, cat_set)

    log.info("Loading generators_v1 (Phase B output)...")
    v1 = load_generators_v1()
    # Build a v1 callable lookup: code → list[callable]
    # GENERATORS_BY_CODE in v1 is dict[code, list[callable]] (Phase B output)
    v1_gens_by_code: dict[str, list] = {}
    for code, gen_list in v1["generators_by_code"].items():
        if not isinstance(gen_list, list):
            gen_list = [gen_list]  # backward compat: single callable
        v1_gens_by_code[code] = [g for g in gen_list if callable(g)]
    log.info("  v1 generators available for %d codes", len(v1_gens_by_code))

    # Coverage check: per-code, do we have ANY generator available?
    rng = random.Random(seed)
    rows: list[dict] = []
    per_code_counts: dict[str, int] = {}
    skipped_no_gen: list[str] = []
    t0 = time.time()
    for cat_idx, code in enumerate(sorted(cats_by_code.keys())):
        cat = cats_by_code[code]
        mnemonic = getattr(cat, "abbrev", "") or ""

        # Resolve generators: v1 first, then base registry
        candidate_gens: list = []
        if code in v1_gens_by_code and v1_gens_by_code[code]:
            candidate_gens = v1_gens_by_code[code]
            gen_source = "v1"
        else:
            spec = base_registry.get(code)
            if spec is not None:
                candidate_gens = [spec.generator]
                gen_source = spec.source
            else:
                skipped_no_gen.append(code)
                continue

        # Resolve name pool: v1 first, then fall back to style rotation
        v1_names = v1["name_variants_by_code"].get(code)

        # Resolve table-context templates: v1 first, then fallback
        v1_table_ctx = v1["table_context_by_code"].get(code)

        # Generate rows_per_node columns for this code
        for col_i in range(rows_per_node):
            gen = rng.choice(candidate_gens) if len(candidate_gens) > 1 else candidate_gens[0]

            # 5 sample values from the generator
            sample_values: list[str] = []
            for _ in range(5):
                try:
                    v = gen(rng)
                    sample_values.append(str(v))
                except Exception:  # noqa: BLE001
                    sample_values.append("")
            # Skip if generator produced all-empty (shouldn't happen post-validation)
            if not any(s for s in sample_values):
                continue

            # Column type — probe 10 samples + 20% adjacent-type rows for diversity
            probe_samples = sample_values + [
                str(gen(rng)) if (s_i := True) else "" for _ in range(5)
            ]
            col_type = infer_column_type(probe_samples)
            if rng.random() < 0.2:
                # Adjacent type for diversity
                adjacent_map = {
                    "string": ["varchar", "text", "char"],
                    "varchar": ["string", "text"],
                    "text": ["string", "varchar"],
                    "bigint": ["int", "integer", "long"],
                    "double": ["float", "decimal", "numeric"],
                    "boolean": ["bool", "bit"],
                    "timestamp": ["datetime", "date"],
                }
                col_type = rng.choice(adjacent_map.get(col_type, [col_type]))

            # Column name (75% semantic / 25% opaque)
            opaque = rng.random() < 0.25
            col_name = synth_column_name(
                cat, rng, v1_names=v1_names,
                variant_idx=col_i, opaque=opaque,
            )

            # Table + siblings
            table_name, siblings = pick_table_context(
                rng, code_templates=v1_table_ctx,
                fallback_table_names=FALLBACK_TABLE_NAMES,
                fallback_siblings=FALLBACK_SIBLINGS,
            )
            siblings_full = [col_name] + siblings  # match Row.siblings_full shape

            rows.append({
                "table": table_name,
                "column": col_name,
                "column_type": col_type,
                "sample_values": sample_values,
                "siblings_full": siblings_full,
                "mnemonic": mnemonic,
                "code": code,
                "gen_source": gen_source,
            })
            per_code_counts[code] = per_code_counts.get(code, 0) + 1

        if (cat_idx + 1) % 25 == 0:
            elapsed = time.time() - t0
            log.info("  %d/%d nodes done  %d rows  %.1fs",
                     cat_idx + 1, len(cats_by_code), len(rows), elapsed)

    elapsed = time.time() - t0
    log.info("Generated %d total rows across %d nodes in %.1fs (%d skipped for no generator)",
             len(rows), len(per_code_counts), elapsed, len(skipped_no_gen))

    # Persist
    with SYNTH_ROWS.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    manifest = {
        "rows_per_node_target": rows_per_node,
        "n_rows_total": len(rows),
        "n_nodes_covered": len(per_code_counts),
        "n_nodes_skipped": len(skipped_no_gen),
        "skipped_codes": sorted(skipped_no_gen),
        "per_code_counts": per_code_counts,
        "elapsed_sec": round(elapsed, 1),
        "seed": seed,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    log.info("Wrote %s and %s", SYNTH_ROWS, MANIFEST)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows-per-node", type=int, default=200,
                    help="Target synthetic columns per taxonomy node (default 200)")
    ap.add_argument("--payloads", type=Path, default=DEFAULT_PAYLOADS,
                    help="Enrichment payloads JSON path")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for reproducibility")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    manifest = generate_corpus(
        rows_per_node=args.rows_per_node,
        payloads_path=args.payloads,
        seed=args.seed,
    )
    print()
    print("=== Corpus generation complete ===")
    print(f"  total rows: {manifest['n_rows_total']}")
    print(f"  nodes covered: {manifest['n_nodes_covered']}")
    print(f"  nodes skipped (no generator): {manifest['n_nodes_skipped']}")
    print(f"  output: {SYNTH_ROWS}")
    print(f"  manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
