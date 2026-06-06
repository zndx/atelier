"""Synthetic data generation for CatBoost/SVM classifier training.

Generates synthetic columns with known reference labels for every
taggable node (leaf and internal) in the controlled vocabulary.  Each
category gets both semantic column names (human-readable variants) and
opaque names (coded/random) to force classifiers to learn from VALUE
PATTERNS, not just names.

Value generators are sourced from synth_generators.py (shared with
generate_sample_source.py). The GeneratorRegistry from synth_registry.py
provides extensible coverage tracking and vocabulary-driven generation.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import re
import string
from collections.abc import Callable
from pathlib import Path
from typing import Any

from atelier.classify.synth_generators import GENERATORS

logger = logging.getLogger(__name__)

# ── Name generation infrastructure ──────────────────────────────

SYNONYMS: dict[str, list[str]] = {
    "number": ["num", "no", "nbr"],
    "address": ["addr", "adr"],
    "phone": ["tel", "ph", "phn"],
    "telephone": ["tel", "phone", "ph"],
    "identifier": ["id", "ident"],
    "account": ["acct", "acc"],
    "transaction": ["txn", "trans", "trx"],
    "payment": ["pay", "pmt"],
    "card": ["crd", "cd"],
    "date": ["dt", "dte"],
    "email": ["mail", "eml"],
    "name": ["nm"],
    "security": ["sec"],
    "social": ["soc"],
    "device": ["dev"],
    "code": ["cd"],
    "value": ["val"],
    "status": ["stat", "sts"],
    "record": ["rec"],
    "amount": ["amt"],
    "bank": ["bnk"],
    "mailing": ["mail", "postal"],
    "birth": ["bday", "bth"],
    "credit": ["cr", "cred"],
    "ip": ["ipaddr"],
    "url": ["uri", "link"],
    "timestamp": ["ts", "tstamp"],
}

_OPAQUE_PREFIXES = [
    "field_", "col_", "meta_", "v_", "x_", "dim_", "f_",
    "attr_", "var_", "c_", "d_", "val_", "p_",
]


def _snake_case(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower().replace(" ", "_")


def _camel_case(s: str) -> str:
    parts = s.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _generate_semantic_names(
    label: str,
    abbrev: str,
    common_names: str,
    rng: random.Random,
    count: int = 15,
) -> list[str]:
    """Generate diverse semantic column name variants for a category."""
    names: list[str] = []
    seen: set[str] = set()

    def _add(name: str) -> None:
        n = name.strip().strip("_")
        if n and n not in seen and len(n) > 1:
            seen.add(n)
            names.append(n)

    base = _snake_case(label)
    _add(base)
    _add(_camel_case(base))
    _add(base.upper())

    if abbrev:
        a = abbrev.strip().lower()
        _add(a)
        _add(a.upper())

    if common_names:
        for cn in re.split(r"[,|]", common_names):
            cn = cn.strip()
            if cn:
                sn = _snake_case(cn)
                _add(sn)
                _add(_camel_case(sn))

    # Word-level synonym expansion
    words = base.split("_")
    for i, word in enumerate(words):
        if word in SYNONYMS:
            for syn in SYNONYMS[word]:
                variant = "_".join(words[:i] + [syn] + words[i + 1:])
                _add(variant)

    # Prefixed variants
    for prefix in ["user_", "customer_", "src_", "raw_"]:
        _add(prefix + base)

    if len(names) > count:
        result = [names[0]]
        rest = names[1:]
        rng.shuffle(rest)
        result.extend(rest[:count - 1])
        return result

    return names[:count] if names else [base]


def _generate_opaque_names(rng: random.Random, count: int = 15) -> list[str]:
    """Generate opaque/coded column names."""
    names: list[str] = []
    for _ in range(count):
        prefix = rng.choice(_OPAQUE_PREFIXES)
        suffix_type = rng.randint(0, 3)
        if suffix_type == 0:
            suffix = str(rng.randint(1, 999))
        elif suffix_type == 1:
            suffix = rng.choice(string.ascii_lowercase) + str(rng.randint(1, 99))
        elif suffix_type == 2:
            suffix = "".join(rng.choices(string.ascii_lowercase, k=2)) + str(rng.randint(1, 9))
        else:
            suffix = str(rng.randint(1, 9)) + "_" + str(rng.randint(1, 9))
        names.append(prefix + suffix)
    return names


# ── Template-based fallback generator ──────────────────────────


def _make_template_generator(
    templates: list[str],
) -> Callable[[random.Random], str]:
    """Create a generator that samples from real data value templates.

    Applies mild perturbation to avoid verbatim copies: character-level
    noise for strings, small numeric jitter for numbers.
    """
    def _gen(rng: random.Random) -> str:
        base = rng.choice(templates)
        # Try numeric perturbation first
        try:
            val = float(base.replace(",", ""))
            jitter = val * rng.uniform(-0.1, 0.1)
            result = val + jitter
            if "." not in base and "e" not in base.lower():
                return str(int(result))
            return f"{result:.2f}"
        except (ValueError, OverflowError):
            pass
        # String perturbation: swap a random character ~30% of the time
        if len(base) > 3 and rng.random() < 0.3:
            chars = list(base)
            idx = rng.randint(0, len(chars) - 1)
            if chars[idx].isalpha():
                chars[idx] = rng.choice(string.ascii_letters)
            elif chars[idx].isdigit():
                chars[idx] = str(rng.randint(0, 9))
            return "".join(chars)
        return base
    return _gen


# ── Public API ──────────────────────────────────────────────────


def generate_synth_tables(
    category_set,
    output_dir: str | Path,
    *,
    value_templates: dict[str, list[str]] | None = None,
    registry=None,
    tables_per_category: int = 2,
    columns_per_table: int = 50,
    rows_per_table: int = 100,
    variants_per_category: int = 30,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate synthetic training tables with a known reference label per column.

    For every taggable category (leaf and internal), generates semantic +
    opaque column name variants with category-appropriate values.  Outputs
    CSV files and reference_labels.json.

    Args:
        category_set: HierarchicalCategorySet (uses all_categories).
        output_dir: Directory to write CSV + reference_labels.json.
        value_templates: Optional {code: [values...]} for template-based generators.
        registry: Optional GeneratorRegistry. When provided, uses registry's
            generators instead of the default GENERATORS dict.
        tables_per_category: Ignored (kept for API compat). Use variants_per_category.
        columns_per_table: Max columns per CSV file.
        rows_per_table: Rows per column.
        variants_per_category: Total name variants per category (semantic + opaque).
        seed: RNG seed for deterministic generation.

    Returns:
        List of table metadata dicts, each carrying its ``reference_labels`` map.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # Build merged generator lookup: registry > hand-coded > template fallbacks
    generators: dict[str, Callable[[random.Random], str]] = {}
    all_cats = getattr(category_set, "all_categories", category_set.categories)

    if registry is not None:
        # Use registry generators
        for cat in all_cats:
            spec = registry.get(cat.code)
            if spec:
                generators[cat.code] = spec.generator
    else:
        # Fall back to GENERATORS dict + template fallbacks
        generators = dict(GENERATORS)
        template_count = 0
        if value_templates:
            for code, values in value_templates.items():
                if code not in generators and len(values) >= 3:
                    generators[code] = _make_template_generator(values)
                    template_count += 1
            if template_count:
                logger.info(
                    "Added %d template generators (%d hand-coded + %d template = %d total)",
                    template_count, len(GENERATORS), template_count, len(generators),
                )

    # Collect all categories (leaf + internal) that have generators
    leaf_specs: list[dict] = []
    for cat in all_cats:
        code = cat.code
        if code not in generators:
            continue
        leaf_specs.append({
            "code": code,
            "label": cat.label,
            "abbrev": getattr(cat, "abbrev", "") or "",
            "common_names": getattr(cat, "common_names", "") or "",
            "generator": generators[code],
        })

    if not leaf_specs:
        logger.warning("No generators found for any leaf category")
        return []

    # Generate columns for each category
    all_columns: dict[str, list[str]] = {}  # column_name → [values]
    reference_labels: dict[str, str] = {}  # column_name → category_code
    seen_names: set[str] = set()

    semantic_count = variants_per_category // 2
    opaque_count = variants_per_category - semantic_count

    for spec in leaf_specs:
        code = spec["code"]
        gen = spec["generator"]

        # Generate column names
        semantic = _generate_semantic_names(
            spec["label"], spec["abbrev"], spec["common_names"], rng, semantic_count,
        )
        opaque = _generate_opaque_names(rng, opaque_count)
        col_names = semantic + opaque

        # Deduplicate against global seen set
        for name in col_names:
            if name in seen_names:
                name = f"{name}_{code.replace('.', '_')}"
            if name in seen_names:
                continue
            seen_names.add(name)

            values = [gen(rng) for _ in range(rows_per_table)]
            all_columns[name] = values
            reference_labels[name] = code

    # Write CSV files in chunks
    col_names_list = list(all_columns.keys())
    results: list[dict[str, Any]] = []
    file_idx = 0

    for chunk_start in range(0, len(col_names_list), columns_per_table):
        chunk_names = col_names_list[chunk_start:chunk_start + columns_per_table]
        file_idx += 1
        csv_path = output_dir / f"synth_{file_idx:03d}.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(chunk_names)
            for row_idx in range(rows_per_table):
                writer.writerow(all_columns[name][row_idx] for name in chunk_names)

        results.append({
            "table_name": csv_path.stem,
            "database": "synth",
            "column_count": len(chunk_names),
            "row_count": rows_per_table,
            "reference_labels": {n: reference_labels[n] for n in chunk_names},
        })

    # Write reference labels sidecar
    ref_path = output_dir / "reference_labels.json"
    with open(ref_path, "w") as f:
        json.dump(reference_labels, f, indent=2)

    logger.info(
        "Generated %d columns across %d files for %d categories in %s",
        len(reference_labels), file_idx, len(leaf_specs), output_dir,
    )

    return results


def generate_user_taxonomy_corpus(
    category_set,
    payloads: dict[str, dict],
    output_dir: str | Path,
    *,
    seed: int = 42,
    rows_per_table: int = 100,
    columns_per_table: int = 50,
    variants_per_category: int = 30,
    name_hints: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Generate synthetic corpus labeled with user-taxonomy codes.

    Uses enrichment payloads as the primary generator source.
    Returns ``(table_metadata, coverage_report)`` where coverage maps
    every taggable code (leaf + internal) to its generator source.
    """
    from atelier.classify.synth_registry import GeneratorRegistry

    registry = GeneratorRegistry.from_enrichment_payloads(payloads, category_set)
    coverage = registry.coverage_report(category_set)

    results = generate_synth_tables(
        category_set,
        output_dir,
        registry=registry,
        seed=seed,
        rows_per_table=rows_per_table,
        columns_per_table=columns_per_table,
        variants_per_category=variants_per_category,
    )

    return results, coverage
