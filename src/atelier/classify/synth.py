"""Synthetic data generation for CatBoost/SVM classifier training.

Generates synthetic columns with known ground truth labels for each leaf
category in the controlled vocabulary. Each category gets both semantic
column names (human-readable variants) and opaque names (coded/random)
to force classifiers to learn from VALUE PATTERNS, not just names.

Ported from signals/scripts/generate_meta_tagging_train.py, scoped to
the 24-category mock taxonomy.
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


# ── Value generators ────────────────────────────────────────────

_FIRST_NAMES = [
    "James", "John", "Robert", "Michael", "William", "David", "Richard",
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Susan", "Jessica",
    "Sarah", "Emily", "Amy", "Anna", "Emma", "Sophia", "Olivia",
]
_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas",
    "Jackson", "White", "Harris", "Martin", "Thompson", "Moore", "Lee",
]
_CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Seattle",
    "Denver", "Boston", "Austin", "Portland", "Miami", "Dallas", "Atlanta",
]
_STATES = ["CA", "TX", "NY", "FL", "WA", "IL", "CO", "MA", "GA", "PA"]
_STREETS = [
    "Main St", "Oak Ave", "Maple Dr", "Cedar Ln", "Pine Rd",
    "Elm St", "Birch Way", "Park Blvd", "Lake Dr", "Hill Rd",
]
_DOMAINS = ["gmail.com", "outlook.com", "yahoo.com", "company.com", "protonmail.com"]
_STATUSES = ["active", "inactive", "pending", "completed", "failed", "cancelled", "approved", "rejected"]


def _gen_email(rng: random.Random) -> str:
    first = rng.choice(_FIRST_NAMES).lower()
    last = rng.choice(_LAST_NAMES).lower()
    sep = rng.choice([".", "_", ""])
    domain = rng.choice(_DOMAINS)
    suffix = "" if rng.random() > 0.3 else str(rng.randint(1, 99))
    return f"{first}{sep}{last}{suffix}@{domain}"


def _gen_phone(rng: random.Random) -> str:
    area = rng.randint(200, 999)
    mid = rng.randint(200, 999)
    last = rng.randint(1000, 9999)
    fmt = rng.choice(["({}) {}-{}", "{}-{}-{}", "{}.{}.{}", "+1{}{}{}", "{} {} {}"])
    return fmt.format(area, mid, last)


def _gen_address(rng: random.Random) -> str:
    num = rng.randint(100, 9999)
    street = rng.choice(_STREETS)
    city = rng.choice(_CITIES)
    state = rng.choice(_STATES)
    zipcode = rng.randint(10000, 99999)
    return f"{num} {street}, {city}, {state} {zipcode}"


def _gen_full_name(rng: random.Random) -> str:
    first = rng.choice(_FIRST_NAMES)
    last = rng.choice(_LAST_NAMES)
    if rng.random() < 0.2:
        middle = rng.choice(string.ascii_uppercase)
        return f"{first} {middle}. {last}"
    return f"{first} {last}"


def _gen_dob(rng: random.Random) -> str:
    year = rng.randint(1940, 2010)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    fmt = rng.choice(["{:04d}-{:02d}-{:02d}", "{:02d}/{:02d}/{:04d}", "{:02d}-{:02d}-{:04d}"])
    if "{:04d}" == fmt[:5]:
        return fmt.format(year, month, day)
    return fmt.format(month, day, year)


def _gen_ssn(rng: random.Random) -> str:
    a = rng.randint(100, 899)
    b = rng.randint(10, 99)
    c = rng.randint(1000, 9999)
    if rng.random() < 0.3:
        return f"XXX-XX-{c}"
    return f"{a}-{b}-{c}"


def _gen_credit_card(rng: random.Random) -> str:
    prefix = rng.choice(["4", "5", "37", "6011"])
    remaining = 16 - len(prefix)
    digits = prefix + "".join(str(rng.randint(0, 9)) for _ in range(remaining))
    if rng.random() < 0.3:
        return f"XXXX-XXXX-XXXX-{digits[-4:]}"
    return f"{digits[:4]}-{digits[4:8]}-{digits[8:12]}-{digits[12:]}"


def _gen_bank_account(rng: random.Random) -> str:
    length = rng.choice([8, 10, 12])
    return "".join(str(rng.randint(0, 9)) for _ in range(length))


def _gen_amount(rng: random.Random) -> str:
    val = round(rng.uniform(0.50, 99999.99), 2)
    if rng.random() < 0.5:
        return f"${val:,.2f}"
    return f"{val:.2f}"


def _gen_ipv4(rng: random.Random) -> str:
    return ".".join(str(rng.randint(1, 254)) for _ in range(4))


def _gen_uuid(rng: random.Random) -> str:
    hexchars = "0123456789abcdef"
    parts = [
        "".join(rng.choices(hexchars, k=8)),
        "".join(rng.choices(hexchars, k=4)),
        "4" + "".join(rng.choices(hexchars, k=3)),
        rng.choice("89ab") + "".join(rng.choices(hexchars, k=3)),
        "".join(rng.choices(hexchars, k=12)),
    ]
    return "-".join(parts)


def _gen_url(rng: random.Random) -> str:
    domain = rng.choice(["example.com", "acme.org", "data.io", "api.service.com"])
    path = "/".join(rng.choices(["users", "api", "v2", "data", "docs", "status"], k=rng.randint(1, 3)))
    return f"https://{domain}/{path}"


def _gen_timestamp(rng: random.Random) -> str:
    year = rng.randint(2018, 2026)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    if rng.random() < 0.5:
        return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}Z"
    return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"


def _gen_record_id(rng: random.Random) -> str:
    if rng.random() < 0.6:
        return str(rng.randint(1, 9999999))
    prefix = rng.choice(["REC", "ID", "TXN", "ORD", "USR"])
    return f"{prefix}-{rng.randint(100000, 999999)}"


def _gen_status(rng: random.Random) -> str:
    return rng.choice(_STATUSES)


def _gen_generic_string(rng: random.Random) -> str:
    """Not sensitive — generic filler data."""
    templates = [
        lambda: "".join(rng.choices(string.ascii_letters, k=rng.randint(5, 20))),
        lambda: f"item_{rng.randint(1, 10000)}",
        lambda: rng.choice(["red", "blue", "green", "yellow", "black", "white"]),
        lambda: f"category_{rng.choice(string.ascii_uppercase)}",
        lambda: str(rng.randint(1, 100)),
    ]
    return rng.choice(templates)()


def _gen_internal_code(rng: random.Random) -> str:
    """Internal non-sensitive — department/system codes."""
    prefixes = ["DEPT", "DIV", "SYS", "MOD", "GRP", "UNIT"]
    return f"{rng.choice(prefixes)}-{rng.randint(100, 999)}"


# ── Generator registry ──────────────────────────────────────────

_GENERATORS: dict[str, Callable[[random.Random], str]] = {
    "0": _gen_generic_string,
    "0.1": _gen_internal_code,
    "1.1.1.1": _gen_email,
    "1.1.1.2": _gen_phone,
    "1.1.1.3": _gen_address,
    "1.1.2.1": _gen_full_name,
    "1.1.2.2": _gen_dob,
    "1.1.2.3": _gen_ssn,
    "1.1.3.1": _gen_credit_card,
    "1.1.3.2": _gen_bank_account,
    "1.1.3.3": _gen_amount,
    "1.2.1": _gen_ipv4,
    "1.2.2": _gen_uuid,
    "1.2.3": _gen_url,
    "2.1": _gen_timestamp,
    "2.2": _gen_record_id,
    "2.3": _gen_status,
}


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
    tables_per_category: int = 2,
    columns_per_table: int = 50,
    rows_per_table: int = 100,
    variants_per_category: int = 30,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate synthetic training tables with known ground truth.

    For each leaf category, generates semantic + opaque column name variants
    with category-appropriate values. Outputs CSV files and ground_truth.json.

    Args:
        category_set: HierarchicalCategorySet with leaf categories.
        output_dir: Directory to write CSV + ground_truth.json.
        value_templates: Optional {code: [values...]} for template-based generators.
            When provided, categories without hand-coded generators use template
            sampling with mild perturbation.
        tables_per_category: Ignored (kept for API compat). Use variants_per_category.
        columns_per_table: Max columns per CSV file.
        rows_per_table: Rows per column.
        variants_per_category: Total name variants per category (semantic + opaque).
        seed: RNG seed for deterministic generation.

    Returns:
        List of table metadata dicts with ground_truth.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # Build merged generator registry: hand-coded + template fallbacks
    generators = dict(_GENERATORS)
    template_count = 0
    if value_templates:
        for code, values in value_templates.items():
            if code not in generators and len(values) >= 3:
                generators[code] = _make_template_generator(values)
                template_count += 1
        if template_count:
            logger.info(
                "Added %d template generators (%d hand-coded + %d template = %d total)",
                template_count, len(_GENERATORS), template_count, len(generators),
            )

    # Collect leaf categories that have generators
    leaf_specs: list[dict] = []
    for cat in category_set.categories:
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
    ground_truth: dict[str, str] = {}  # column_name → category_code
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
            ground_truth[name] = code

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
            "ground_truth": {n: ground_truth[n] for n in chunk_names},
        })

    # Write ground truth
    gt_path = output_dir / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    logger.info(
        "Generated %d columns across %d files for %d categories in %s",
        len(ground_truth), file_idx, len(leaf_specs), output_dir,
    )

    return results
