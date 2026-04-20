#!/usr/bin/env python3
"""Generate a large synthetic 'synth' database with ~100 tables.

Creates ~100 mixed-domain tables whose column counts follow a normal
distribution centered on 100 columns (sigma=30, clipped to [30, 200]).
With 316 leaf categories, each category is reused ~30x across tables
with different column name variants — same value generators, different
surface-level naming to produce a realistic, large-scale data source.

Mirrors the pattern of generate_sample_source.py but at ~10x scale:
  - data/synth/tables/*.csv          — ~100 tables, 100 rows each
  - data/synth/reference_labels.json — table.column -> category mapping

Usage:
    uv run python scripts/generate_synth_source.py [--tables 100] [--center 100] [--seed 7]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import string
import sys
from pathlib import Path

# ── Bootstrap: import synth_generators without triggering full atelier ──

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "synth_generators",
    PROJECT_ROOT / "src" / "atelier" / "classify" / "synth_generators.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
build_generators = _mod.build_generators

ONTOLOGY_PATH = PROJECT_ROOT / "data" / "sample" / "ontology.json"
SYNTH_DIR = PROJECT_ROOT / "data" / "synth"
TABLES_DIR = SYNTH_DIR / "tables"
REFERENCE_LABELS_PATH = SYNTH_DIR / "reference_labels.json"

ROWS_PER_TABLE = 100

# ── Table name generation ──────────────────────────────────────────

# Realistic table name pools organized by domain. Mixed naming styles
# (snake_case, abbreviations, opaque) to prevent schema-pattern bias.
_TABLE_NAME_POOLS = {
    "crm": [
        "customers", "contacts", "leads", "accounts", "opportunities",
        "customer_master", "crm_contacts", "client_profiles", "prospects",
        "subscriber_list", "customer_360", "account_details",
    ],
    "finance": [
        "transactions", "payments", "invoices", "ledger_entries",
        "billing_records", "revenue_detail", "expense_report",
        "financial_summary", "accounts_receivable", "accounts_payable",
        "journal_entries", "credit_memos",
    ],
    "hr": [
        "employees", "payroll", "benefits", "performance_reviews",
        "hiring_pipeline", "hr_records", "compensation", "workforce",
        "employee_directory", "time_tracking", "leave_requests",
    ],
    "product": [
        "products", "catalog", "inventory", "product_variants",
        "sku_master", "pricing", "product_metadata", "item_master",
        "warehouse_stock", "product_categories", "assortment",
    ],
    "health": [
        "patients", "encounters", "diagnoses", "prescriptions",
        "lab_results", "patient_records", "clinical_notes", "vitals",
        "health_screenings", "immunization_records",
    ],
    "tech": [
        "servers", "deployments", "incidents", "service_configs",
        "api_logs", "infrastructure", "monitoring_metrics",
        "host_inventory", "network_devices", "cert_registry",
    ],
    "analytics": [
        "events", "page_views", "sessions", "conversions",
        "funnel_stages", "cohort_analysis", "ab_test_results",
        "engagement_metrics", "user_activity", "attribution_data",
    ],
    "content": [
        "articles", "publications", "media_library", "content_items",
        "documents", "knowledge_base", "assets", "attachments",
        "editorial_calendar", "content_versions",
    ],
    "geo": [
        "locations", "regions", "territories", "site_directory",
        "address_book", "geo_lookup", "postal_codes", "branch_offices",
        "distribution_centers",
    ],
    "logistics": [
        "shipments", "routes", "fleet_vehicles", "delivery_records",
        "tracking_events", "carrier_rates", "warehouse_transfers",
        "customs_declarations",
    ],
    "research": [
        "experiments", "research_data", "survey_responses", "datasets",
        "observations", "sensor_readings", "measurement_log",
        "calibration_records",
    ],
    "compliance": [
        "audit_log", "access_control", "policy_registry", "consent_records",
        "risk_assessments", "compliance_checks", "retention_schedule",
        "data_classification",
    ],
    "opaque": [
        "dataset_7", "raw_import", "stg_load_001", "tmp_extract",
        "feed_20240101", "batch_input", "dw_fact_001", "dim_lookup",
        "etl_staging", "data_dump", "flat_export", "src_system_a",
        "migration_tmp", "legacy_table", "ext_feed_daily",
    ],
}

# Assign ontology subtrees to domain pools for weighted table composition
_DOMAIN_CATEGORY_PREFIXES = {
    "crm": [
        "ICE.SENSITIVE.PID.IDENTITY", "ICE.SENSITIVE.PID.CONTACT",
        "ICE.METADATA.RECID", "ICE.METADATA.STATUS",
    ],
    "finance": [
        "ICE.SENSITIVE.PID.FINANCIAL", "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.PRICE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.REVENUE",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.ISO_CURRENCY",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.CURRENCY_NAME",
    ],
    "hr": [
        "ICE.SENSITIVE.PID.IDENTITY.NAME", "ICE.SENSITIVE.PID.IDENTITY.DOB",
        "ICE.SENSITIVE.PID.FINANCIAL.INCOME", "ICE.SENSITIVE.BUSINESS.EMPLOYEE_RECORD",
        "ICE.SENSITIVE.BUSINESS.PERFORMANCE_REVIEW",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.DEPARTMENT",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.ROLE",
    ],
    "product": [
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.PRODUCT",
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.BRAND",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.SKU",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.GTIN",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.CATEGORY",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT",
    ],
    "health": [
        "ICE.SENSITIVE.PID.HEALTH",
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.DRUG",
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.DISEASE",
    ],
    "tech": [
        "ICE.SENSITIVE.TECHNICAL",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.CPU_USAGE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.MEMORY_USAGE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.BANDWIDTH",
    ],
    "analytics": [
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.CHANNEL",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.BROWSER",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.DEVICE_TYPE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.SCORE",
    ],
    "content": [
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT",
        "ICE.NONSENSITIVE.DESIGNATIVE.TITLE",
        "ICE.NONSENSITIVE.DESIGNATIVE.LABEL",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.DOI",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.ISBN",
    ],
    "geo": [
        "ICE.NONSENSITIVE.DESIGNATIVE.GEO",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.LATITUDE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.LONGITUDE",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.ISO_COUNTRY",
    ],
    "logistics": [
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.VIN",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.IATA",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.DISTANCE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.WEIGHT",
        "ICE.NONSENSITIVE.PRESCRIPTIVE.ROUTE",
    ],
    "research": [
        "ICE.NONSENSITIVE.DESCRIPTIVE.STATISTICAL",
        "ICE.NONSENSITIVE.DESCRIPTIVE.EMBEDDING",
        "ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC",
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.SCIENTIFIC",
    ],
    "compliance": [
        "ICE.METADATA", "ICE.NONSENSITIVE.PRESCRIPTIVE",
        "ICE.SENSITIVE.BUSINESS",
    ],
}


# ── Column naming ──────────────────────────────────────────────────

_OPAQUE_PREFIXES = [
    "field_", "col_", "meta_", "v_", "x_", "dim_", "f_",
    "attr_", "var_", "c_", "d_", "val_", "p_",
]

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


def _snake_case(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower().replace(" ", "_")


def _camel_case(s: str) -> str:
    parts = s.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _generate_column_name(
    cat: dict,
    rng: random.Random,
    *,
    opaque: bool = False,
    variant_idx: int = 0,
) -> str:
    """Generate a column name variant for a category.

    Different variant_idx values produce different naming styles to ensure
    variety when the same category appears across many tables.
    """
    if opaque:
        prefix = rng.choice(_OPAQUE_PREFIXES)
        style = rng.randint(0, 3)
        if style == 0:
            return f"{prefix}{rng.randint(1, 999)}"
        elif style == 1:
            return f"{prefix}{''.join(rng.choices(string.ascii_lowercase, k=rng.randint(2, 4)))}"
        elif style == 2:
            parts = cat["code"].split(".")
            token = parts[-1].lower()[:rng.randint(3, 5)]
            return f"{prefix}{token}"
        else:
            return f"{''.join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 6)))}{rng.randint(1, 99)}"

    label = cat.get("label", cat["code"].split(".")[-1])
    common_names = cat.get("common_names", "")
    abbrev = cat.get("abbrev", "")
    base = _snake_case(label)

    # Rotate through naming styles based on variant_idx
    style = variant_idx % 8

    if style == 0:
        # Plain snake_case
        return base
    elif style == 1:
        # camelCase
        return _camel_case(base)
    elif style == 2:
        # UPPER
        return base.upper()
    elif style == 3 and abbrev:
        # Abbreviation
        return abbrev.strip().lower()
    elif style == 4 and common_names:
        # Pick from common names
        variants = [cn.strip() for cn in re.split(r"[,|]", common_names) if cn.strip()]
        if variants:
            return _snake_case(rng.choice(variants))
    elif style == 5:
        # Prefixed variant
        prefix = rng.choice(["user_", "customer_", "src_", "raw_", "dim_", "fact_"])
        return prefix + base
    elif style == 6:
        # Synonym-expanded
        words = base.split("_")
        for i, word in enumerate(words):
            if word in SYNONYMS:
                syn = rng.choice(SYNONYMS[word])
                return "_".join(words[:i] + [syn] + words[i + 1:])
    elif style == 7:
        # Suffixed variant
        suffix = rng.choice(["_val", "_field", "_data", "_info", "_raw", "_v2"])
        return base + suffix

    # Default fallback: base + numeric discriminator
    return f"{base}_{variant_idx}" if variant_idx > 0 else base


# ── Table composition ──────────────────────────────────────────────


def _categorize_leaves(cats: list[dict]) -> tuple[set[str], dict[str, list[str]]]:
    """Return (all_leaf_codes, domain -> [leaf_codes]) mapping."""
    leaves = {
        c["code"]
        for c in cats
        if not any(c2["parent_code"] == c["code"] for c2 in cats)
    }

    domain_leaves: dict[str, list[str]] = {}
    for domain, prefixes in _DOMAIN_CATEGORY_PREFIXES.items():
        matched = sorted(
            code for code in leaves
            if any(code == p or code.startswith(p + ".") for p in prefixes)
        )
        domain_leaves[domain] = matched

    return leaves, domain_leaves


def _assign_tables(
    cats: list[dict],
    leaves: set[str],
    domain_leaves: dict[str, list[str]],
    generators: dict,
    rng: random.Random,
    *,
    num_tables: int = 100,
    center_columns: int = 100,
    sigma_columns: int = 30,
    min_columns: int = 30,
    max_columns: int = 200,
) -> list[tuple[str, list[str]]]:
    """Build ~num_tables tables with column counts ~ N(center, sigma).

    Strategy:
    1. Draw column counts from a truncated normal distribution.
    2. For each table, pick a primary domain (weighted round-robin) and
       fill 50-70% of slots from that domain's categories, then top up
       with cross-domain columns from the full leaf pool.
    3. Each leaf category can appear in multiple tables (with different
       column names) — this is intentional to reach ~10k total columns.
    """
    # Only use leaves that have generators
    usable_leaves = sorted(code for code in leaves if code in generators)
    if not usable_leaves:
        raise ValueError("No usable leaf categories with generators")

    # Build flat list of (domain, leaf_code) for weighted sampling
    domain_pools: dict[str, list[str]] = {}
    for domain, codes in domain_leaves.items():
        usable = [c for c in codes if c in generators]
        if usable:
            domain_pools[domain] = usable

    domains = list(domain_pools.keys())

    # Generate column counts
    col_counts: list[int] = []
    for _ in range(num_tables):
        n = int(rng.gauss(center_columns, sigma_columns))
        n = max(min_columns, min(max_columns, n))
        col_counts.append(n)

    # Generate table names
    used_names: set[str] = set()
    table_names: list[str] = []

    # Cycle through domains to assign primary domain per table
    domain_cycle = list(domains) * (num_tables // len(domains) + 1)
    rng.shuffle(domain_cycle)

    for i in range(num_tables):
        domain = domain_cycle[i % len(domain_cycle)]
        pool = _TABLE_NAME_POOLS.get(domain, _TABLE_NAME_POOLS["opaque"])
        # Find an unused name
        candidates = [n for n in pool if n not in used_names]
        if not candidates:
            # Generate a synthetic table name
            name = f"{domain}_{i + 1:03d}"
        else:
            name = rng.choice(candidates)
        used_names.add(name)
        table_names.append(name)

    # Assign columns to tables
    # Track how many times each category has been used (for variant naming)
    category_usage_count: dict[str, int] = {c: 0 for c in usable_leaves}

    tables: list[tuple[str, list[str]]] = []

    for i in range(num_tables):
        target_cols = col_counts[i]
        domain = domain_cycle[i % len(domain_cycle)]

        cols: list[str] = []
        seen_in_table: set[str] = set()

        # Primary domain fill: 50-70% of columns
        primary_ratio = rng.uniform(0.50, 0.70)
        primary_count = int(target_cols * primary_ratio)

        if domain in domain_pools:
            primary_pool = domain_pools[domain]
            for _ in range(primary_count):
                code = rng.choice(primary_pool)
                cols.append(code)
                seen_in_table.add(code)
        else:
            primary_count = 0

        # Cross-domain fill: remaining columns from full pool
        remaining = target_cols - len(cols)
        for _ in range(remaining):
            code = rng.choice(usable_leaves)
            cols.append(code)
            seen_in_table.add(code)

        # Always sprinkle in some metadata columns (realistic tables have these)
        metadata_codes = [
            c for c in usable_leaves if c.startswith("ICE.METADATA.")
        ]
        if metadata_codes:
            meta_count = rng.randint(2, min(6, len(metadata_codes)))
            meta_sample = rng.sample(metadata_codes, min(meta_count, len(metadata_codes)))
            # Replace some cross-domain slots with metadata
            for j, mc in enumerate(meta_sample):
                if len(cols) > primary_count + j:
                    cols[primary_count + j] = mc

        # Shuffle to prevent positional patterns
        rng.shuffle(cols)

        # Update usage counts
        for code in cols:
            category_usage_count[code] = category_usage_count.get(code, 0) + 1

        tables.append((table_names[i], cols))

    return tables


# ── Main generation ────────────────────────────────────────────────


def generate_synth_database(
    *,
    num_tables: int = 100,
    center_columns: int = 100,
    sigma_columns: int = 30,
    rows_per_table: int = 100,
    seed: int = 7,
) -> None:
    """Generate the synth database."""
    rng = random.Random(seed)

    # Load ontology
    with open(ONTOLOGY_PATH) as f:
        cats = json.load(f)
    cat_lookup = {c["code"]: c for c in cats}

    # Initialize generators
    generators = build_generators()

    # Identify leaves and domain mapping
    leaves, domain_leaves = _categorize_leaves(cats)

    # Check generator coverage
    covered = sum(1 for c in leaves if c in generators)
    missing = sorted(c for c in leaves if c not in generators)
    print(f"Ontology: {len(cats)} categories, {len(leaves)} leaves")
    print(f"Generator coverage: {covered}/{len(leaves)}")
    if missing:
        print(f"  Missing generators: {len(missing)} (will use fallback)")
        for c in missing[:5]:
            print(f"    {c}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")

    # Build table assignments
    tables = _assign_tables(
        cats, leaves, domain_leaves, generators, rng,
        num_tables=num_tables,
        center_columns=center_columns,
        sigma_columns=sigma_columns,
    )

    # Create output directories
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Decide which columns get opaque names (~25%)
    opaque_rng = random.Random(seed + 1)

    # Track category variant indices (how many times each code has been named)
    variant_counter: dict[str, int] = {}

    reference_labels: dict[str, str] = {}
    total_columns = 0
    opaque_count = 0

    for table_name, codes in tables:
        csv_path = TABLES_DIR / f"{table_name}.csv"
        col_entries: list[tuple[str, str]] = []  # (col_name, code)
        used_col_names: set[str] = set()

        for code in codes:
            is_opaque = opaque_rng.random() < 0.25
            cat = cat_lookup.get(code, {"code": code, "label": code.split(".")[-1]})

            # Get variant index for this category
            vid = variant_counter.get(code, 0)
            variant_counter[code] = vid + 1

            col_name = _generate_column_name(
                cat, rng, opaque=is_opaque, variant_idx=vid,
            )

            # Ensure uniqueness within table
            base_name = col_name
            suffix_idx = 2
            while col_name in used_col_names:
                col_name = f"{base_name}_{suffix_idx}"
                suffix_idx += 1
            used_col_names.add(col_name)

            col_entries.append((col_name, code))
            if is_opaque:
                opaque_count += 1

        # Generate CSV
        col_names = [e[0] for e in col_entries]
        col_codes = [e[1] for e in col_entries]

        fallback_gen = lambda rng: f"value_{rng.randint(1, 9999)}"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(col_names)
            for _ in range(rows_per_table):
                row = []
                for code in col_codes:
                    gen = generators.get(code, fallback_gen)
                    row.append(gen(rng))
                writer.writerow(row)

        # Record reference labels
        for col_name, code in col_entries:
            ref_key = f"{table_name}.{col_name}"
            reference_labels[ref_key] = code
            total_columns += 1

    # Write reference labels
    with open(REFERENCE_LABELS_PATH, "w") as f:
        json.dump(reference_labels, f, indent=2, ensure_ascii=False)

    # ── Summary ────────────────────────────────────────────────────
    col_counts = [len(codes) for _, codes in tables]
    mean_cols = sum(col_counts) / len(col_counts)
    std_cols = math.sqrt(sum((c - mean_cols) ** 2 for c in col_counts) / len(col_counts))
    min_cols = min(col_counts)
    max_cols = max(col_counts)

    # Category coverage: how many distinct categories were used?
    used_cats = set()
    for _, codes in tables:
        used_cats.update(codes)

    print(f"\nGenerated {len(tables)} tables with {total_columns} total columns")
    print(f"  Output:       {TABLES_DIR}/")
    print(f"  Reference labels: {REFERENCE_LABELS_PATH}")
    print(f"  Rows/table:   {rows_per_table}")
    print(f"  Column distribution: mean={mean_cols:.1f}, std={std_cols:.1f}, "
          f"min={min_cols}, max={max_cols}")
    print(f"  Distinct categories used: {len(used_cats)}/{len(leaves)}")
    print(f"  Opaque column names: {opaque_count}/{total_columns} "
          f"({opaque_count * 100 // total_columns}%)")
    print(f"\nTable breakdown:")
    for name, codes in sorted(tables, key=lambda t: -len(t[1]))[:10]:
        print(f"  {name}: {len(codes)} columns")
    print(f"  ... ({len(tables) - 10} more)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a large synthetic 'synth' database for Atelier."
    )
    parser.add_argument(
        "--tables", type=int, default=100,
        help="Number of tables to generate (default: 100)",
    )
    parser.add_argument(
        "--center", type=int, default=100,
        help="Mean column count per table (default: 100)",
    )
    parser.add_argument(
        "--sigma", type=int, default=30,
        help="Std dev of column count distribution (default: 30)",
    )
    parser.add_argument(
        "--rows", type=int, default=100,
        help="Rows per table (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=7,
        help="RNG seed for reproducibility (default: 7)",
    )
    args = parser.parse_args()

    generate_synth_database(
        num_tables=args.tables,
        center_columns=args.center,
        sigma_columns=args.sigma,
        rows_per_table=args.rows,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
