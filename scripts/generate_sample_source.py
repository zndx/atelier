#!/usr/bin/env python3
"""Generate OOTB sample data from the expanded ontology.

Creates ~25 realistic mixed-domain tables whose columns are typed by
our ICE.* vocabulary. Tables mix columns from different ontology
subtrees — like real relational tables (a `customers` table has name +
email + status + created_at from different branches). ~25% of columns
get opaque/abbreviated names to prevent the classifier from relying
on naming alone.

Since we control both the vocabulary AND the data generation, ground
truth is tautologically correct — every column is labeled with the
ICE.* code it was generated for.

Usage:
    uv run python scripts/generate_sample_source.py

Output:
    data/sample/tables/*.csv          — ~25 mixed-domain tables, 100 rows each
    data/sample/reference_labels.json — column → category mapping
"""

from __future__ import annotations

import csv
import json
import random
import string
import sys
from pathlib import Path

# Allow importing synth_generators directly (avoids numpy dependency
# chain from atelier.classify.__init__.py)
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
TABLES_DIR = PROJECT_ROOT / "data" / "sample" / "tables"
REFERENCE_LABELS_PATH = PROJECT_ROOT / "data" / "sample" / "reference_labels.json"

ROWS_PER_TABLE = 100
SEED = 42

# ── Generator registry (shared) ───────────────────────────────────
#
# All value generators come from synth_generators.py — the shared
# library used by both this script and the classify.synth module.
# build_generators() returns the full code → generator mapping.
_GENERATORS: dict[str, callable] = {}


def _init_generators():
    """Initialize _GENERATORS from the shared synth_generators module."""
    global _GENERATORS
    _GENERATORS = build_generators()


# NOTE: All inline generator functions and _register_generators() have been
# removed. Generators are now in src/atelier/classify/synth_generators.py.
# The _GENERATORS dict is populated by _init_generators() → build_generators().

# ── Table definitions ────────────────────────────────────────────
#
# Realistic mixed-domain tables inspired by GitTables organic patterns.
# Each table has a primary domain but mixes in columns from other
# subtrees — just like real relational tables where a `customers`
# table has name + email + status + created_at from different
# ontology branches.
#
# ~25% of columns get opaque/abbreviated names to ensure the
# classifier can't rely on naming alone.

# Code prefix helpers
def _codes_matching(leaves: set[str], *prefixes: str) -> list[str]:
    """Return sorted leaf codes matching any of the given prefixes."""
    return sorted(c for c in leaves if any(c == p or c.startswith(p + ".") for p in prefixes))


def _build_table_assignments(cats: list[dict], rng: random.Random) -> list[tuple[str, list[str]]]:
    """Assign all 300 leaf categories to ~25 realistic mixed-domain tables.

    Design principles:
    - Each table has a primary domain (its "theme") PLUS cross-domain columns
      (metadata timestamps, status fields, IDs) mixed in — like real tables.
    - Some tables are deliberately domain-ambiguous ("raw_import", "dataset_7")
      to force value-pattern classification over schema-pattern classification.
    - Every leaf category appears in exactly one table.
    """
    leaves = {c["code"] for c in cats if not any(c2["parent_code"] == c["code"] for c2 in cats)}
    assigned: set[str] = set()

    def _take(codes: list[str]) -> list[str]:
        """Take codes that haven't been assigned yet."""
        result = [c for c in codes if c not in assigned]
        assigned.update(result)
        return result

    def _take_n(codes: list[str], n: int) -> list[str]:
        """Take up to n unassigned codes."""
        result = []
        for c in codes:
            if c not in assigned and len(result) < n:
                result.append(c)
                assigned.add(c)
        return result

    tables: list[tuple[str, list[str]]] = []

    # ── 1. customers — identity + contact + metadata (like a CRM table) ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.SENSITIVE.PID.IDENTITY.NAME.FULLNAME",
        "ICE.SENSITIVE.PID.IDENTITY.NAME.FIRST_NAME",
        "ICE.SENSITIVE.PID.IDENTITY.NAME.LAST_NAME",
        "ICE.SENSITIVE.PID.IDENTITY.DOB",
        "ICE.SENSITIVE.PID.IDENTITY.GENDER_PII",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.SENSITIVE.PID.CONTACT.EMAIL",
        "ICE.SENSITIVE.PID.CONTACT.PHONE",
        "ICE.SENSITIVE.PID.CONTACT.ADDRESS",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.METADATA.RECID",
        "ICE.METADATA.CREATED_AT",
        "ICE.METADATA.STATUS",
        "ICE.METADATA.TENANT_ID",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.INDUSTRY",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.CHANNEL",
    ))
    tables.append(("customers", cols))

    # ── 2. orders — financial + temporal + categorical ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.PAN",
        "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.TXNAMT",
        "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.CVV",
        "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.EXPIRY",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.DATE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.DATETIME",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.STATUS",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.PAYMENT_METHOD",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.SHIPPING_METHOD",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.SKU",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.PRICE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.COUNT",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.METADATA.TIMESTAMP",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.CURRENCY_NAME",
    ))
    tables.append(("orders", cols))

    # ── 3. employees — HR-style table mixing identity + org + financial ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.SENSITIVE.PID.IDENTITY.NAME.MIDDLE_NAME",
        "ICE.SENSITIVE.PID.IDENTITY.NAME.MAIDEN_NAME",
        "ICE.SENSITIVE.PID.IDENTITY.NATIONALITY",
        "ICE.SENSITIVE.PID.IDENTITY.MARITAL_STATUS",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.SENSITIVE.PID.FINANCIAL.INCOME.SALARY",
        "ICE.SENSITIVE.PID.IDENTITY.GOVID.TAX_ID",
        "ICE.SENSITIVE.PID.FINANCIAL.ACCOUNT.BAN",
        "ICE.SENSITIVE.PID.FINANCIAL.ACCOUNT.ROUTING_NUM",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.DEPARTMENT",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.ROLE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.EDUCATION_LEVEL",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.METADATA.CREATED_BY",
        "ICE.METADATA.MODIFIED_AT",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.START_DATE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.AGE",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.SENSITIVE.BUSINESS.EMPLOYEE_RECORD",
        "ICE.SENSITIVE.BUSINESS.PERFORMANCE_REVIEW",
    ))
    tables.append(("employees", cols))

    # ── 4. products — catalog table ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.PRODUCT",
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.BRAND",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.GTIN",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.DESCRIPTION",
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.KEYWORDS",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.CATEGORY",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.COLOR",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.SIZE",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.WEIGHT",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.LENGTH",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.REVENUE",
        "ICE.NONSENSITIVE.PRESCRIPTIVE.LICENSE",
    ))
    tables.append(("products", cols))

    # ── 5. patients — health + identity + contact ──
    cols = []
    cols += _take(_codes_matching(leaves, "ICE.SENSITIVE.PID.HEALTH"))
    cols += _take(_codes_matching(leaves,
        "ICE.SENSITIVE.PID.IDENTITY.BIRTH_PLACE",
        "ICE.SENSITIVE.PID.IDENTITY.ETHNICITY",
        "ICE.SENSITIVE.PID.IDENTITY.BIOMETRIC.PHOTO",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.SENSITIVE.PID.FINANCIAL.CREDIT.INSURANCE_ID",
        "ICE.SENSITIVE.PID.CONTACT.FAX",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.GENDER",
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.PERIOD",
    ))
    tables.append(("patients", cols))

    # ── 6. transactions — financial detail ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.SENSITIVE.PID.FINANCIAL.CREDIT.CREDIT_SCORE",
        "ICE.SENSITIVE.PID.FINANCIAL.ACCOUNT.INVESTMENT",
        "ICE.SENSITIVE.PID.FINANCIAL.ACCOUNT.CRYPTO_ADDR",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.EXCHANGE_RATE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.MARKET_VALUE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.BUDGET",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.TICKER",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.LEI",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.CUSIP",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.ISO_CURRENCY",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.QUARTER",
        "ICE.METADATA.ETL_BATCH",
    ))
    tables.append(("transactions", cols))

    # ── 7. servers — infra monitoring ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.SENSITIVE.TECHNICAL.HOSTNAME",
        "ICE.SENSITIVE.TECHNICAL.IPADDR",
        "ICE.SENSITIVE.TECHNICAL.DEVID",
        "ICE.SENSITIVE.TECHNICAL.SSH_KEY",
        "ICE.SENSITIVE.TECHNICAL.CERTIFICATE",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.CPU_USAGE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.MEMORY_USAGE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.UPTIME",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.BANDWIDTH",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.STORAGE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.LATENCY",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.PLATFORM",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.RESOLUTION",
    ))
    cols += _take(_codes_matching(leaves, "ICE.METADATA.LOAD_TIMESTAMP"))
    tables.append(("servers", cols))

    # ── 8. events — mixed designative + temporal ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.EVENT",
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.VENUE",
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.GEO",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.GEO.CITY",
        "ICE.NONSENSITIVE.DESIGNATIVE.GEO.COUNTRY",
        "ICE.NONSENSITIVE.DESIGNATIVE.GEO.TIMEZONE",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.END_DATE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.DURATION",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.POPULATION",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.HEADLINE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.CONTENT_TYPE",
    ))
    cols += _take(_codes_matching(leaves, "ICE.NONSENSITIVE.DESIGNATIVE.REF.URL"))
    tables.append(("events", cols))

    # ── 9. publications — academic/reference ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.DOI",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.ISBN",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.ISSN",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.TITLE",
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.BOOK",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.ABSTRACT",
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.BODY",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.REF.CITATION",
        "ICE.NONSENSITIVE.DESIGNATIVE.REF.SOURCE",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.YEAR",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.LANGUAGE",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.SCORE",
    ))
    tables.append(("publications", cols))

    # ── 10. locations — geographic + measurements ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.GEO.REGION",
        "ICE.NONSENSITIVE.DESIGNATIVE.GEO.LOCATION",
        "ICE.NONSENSITIVE.DESIGNATIVE.GEO.COORDINATES",
        "ICE.NONSENSITIVE.DESIGNATIVE.GEO.CONTINENT",
        "ICE.NONSENSITIVE.DESIGNATIVE.GEO.DISTRICT",
        "ICE.NONSENSITIVE.DESIGNATIVE.GEO.LANDMARK",
        "ICE.NONSENSITIVE.DESIGNATIVE.GEO.AIRPORT",
        "ICE.NONSENSITIVE.DESIGNATIVE.GEO.PORT",
        "ICE.NONSENSITIVE.DESIGNATIVE.GEO.ADDRESS_LINE",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.LATITUDE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.LONGITUDE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.ELEVATION",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.DISTANCE",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.ISO_COUNTRY",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.FIPS",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.POSTAL",
    ))
    tables.append(("locations", cols))

    # ── 11. support_tickets — text-heavy + categorical ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.COMMENT",
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.INSTRUCTION",
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.REVIEW",
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.LOG_MESSAGE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.ERROR_MESSAGE",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.SEVERITY",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.SENTIMENT",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.LIFECYCLE",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.SENSITIVE.PID.CONTACT.SOCIAL_MEDIA",
        "ICE.SENSITIVE.PID.CONTACT.MESSENGER",
    ))
    cols += _take(_codes_matching(leaves, "ICE.METADATA.MODIFIED_BY"))
    tables.append(("support_tickets", cols))

    # ── 12. contracts — business sensitive + prescriptive ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.SENSITIVE.BUSINESS.CONTRACT_VALUE",
        "ICE.SENSITIVE.BUSINESS.TRADE_SECRET",
        "ICE.SENSITIVE.BUSINESS.PRICING_STRATEGY",
        "ICE.SENSITIVE.BUSINESS.LEGAL_HOLD",
        "ICE.SENSITIVE.BUSINESS.AUDIT_FINDING",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.PRESCRIPTIVE.SLA",
        "ICE.NONSENSITIVE.PRESCRIPTIVE.POLICY",
        "ICE.NONSENSITIVE.PRESCRIPTIVE.RETENTION_POLICY",
        "ICE.NONSENSITIVE.PRESCRIPTIVE.VALIDATION_RULE",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.ORG",
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.TIME",
    ))
    tables.append(("contracts", cols))

    # ── 13. campaigns — marketing + financial ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.SENSITIVE.BUSINESS.MARKET_INTEL",
        "ICE.SENSITIVE.BUSINESS.STRATEGIC_PLAN",
        "ICE.SENSITIVE.BUSINESS.FINANCIAL_FORECAST",
        "ICE.SENSITIVE.BUSINESS.CUSTOMER_LIST",
        "ICE.SENSITIVE.BUSINESS.INTERNAL_MEMO",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.PERCENTAGE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.RATIO",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.BOOLEAN_ENUM",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.MONTH",
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.DAY_OF_WEEK",
    ))
    tables.append(("campaigns", cols))

    # ── 14. user_sessions — technical + behavioral ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.SENSITIVE.TECHNICAL.SESSION_TOKEN",
        "ICE.SENSITIVE.TECHNICAL.COOKIE",
        "ICE.SENSITIVE.TECHNICAL.URL_PII",
        "ICE.SENSITIVE.TECHNICAL.API_KEY",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.BROWSER",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.DEVICE_TYPE",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.QUERY",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.DOMAIN_NAME",
        "ICE.NONSENSITIVE.DESIGNATIVE.REF.FILEPATH",
    ))
    cols += _take(_codes_matching(leaves, "ICE.METADATA.SOURCE_SYSTEM"))
    tables.append(("user_sessions", cols))

    # ── 15. sensor_data — IoT measurements ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.TEMPERATURE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.HUMIDITY",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.PRESSURE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.VOLTAGE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.CURRENT",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.FLOW_RATE",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.DECIBEL",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.LUMINOSITY",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.PH",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.CRON",
        "ICE.NONSENSITIVE.PRESCRIPTIVE.THRESHOLD",
        "ICE.NONSENSITIVE.PRESCRIPTIVE.SCHEDULE",
    ))
    cols += _take(_codes_matching(leaves, "ICE.METADATA.PARTITION"))
    tables.append(("sensor_data", cols))

    # ── 16. audit_log — metadata-heavy ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.METADATA.SCHEMA",
        "ICE.METADATA.COLUMN_NAME",
        "ICE.METADATA.DATA_TYPE",
        "ICE.METADATA.ROW_COUNT",
        "ICE.METADATA.CHECKSUM",
        "ICE.METADATA.LINEAGE",
        "ICE.METADATA.NULLABLE",
        "ICE.METADATA.RECORD_TYPE",
        "ICE.METADATA.IS_DELETED",
        "ICE.METADATA.DELETED_AT",
        "ICE.METADATA.VERSION",
        "ICE.METADATA.TTL",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.PERMISSION_LEVEL",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.APPROVAL_STATUS",
    ))
    tables.append(("audit_log", cols))

    # ── 17. content_library — creative/media ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.ARTWORK",
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.SONG",
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.MOVIE",
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.AWARD",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.CAPTION",
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.BIOGRAPHY",
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.TRANSLATION",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.FORMAT",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.MIME_TYPE",
    ))
    cols += _take(_codes_matching(leaves, "ICE.NONSENSITIVE.DESIGNATIVE.LABEL"))
    tables.append(("content_library", cols))

    # ── 18. vehicles — transportation/logistics ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.VIN",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.PLATE_NUMBER",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.FLIGHT_NUMBER",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.IATA",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.SPEED",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.ENERGY",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.POWER",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.MATERIAL",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.PRESCRIPTIVE.ROUTE",
    )) if "ICE.NONSENSITIVE.PRESCRIPTIVE.ROUTE" in leaves else []
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.ANGLE",
    ))
    tables.append(("vehicles", cols))

    # ── 19. research_data — scientific + statistical ──
    cols = []
    cols += _take(_codes_matching(leaves, "ICE.NONSENSITIVE.DESCRIPTIVE.STATISTICAL"))
    cols += _take(_codes_matching(leaves, "ICE.NONSENSITIVE.DESCRIPTIVE.EMBEDDING"))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC.PROBABILITY",
        "ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC.DECIMAL",
        "ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC.RANGE",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.SCIENTIFIC",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.CAS",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.DEFINITION",
    ))
    tables.append(("research_data", cols))

    # ── 20. config_registry — prescriptive-heavy ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.PRESCRIPTIVE.CONFIG",
        "ICE.NONSENSITIVE.PRESCRIPTIVE.PERMISSION",
        "ICE.NONSENSITIVE.PRESCRIPTIVE.COMMAND",
        "ICE.NONSENSITIVE.PRESCRIPTIVE.TEMPLATE",
        "ICE.NONSENSITIVE.PRESCRIPTIVE.FILTER",
        "ICE.NONSENSITIVE.PRESCRIPTIVE.MAPPING",
        "ICE.NONSENSITIVE.PRESCRIPTIVE.SCHEMA_DEF",
        "ICE.NONSENSITIVE.PRESCRIPTIVE.REGEX",
        "ICE.NONSENSITIVE.PRESCRIPTIVE.CRON_SPEC",
        "ICE.NONSENSITIVE.PRESCRIPTIVE.DEFAULT_VALUE",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.PROTOCOL",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.SEMANTIC_VERSION",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.PRESCRIPTIVE.API_SPEC",
        "ICE.NONSENSITIVE.PRESCRIPTIVE.FORMAT_SPEC",
    ))
    cols += _take(_codes_matching(leaves, "ICE.METADATA.ENCODING"))
    tables.append(("config_registry", cols))

    # ── 21. security_credentials — sensitive technical ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.SENSITIVE.TECHNICAL.PASSWORD_HASH",
        "ICE.SENSITIVE.TECHNICAL.DATABASE_CONN",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.SENSITIVE.PID.IDENTITY.GOVID.SSN",
        "ICE.SENSITIVE.PID.IDENTITY.BIOMETRIC.SIGNATURE",
        "ICE.SENSITIVE.PID.IDENTITY.BIOMETRIC.VOICEPRINT",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.HASH_ID",
        "ICE.NONSENSITIVE.DESIGNATIVE.BOOLEAN",
    ))
    tables.append(("security_credentials", cols))

    # ── 22. inventory — mixed measurements + codes ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.AREA",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.VOLUME",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.DENSITY",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.CONCENTRATION",
        "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.FREQUENCY",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.NAICS",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.DUNS",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.UNIT",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.RISK_LEVEL",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.CHANGELOG",
        "ICE.NONSENSITIVE.DESIGNATIVE.REF.VERSION",
    ))
    tables.append(("inventory", cols))

    # ── 23. food_and_health — domain crossover ──
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.FOOD",
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.DRUG",
        "ICE.NONSENSITIVE.DESIGNATIVE.NAME.DISEASE",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.RECIPE",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.PRESCRIPTIVE.FORMULA",
        "ICE.NONSENSITIVE.PRESCRIPTIVE.RULE",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.RELATIONSHIP",
        "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.TYPE",
    ))
    tables.append(("food_and_health", cols))

    # ── 24. dataset_7 (opaque table — codes + misc) ──
    # Deliberately vague table name to defeat schema-based classification
    cols = []
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.ID",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.ABBREV",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.ISO_LANGUAGE",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.CIK",
        "ICE.NONSENSITIVE.DESIGNATIVE.CODE.PHONE_CODE",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC.INTEGER",
        "ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC.ORDINAL",
        "ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC.BINARY",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.NONSENSITIVE.DESIGNATIVE.EMAIL_DOMAIN",
        "ICE.NONSENSITIVE.DESIGNATIVE.PHONE_FORMAT",
    ))
    tables.append(("dataset_7", cols))

    # ── 25. raw_import (catch-all with opaque character) ──
    # All remaining unassigned leaves go here — a "raw data dump" table
    remaining = sorted(leaves - assigned)
    if remaining:
        tables.append(("raw_import", _take(remaining)))

    # Validate: every leaf must be assigned exactly once
    assert assigned == leaves, f"Mismatch: {len(assigned)} assigned vs {len(leaves)} leaves"

    # Shuffle column order within each table to prevent position-based patterns
    for i, (name, cols) in enumerate(tables):
        shuffled = list(cols)
        rng.shuffle(shuffled)
        tables[i] = (name, shuffled)

    return tables


# ── Column naming ────────────────────────────────────────────────

_OPAQUE_PREFIXES = ["field_", "col_", "v_", "x_", "dim_", "f_", "attr_", "var_", "c_", "d_", "p_"]


def _code_to_column_name(code: str, cat_lookup: dict, rng: random.Random, opaque: bool = False) -> str:
    """Generate a column name from a category code.

    When opaque=True, generate a coded/abbreviated name that doesn't reveal
    the semantic meaning — forcing the classifier to rely on value patterns.
    """
    if opaque:
        prefix = rng.choice(_OPAQUE_PREFIXES)
        style = rng.randint(0, 3)
        if style == 0:
            return f"{prefix}{rng.randint(1, 999)}"
        elif style == 1:
            return f"{prefix}{''.join(rng.choices(string.ascii_lowercase, k=rng.randint(2, 4)))}"
        elif style == 2:
            # Single abbreviated token from the category code
            parts = code.split(".")
            token = parts[-1].lower()[:rng.randint(3, 5)]
            return f"{prefix}{token}"
        else:
            return f"{''.join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 6)))}{rng.randint(1, 99)}"

    cat = cat_lookup.get(code, {})
    label = cat.get("label", code.split(".")[-1])

    # Choose naming style
    style = rng.random()

    if style < 0.45:
        # snake_case from label
        name = label.lower().replace(" / ", "_").replace(" ", "_").replace("-", "_")
        name = "".join(c for c in name if c.isalnum() or c == "_")
    elif style < 0.65:
        # camelCase from label
        parts = label.lower().split()
        name = parts[0] + "".join(p.capitalize() for p in parts[1:]) if len(parts) > 1 else parts[0]
        name = "".join(c for c in name if c.isalnum())
    elif style < 0.80:
        # Use common_names variant if available
        common = cat.get("common_names", "")
        variants = [cn.strip() for cn in common.split(",") if cn.strip()] if common else []
        if variants:
            v = rng.choice(variants)
            name = v.lower().replace(" ", "_")
            name = "".join(c for c in name if c.isalnum() or c == "_")
        else:
            name = label.lower().replace(" ", "_")
            name = "".join(c for c in name if c.isalnum() or c == "_")
    else:
        # Use abbrev if available
        abbrev = cat.get("abbrev", "")
        if abbrev:
            name = abbrev.lower().replace(" ", "_")
            name = "".join(c for c in name if c.isalnum() or c == "_")
        else:
            name = label.lower().replace(" ", "_")
            name = "".join(c for c in name if c.isalnum() or c == "_")

    return name or label.lower().replace(" ", "_")


# ── Main ─────────────────────────────────────────────────────────


def main():
    _init_generators()

    with open(ONTOLOGY_PATH) as f:
        cats = json.load(f)

    cat_lookup = {c["code"]: c for c in cats}
    leaves = {c["code"] for c in cats if not any(c2["parent_code"] == c["code"] for c2 in cats)}

    rng = random.Random(SEED)
    tables = _build_table_assignments(cats, rng)

    # Check generator coverage
    missing_gens = []
    for code in sorted(leaves):
        if code not in _GENERATORS:
            missing_gens.append(code)
    if missing_gens:
        print(f"WARNING: {len(missing_gens)} leaves without generators (will use fallback):")
        for c in missing_gens[:10]:
            print(f"  {c}")
        if len(missing_gens) > 10:
            print(f"  ... and {len(missing_gens) - 10} more")

    # Decide which columns get opaque names (~25% overall)
    # Use a deterministic assignment so results are reproducible
    opaque_rng = random.Random(SEED + 1)
    opaque_codes: set[str] = set()
    for code in sorted(leaves):
        if opaque_rng.random() < 0.25:
            opaque_codes.add(code)

    # Generate tables
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    reference_labels = {}
    total_columns = 0
    opaque_count = 0

    for table_name, codes in tables:
        csv_path = TABLES_DIR / f"{table_name}.csv"
        col_names = []
        col_gens = []

        for code in codes:
            is_opaque = code in opaque_codes
            col_name = _code_to_column_name(code, cat_lookup, rng, opaque=is_opaque)
            # Ensure uniqueness within table
            base = col_name
            i = 2
            while col_name in [c for c, _ in col_gens]:
                col_name = f"{base}_{i}"
                i += 1

            gen = _GENERATORS.get(code, lambda rng: f"value_{rng.randint(1, 9999)}")
            col_names.append(col_name)
            col_gens.append((col_name, gen))
            reference_labels[f"{table_name}.{col_name}"] = code
            total_columns += 1
            if is_opaque:
                opaque_count += 1

        # Write CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(col_names)
            for _ in range(ROWS_PER_TABLE):
                row = [gen(rng) for _, gen in col_gens]
                writer.writerow(row)

    # Write reference labels
    with open(REFERENCE_LABELS_PATH, "w") as f:
        json.dump(reference_labels, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(tables)} tables with {total_columns} columns ({ROWS_PER_TABLE} rows each)")
    print(f"  Tables: {TABLES_DIR}/")
    print(f"  Reference labels: {REFERENCE_LABELS_PATH}")
    print(f"  Leaf categories: {len(leaves)}")
    print(f"  Generator coverage: {len(leaves) - len(missing_gens)}/{len(leaves)}")
    print(f"  Opaque column names: {opaque_count}/{total_columns} ({opaque_count*100//total_columns}%)")
    print(f"\nTable breakdown:")
    for name, codes in tables:
        print(f"  {name}: {len(codes)} columns")


if __name__ == "__main__":
    main()
