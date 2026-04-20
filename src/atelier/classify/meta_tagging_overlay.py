"""Meta-tagging code mapping and blended vocabulary builder.

Maps the enterprise meta-tagging annotation codes (numeric dot-notation like
1.1.1.1.1.1.1) to ICE.* codes from our hierarchical ontology. The mapping
is hand-maintained because the two taxonomies use different conceptual
groupings — automatic alignment would be unreliable.

The meta-tagging data files live in ``<repo>/build/meta-tagging/`` (UAT
snapshot, gitignored) or the legacy ``~/local/tmp/meta-tagging/`` —
resolved by ``meta_tagging_source.resolve_meta_tagging_mount``.  Never
committed to git.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Static code mapping ──────────────────────────────────────────
#
# meta-tagging numeric code → ICE.* leaf code
#
# Methodology: hand-aligned using annotations.csv ontology labels
# against expand_vocabulary.py category definitions.

META_TO_ICE: dict[str, str] = {
    # ── Financial: Payment Card Data (1.1.1.1.1.1.*) ──
    "1.1.1.1.1.1.1": "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.PAN",
    "1.1.1.1.1.1.2": "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.CVV",
    "1.1.1.1.1.1.3": "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.MAGSTRIPE",
    "1.1.1.1.1.1.4": "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.BIN",
    "1.1.1.1.1.1.5": "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.LAST4",
    "1.1.1.1.1.1.6": "ICE.SENSITIVE.PID.FINANCIAL.ACCOUNT.BAN",
    "1.1.1.1.1.1.7": "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.EXPIRY",
    "1.1.1.1.1.1.8": "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.LAST4",  # MASKPAN → LAST4
    "1.1.1.1.1.1.9": "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.CVV",   # DEBITPIN → CVV (closest)
    # ── Financial: Other Billing (1.1.1.1.1.2.*) ──
    "1.1.1.1.1.2.1": "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.BILLING.PAYPAL",
    "1.1.1.1.1.2.2": "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.BILLING.BILLING_ACCT",
    # ── Financial: Credit (1.1.1.1.2.*) ──
    "1.1.1.1.2.1": "ICE.SENSITIVE.PID.FINANCIAL.CREDIT.CREDIT_SCORE",
    "1.1.1.1.2.2": "ICE.SENSITIVE.PID.FINANCIAL.CREDIT.FRAUD_SCORE",
    "1.1.1.1.2.3": "ICE.SENSITIVE.PID.FINANCIAL.CREDIT.CREDIT_SCORE",  # DCSNCOLOR → CREDIT_SCORE (closest)
    # ── Demographic: Income (1.1.1.2.1.*) ──
    "1.1.1.2.1.1": "ICE.SENSITIVE.PID.FINANCIAL.INCOME.SALARY",
    "1.1.1.2.1.2": "ICE.SENSITIVE.PID.FINANCIAL.INCOME.STOCK_RSU",
    "1.1.1.2.1.3": "ICE.SENSITIVE.PID.FINANCIAL.INCOME.BONUS",
    "1.1.1.2.1.4": "ICE.SENSITIVE.PID.FINANCIAL.CREDIT.INSURANCE_ID",  # Benefits → Insurance
    # ── Demographic: Gender (1.1.1.2.2) ──
    "1.1.1.2.2": "ICE.SENSITIVE.PID.IDENTITY.GENDER_PII",
    # ── Demographic: Age (1.1.1.2.3.*) ──
    "1.1.1.2.3.1": "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.AGE",
    "1.1.1.2.3.2": "ICE.SENSITIVE.PID.IDENTITY.DOB",
    "1.1.1.2.3.3": "ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.YEAR",  # Birth Year
    "1.1.1.2.3.4": "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.AGE",  # Age Group
    "1.1.1.2.3.4.1": "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.AGE",  # Under 18
    "1.1.1.2.3.4.2": "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.AGE",  # Under 13
    "1.1.1.2.3.4.3": "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.AGE",  # Age Differentiating
    "1.1.1.2.3.5": "ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.MONTH",     # Birth Month
    "1.1.1.2.3.6": "ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC.INTEGER",     # Birth Day of Month
    # ── Demographic: Race/Ethnicity (1.1.1.2.4) ──
    "1.1.1.2.4": "ICE.SENSITIVE.PID.IDENTITY.ETHNICITY",
    # ── Demographic: Occupation (1.1.1.2.5.*) ──
    "1.1.1.2.5.1": "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.ROLE",   # Job Title
    "1.1.1.2.5.2": "ICE.SENSITIVE.BUSINESS.PERFORMANCE_REVIEW",
    "1.1.1.2.5.3": "ICE.SENSITIVE.BUSINESS.EMPLOYEE_RECORD",
    # ── Demographic: Education (1.1.1.2.6) ──
    "1.1.1.2.6": "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.EDUCATION_LEVEL",
    # ── Demographic: Background (1.1.1.2.7) ──
    "1.1.1.2.7": "ICE.SENSITIVE.PID.IDENTITY.NATIONALITY",  # Background → closest
    # ── Psychographic (1.1.1.3.*) ──
    "1.1.1.3.1": "ICE.SENSITIVE.PID.IDENTITY.ETHNICITY",  # Religion → ETHNICITY (sensitive personal)
    "1.1.1.3.4.1": "ICE.SENSITIVE.PID.IDENTITY.MARITAL_STATUS",
    # ── Geographic (1.1.1.4.*) ──
    "1.1.1.4.1": "ICE.SENSITIVE.PID.CONTACT.ADDRESS",           # Full address
    "1.1.1.4.1.1": "ICE.SENSITIVE.PID.CONTACT.ADDRESS",         # Billing address
    "1.1.1.4.1.2": "ICE.SENSITIVE.PID.CONTACT.ADDRESS",         # Shipping address
    "1.1.1.4.2.1": "ICE.NONSENSITIVE.DESIGNATIVE.GEO.ADDRESS_LINE",  # Street
    "1.1.1.4.2.1.1": "ICE.NONSENSITIVE.DESIGNATIVE.GEO.ADDRESS_LINE",
    "1.1.1.4.2.1.2": "ICE.NONSENSITIVE.DESIGNATIVE.GEO.ADDRESS_LINE",
    "1.1.1.4.2.2": "ICE.NONSENSITIVE.DESIGNATIVE.GEO.CITY",
    "1.1.1.4.2.2.1": "ICE.NONSENSITIVE.DESIGNATIVE.GEO.CITY",
    "1.1.1.4.2.2.2": "ICE.NONSENSITIVE.DESIGNATIVE.GEO.CITY",
    "1.1.1.4.2.3": "ICE.NONSENSITIVE.DESIGNATIVE.CODE.POSTAL",
    "1.1.1.4.2.3.1": "ICE.NONSENSITIVE.DESIGNATIVE.CODE.POSTAL",
    "1.1.1.4.2.3.2": "ICE.NONSENSITIVE.DESIGNATIVE.CODE.POSTAL",
    "1.1.1.4.2.4": "ICE.NONSENSITIVE.DESIGNATIVE.GEO.REGION",
    "1.1.1.4.2.4.1": "ICE.NONSENSITIVE.DESIGNATIVE.GEO.REGION",
    "1.1.1.4.2.4.2": "ICE.NONSENSITIVE.DESIGNATIVE.GEO.REGION",
    "1.1.1.4.2.5": "ICE.NONSENSITIVE.DESIGNATIVE.GEO.COUNTRY",
    "1.1.1.4.2.5.1": "ICE.NONSENSITIVE.DESIGNATIVE.GEO.COUNTRY",
    "1.1.1.4.2.5.2": "ICE.NONSENSITIVE.DESIGNATIVE.GEO.COUNTRY",
    "1.1.1.4.3.1": "ICE.NONSENSITIVE.DESIGNATIVE.GEO.LOCATION",  # Coarse Location
    "1.1.1.4.3.2": "ICE.NONSENSITIVE.DESIGNATIVE.GEO.COORDINATES",  # Precise Location
    "1.1.1.4.4": "ICE.SENSITIVE.TECHNICAL.IPADDR",
    "1.1.1.4.6": "ICE.NONSENSITIVE.DESIGNATIVE.CODE.ID",  # Tax Jurisdiction ID
    # ── Health (1.1.1.6.*) ──
    "1.1.1.6.1": "ICE.SENSITIVE.PID.HEALTH.DIAGNOSIS",        # Mental Condition
    "1.1.1.6.2": "ICE.SENSITIVE.PID.HEALTH.VITAL_SIGN",       # Physical Condition
    "1.1.1.6.3": "ICE.SENSITIVE.PID.HEALTH.LAB_RESULT",       # Genetic Data
    "1.1.1.6.4": "ICE.SENSITIVE.PID.IDENTITY.BIOMETRIC.FINGERPRINT",  # Biometric
    # ── Product Usage (1.1.1.7.*) ──
    "1.1.1.7.1": "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.LOG_MESSAGE",  # Login Events
    "1.1.1.7.2": "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.ERROR_MESSAGE",  # Crash Reports
    "1.1.1.7.3.1": "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.ERROR_MESSAGE",  # Gen Security Flaw
    "1.1.1.7.3.2": "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.ERROR_MESSAGE",  # Critical Src Code
    "1.1.1.7.3.3": "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.ERROR_MESSAGE",  # 0-Day
    "1.1.1.7.4.1": "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.COMMENT",  # User Text
    "1.1.1.7.4.1.1": "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.COMMENT",  # Forum Posts
    "1.1.1.7.4.1.2": "ICE.NONSENSITIVE.DESIGNATIVE.REF.URL",     # Personal URL
    "1.1.1.7.4.1.3": "ICE.NONSENSITIVE.DESIGNATIVE.REF.FILEPATH",  # File/Folder Name
    "1.1.1.7.4.1.4": "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.QUERY",   # Search Query
    "1.1.1.7.4.2.1": "ICE.SENSITIVE.PID.IDENTITY.BIOMETRIC.PHOTO",  # Photo
    "1.1.1.7.4.2.2": "ICE.SENSITIVE.PID.IDENTITY.BIOMETRIC.PHOTO",  # Video → Photo (media)
    "1.1.1.7.5": "ICE.NONSENSITIVE.PRESCRIPTIVE.CONFIG",         # Config Settings
    "1.1.1.7.6.1": "ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.CPU_USAGE",  # Device Metrics
    "1.1.1.7.8": "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.PLATFORM",  # Software Version
    "1.1.1.7.10": "ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.BROWSER",  # User Agent
    # ── Authentication (1.1.1.8.*) ──
    "1.1.1.8.1": "ICE.SENSITIVE.TECHNICAL.PASSWORD_HASH",  # Password
    "1.1.1.8.2": "ICE.SENSITIVE.TECHNICAL.PASSWORD_HASH",  # PIN
    "1.1.1.8.3.1": "ICE.SENSITIVE.TECHNICAL.PASSWORD_HASH",  # Security Question
    "1.1.1.8.3.2": "ICE.SENSITIVE.TECHNICAL.PASSWORD_HASH",  # Security Answer
    "1.1.1.8.4": "ICE.SENSITIVE.PID.IDENTITY.BIOMETRIC.SIGNATURE",  # E-Signature
    "1.1.1.8.5": "ICE.SENSITIVE.TECHNICAL.SSH_KEY",          # Key Material
    "1.1.1.8.6": "ICE.NONSENSITIVE.DESIGNATIVE.CODE.HASH_ID",  # Key Digest
    # ── Contact (1.1.1.9.*) ──
    "1.1.1.9.1": "ICE.SENSITIVE.PID.IDENTITY.NAME.FULLNAME",
    "1.1.1.9.2.1": "ICE.SENSITIVE.PID.IDENTITY.NAME.FIRST_NAME",
    "1.1.1.9.2.2": "ICE.SENSITIVE.PID.IDENTITY.NAME.MIDDLE_NAME",
    "1.1.1.9.2.3": "ICE.SENSITIVE.PID.IDENTITY.NAME.LAST_NAME",
    "1.1.1.9.2.4": "ICE.SENSITIVE.PID.IDENTITY.NAME.FULLNAME",  # Nickname
    "1.1.1.9.2.5": "ICE.SENSITIVE.PID.IDENTITY.NAME.MAIDEN_NAME",  # Other Name
    "1.1.1.9.2.6": "ICE.SENSITIVE.PID.IDENTITY.NAME.TITLE_PREFIX",  # Title
    "1.1.1.9.3.1": "ICE.SENSITIVE.PID.CONTACT.EMAIL",
    "1.1.1.9.3.2": "ICE.SENSITIVE.PID.CONTACT.SOCIAL_MEDIA",
    "1.1.1.9.4.1": "ICE.SENSITIVE.PID.CONTACT.PHONE",  # Home
    "1.1.1.9.4.2": "ICE.SENSITIVE.PID.CONTACT.PHONE",  # Office
    "1.1.1.9.4.3": "ICE.SENSITIVE.PID.CONTACT.PHONE",  # Mobile
    "1.1.1.9.4.4": "ICE.SENSITIVE.PID.CONTACT.PHONE",  # Other
    "1.1.1.9.4.5": "ICE.SENSITIVE.PID.CONTACT.FAX",
    # ── Identity: Government (1.1.2.1.*) ──
    "1.1.2.1.1.1": "ICE.SENSITIVE.PID.IDENTITY.GOVID.PASSPORT",
    "1.1.2.1.1.2": "ICE.SENSITIVE.PID.IDENTITY.GOVID.PASSPORT",  # Travel Permit
    "1.1.2.1.1.3": "ICE.SENSITIVE.PID.IDENTITY.GOVID.WORK_PERMIT",
    "1.1.2.1.2.1": "ICE.SENSITIVE.PID.IDENTITY.GOVID.TAX_ID",  # Income Tax ID
    "1.1.2.1.2.1.3": "ICE.SENSITIVE.PID.IDENTITY.GOVID.SSN",
    "1.1.2.1.3": "ICE.SENSITIVE.PID.IDENTITY.GOVID.DLN",
    # ── Identity: Platform (1.1.2.2.*) ──
    "1.1.2.2.1.1": "ICE.NONSENSITIVE.DESIGNATIVE.CODE.ID",  # Person ID
    "1.1.2.2.1.2": "ICE.SENSITIVE.PID.IDENTITY.PLATFORM_ID.USERNAME",  # Platform ID
    "1.1.2.2.1.3": "ICE.NONSENSITIVE.DESIGNATIVE.CODE.ID",  # App ID
    "1.1.2.2.1.3.5": "ICE.SENSITIVE.PID.IDENTITY.PLATFORM_ID.ADVERTISER_ID",
    "1.1.2.2.1.4": "ICE.SENSITIVE.PID.IDENTITY.PLATFORM_ID.EMPLOYEE_ID",
    "1.1.2.2.1.5": "ICE.SENSITIVE.PID.IDENTITY.PLATFORM_ID.USERNAME",  # User ID
    # ── Identity: Device (1.1.2.3.*) ──
    "1.1.2.3.1": "ICE.SENSITIVE.TECHNICAL.DEVID",  # UDID
    "1.1.2.3.2": "ICE.SENSITIVE.TECHNICAL.DEVID",  # ICCID
    "1.1.2.3.3": "ICE.SENSITIVE.TECHNICAL.DEVID",  # IMEI
    "1.1.2.3.4": "ICE.SENSITIVE.TECHNICAL.DEVID",  # SEID
    "1.1.2.3.5": "ICE.SENSITIVE.TECHNICAL.DEVID",  # GUID
    "1.1.2.3.6": "ICE.NONSENSITIVE.DESIGNATIVE.NAME.PRODUCT",  # Device Name
    "1.1.2.3.7": "ICE.SENSITIVE.TECHNICAL.DEVID",  # MEI/MEID
    "1.1.2.4.1": "ICE.SENSITIVE.TECHNICAL.DEVID",  # Device Serial Number
    "1.1.2.4.2": "ICE.SENSITIVE.TECHNICAL.DEVID",  # MAC Address
    # ── Transaction (1.2.*) ──
    "1.2.1": "ICE.METADATA.RECID",  # Transaction ID
    "1.2.1.1": "ICE.METADATA.RECID",  # Subscription ID
    "1.2.1.2": "ICE.METADATA.RECID",  # Web Order ID
    "1.2.2": "ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.DATE",  # Transaction Date
    "1.2.4": "ICE.METADATA.TIMESTAMP",  # Transaction Timestamp
    "1.2.5": "ICE.SENSITIVE.TECHNICAL.SESSION_TOKEN",  # Session ID
    "1.2.6.1": "ICE.NONSENSITIVE.DESIGNATIVE.CODE.ID",  # Digital Asset ID
    "1.2.6.2": "ICE.NONSENSITIVE.DESIGNATIVE.CODE.ID",  # Bundle ID
    "1.2.6.3": "ICE.NONSENSITIVE.DESIGNATIVE.CODE.ID",  # Org ID
    # ── System Data (1.3.*) ──
    "1.3.1.1": "ICE.SENSITIVE.TECHNICAL.IPADDR",  # Trusted IPs
    "1.3.1.2": "ICE.SENSITIVE.PID.IDENTITY.PLATFORM_ID.EMPLOYEE_ID",  # User IDs
    "1.3.1.4": "ICE.NONSENSITIVE.PRESCRIPTIVE.PERMISSION",  # Permissions
    "1.3.1.5": "ICE.SENSITIVE.TECHNICAL.SESSION_TOKEN",  # Sessions
    "1.3.1.6.1": "ICE.NONSENSITIVE.PRESCRIPTIVE.REGEX",  # Regex validation
    "1.3.2.1.1": "ICE.NONSENSITIVE.DESIGNATIVE.REF.URL",  # System URLs
    "1.3.2.1.3": "ICE.SENSITIVE.TECHNICAL.IPADDR",  # IPs or Ports
    "1.3.2.1.4": "ICE.SENSITIVE.TECHNICAL.API_KEY",  # Secrets
    "1.3.2.2.1": "ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.BODY",  # Serialized Objects
    "1.3.2.2.3": "ICE.NONSENSITIVE.PRESCRIPTIVE.COMMAND",  # Shell Commands
    "1.3.2.2.4": "ICE.NONSENSITIVE.PRESCRIPTIVE.COMMAND",  # Source Code
    "1.3.2.2.6": "ICE.NONSENSITIVE.DESIGNATIVE.REF.FILEPATH",  # File Paths
    # ── Business (1.4.*) ──
    "1.4.1.1.1": "ICE.SENSITIVE.BUSINESS.FINANCIAL_FORECAST",  # Financial Doc
    "1.4.1.1.5": "ICE.SENSITIVE.BUSINESS.FINANCIAL_FORECAST",  # Sales Doc
    "1.4.1.1.6": "ICE.SENSITIVE.BUSINESS.EMPLOYEE_RECORD",  # HR Doc
    "1.4.1.1.7": "ICE.SENSITIVE.BUSINESS.AUDIT_FINDING",  # Incident Doc
    "1.4.2": "ICE.SENSITIVE.BUSINESS.TRADE_SECRET",
}


def translate_reference_labels(
    reference_labels: dict[str, str],
    *,
    code_map: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Translate numeric meta-tagging codes to ICE.* codes.

    Returns:
        (translated, unmapped) where unmapped maps column→original_code
        for any codes not in the mapping.
    """
    if code_map is None:
        code_map = META_TO_ICE

    translated: dict[str, str] = {}
    unmapped: dict[str, str] = {}

    for col_name, meta_code in reference_labels.items():
        ice_code = code_map.get(meta_code)
        if ice_code:
            translated[col_name] = ice_code
        else:
            unmapped[col_name] = meta_code

    if unmapped:
        unique = set(unmapped.values())
        logger.info(
            "Translated %d/%d reference codes (%d unmapped codes: %s)",
            len(translated), len(reference_labels), len(unique),
            ", ".join(sorted(unique)[:10]),
        )
    else:
        logger.info("Translated all %d reference codes", len(translated))

    return translated, unmapped


def build_blended_vocabulary(
    base_category_set,
    meta_tagging_dir: Path,
    *,
    code_map: dict[str, str] | None = None,
):
    """Build a blended vocabulary from base + meta-tagging annotations.

    The base vocabulary provides the ICE.* hierarchy. Meta-tagging
    annotations that map to existing ICE codes get their labels enriched.
    Codes without mappings are logged but not added (the base ontology
    is authoritative).

    Returns a HierarchicalCategorySet (same as base, since all meta-tagging
    codes map into the existing hierarchy).
    """
    if code_map is None:
        code_map = META_TO_ICE

    meta_tagging_dir = Path(meta_tagging_dir)
    ann_path = meta_tagging_dir / "annotations.csv"

    if not ann_path.exists():
        logger.warning("annotations.csv not found at %s, returning base vocabulary", ann_path)
        return base_category_set

    from atelier.classify.real_data_loader import load_annotations_csv
    records = load_annotations_csv(ann_path)

    # Build meta_code → record lookup
    meta_lookup: dict[str, dict] = {}
    for rec in records:
        meta_code = rec.get("id", "").strip()
        if meta_code:
            meta_lookup[meta_code] = rec

    # Report mapping coverage
    mapped = sum(1 for code in meta_lookup if code in code_map)
    total = len(meta_lookup)
    logger.info(
        "Meta-tagging overlay: %d/%d annotation codes mapped to ICE (%.0f%%)",
        mapped, total, (mapped / total * 100) if total else 0,
    )

    # The base vocabulary already contains all the target ICE codes,
    # so we return it unchanged. The mapping is used at classification
    # time via translate_reference_labels().
    return base_category_set


def mapping_coverage_report(
    meta_tagging_dir: Path,
    *,
    code_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Report how well the code mapping covers meta-tagging data files.

    Returns:
      total_annotations: number of codes in annotations.csv
      mapped: number with ICE.* mapping
      unmapped: list of codes without mapping
      data_file_coverage: {filename: {total_cols, mapped_cols, unmapped_cols}}
    """
    if code_map is None:
        code_map = META_TO_ICE

    meta_tagging_dir = Path(meta_tagging_dir)

    # Coverage from annotations
    from atelier.classify.real_data_loader import load_annotations_csv
    ann_path = meta_tagging_dir / "annotations.csv"
    records = load_annotations_csv(ann_path) if ann_path.exists() else []

    all_codes = {r.get("id", "").strip() for r in records if r.get("id", "").strip()}
    mapped_codes = {c for c in all_codes if c in code_map}
    unmapped_codes = sorted(all_codes - mapped_codes)

    # Coverage from data files
    from atelier.classify.real_data_loader import load_real_samples
    samples = load_real_samples(meta_tagging_dir, sample_size=5)

    file_coverage: dict[str, dict] = {}
    for table in samples:
        total = len(table.columns)
        mapped = sum(1 for c in table.columns if c.reference_code in code_map)
        file_coverage[table.name] = {
            "total_cols": total,
            "mapped_cols": mapped,
            "unmapped_cols": total - mapped,
        }

    return {
        "total_annotations": len(all_codes),
        "mapped": len(mapped_codes),
        "unmapped": unmapped_codes,
        "data_file_coverage": file_coverage,
    }
