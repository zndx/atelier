# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Shared value generator library for synthetic data generation.

Consolidates all data lists and generator functions used by both
synth.py (CatBoost/SVM training) and generate_sample_source.py
(OOTB sample data). Each generator takes (rng: random.Random) → str.

This is the single source of truth for value generators — both
the pipeline's synth module and the OOTB scripts import from here.
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable
from datetime import datetime, timedelta

# ── Data lists ─────────────────────────────────────────────────

FIRST_NAMES = [
    "James", "John", "Robert", "Michael", "William", "David", "Richard",
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Susan", "Jessica",
    "Sarah", "Emily", "Amy", "Anna", "Emma", "Sophia", "Olivia", "Liam",
    "Noah", "Ethan", "Mason", "Logan", "Aiden", "Oliver", "Lucas", "Harper",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas",
    "Jackson", "White", "Harris", "Martin", "Thompson", "Moore", "Lee",
    "Clark", "Lewis", "Robinson", "Walker", "Young", "King", "Wright",
]
CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Seattle",
    "Denver", "Boston", "Austin", "Portland", "Miami", "Dallas", "Atlanta",
    "San Francisco", "Toronto", "London", "Paris", "Berlin", "Tokyo", "Sydney",
]
COUNTRIES = [
    "United States", "Canada", "United Kingdom", "Germany", "France", "Japan",
    "Australia", "Brazil", "India", "Mexico", "Italy", "Spain", "Netherlands",
    "South Korea", "Sweden", "Norway", "Switzerland", "Singapore", "Israel",
]
COUNTRY_CODES = ["US", "CA", "GB", "DE", "FR", "JP", "AU", "BR", "IN", "MX"]
DOMAINS = ["gmail.com", "outlook.com", "yahoo.com", "company.com", "acme.org", "example.com"]
STATES = ["CA", "TX", "NY", "FL", "WA", "IL", "CO", "MA", "GA", "PA", "OH", "NC"]
STREETS = ["Main St", "Oak Ave", "Maple Dr", "Cedar Ln", "Pine Rd", "Elm St", "Park Blvd"]
ORGS = [
    "Acme Corp", "TechCo Industries", "Global Solutions", "DataCraft Inc",
    "Quantum Dynamics", "Pacific Research Group", "Alpine Systems",
    "Skyline Partners", "Beacon Analytics", "Vertex Technologies",
    "Meridian Health", "Catalyst Ventures", "Pioneer Software",
]
PRODUCTS = [
    "Widget Pro", "DataSync 3000", "CloudGuard", "SmartHub X1",
    "FlexPay Platform", "NanoCore SDK", "AuditTrail Plus",
    "Sentinel Monitor", "StreamLine ERP", "VaultKeeper",
]
STATUSES = ["active", "inactive", "pending", "completed", "failed", "cancelled", "approved", "rejected"]
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "INR", "BRL"]
LANGUAGES = ["English", "Spanish", "French", "German", "Japanese", "Chinese", "Portuguese", "Korean", "Arabic", "Hindi"]
LANG_CODES = ["en", "es", "fr", "de", "ja", "zh", "pt", "ko", "ar", "hi"]
COLORS = ["Red", "Blue", "Green", "Black", "White", "Yellow", "Purple", "Orange", "Gray", "Brown"]
SIZES = ["XS", "S", "M", "L", "XL", "XXL"]
PLATFORMS = ["Windows", "macOS", "Linux", "iOS", "Android", "Chrome OS"]
BROWSERS = ["Chrome", "Firefox", "Safari", "Edge", "Opera", "Brave"]
DEVICE_TYPES = ["Desktop", "Mobile", "Tablet", "IoT", "Wearable", "Smart TV"]
DEPARTMENTS = ["Engineering", "Sales", "Marketing", "Finance", "HR", "Operations", "Legal", "Support"]
INDUSTRIES = ["Technology", "Healthcare", "Finance", "Retail", "Manufacturing", "Energy", "Education"]
ROLES = ["Software Engineer", "Product Manager", "Data Analyst", "Designer", "VP Engineering", "CTO", "Sales Rep"]
SEVERITIES = ["Critical", "High", "Medium", "Low", "Info"]
SENTIMENTS = ["Positive", "Negative", "Neutral", "Mixed"]
CHANNELS = ["Web", "Mobile", "Email", "Phone", "In-Store", "Social Media", "Chat"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "Wire Transfer", "ACH", "PayPal", "Cash", "Check"]
SHIPPING_METHODS = ["Standard", "Express", "Overnight", "Freight", "Same Day", "Economy"]
MIME_TYPES = ["application/json", "text/csv", "text/html", "image/png", "application/pdf", "application/xml"]
PROTOCOLS = ["HTTP", "HTTPS", "FTP", "SSH", "SMTP", "gRPC", "WebSocket", "MQTT"]


# ── Helper ─────────────────────────────────────────────────────

def _rng_date(rng: random.Random, start_year: int = 2018, end_year: int = 2026) -> datetime:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 28)
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


# ── Generator functions ────────────────────────────────────────

def gen_email(rng: random.Random) -> str:
    first = rng.choice(FIRST_NAMES).lower()
    last = rng.choice(LAST_NAMES).lower()
    sep = rng.choice([".", "_", ""])
    domain = rng.choice(DOMAINS)
    suffix = "" if rng.random() > 0.3 else str(rng.randint(1, 99))
    return f"{first}{sep}{last}{suffix}@{domain}"


def gen_phone(rng: random.Random) -> str:
    area = rng.randint(200, 999)
    mid = rng.randint(200, 999)
    last = rng.randint(1000, 9999)
    fmt = rng.choice(["({}) {}-{}", "{}-{}-{}", "{}.{}.{}", "+1{}{}{}", "{} {} {}"])
    return fmt.format(area, mid, last)


def gen_address(rng: random.Random) -> str:
    num = rng.randint(100, 9999)
    street = rng.choice(STREETS)
    city = rng.choice(CITIES)
    state = rng.choice(STATES)
    zipcode = rng.randint(10000, 99999)
    return f"{num} {street}, {city}, {state} {zipcode}"


def gen_fullname(rng: random.Random) -> str:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    if rng.random() < 0.2:
        middle = rng.choice(string.ascii_uppercase)
        return f"{first} {middle}. {last}"
    return f"{first} {last}"


def gen_firstname(rng: random.Random) -> str:
    return rng.choice(FIRST_NAMES)


def gen_lastname(rng: random.Random) -> str:
    return rng.choice(LAST_NAMES)


def gen_dob(rng: random.Random) -> str:
    year = rng.randint(1940, 2010)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    fmt = rng.choice(["{:04d}-{:02d}-{:02d}", "{:02d}/{:02d}/{:04d}", "{:02d}-{:02d}-{:04d}"])
    if "{:04d}" == fmt[:5]:
        return fmt.format(year, month, day)
    return fmt.format(month, day, year)


def gen_ssn(rng: random.Random) -> str:
    a = rng.randint(100, 899)
    b = rng.randint(10, 99)
    c = rng.randint(1000, 9999)
    if rng.random() < 0.3:
        return f"XXX-XX-{c}"
    return f"{a}-{b}-{c}"


def gen_pan(rng: random.Random) -> str:
    prefix = rng.choice(["4", "5", "37", "6011"])
    remaining = 16 - len(prefix)
    digits = prefix + "".join(str(rng.randint(0, 9)) for _ in range(remaining))
    if rng.random() < 0.3:
        return f"XXXX-XXXX-XXXX-{digits[-4:]}"
    return f"{digits[:4]}-{digits[4:8]}-{digits[8:12]}-{digits[12:]}"


def gen_ban(rng: random.Random) -> str:
    length = rng.choice([8, 10, 12])
    return "".join(str(rng.randint(0, 9)) for _ in range(length))


def gen_amount(rng: random.Random) -> str:
    val = round(rng.uniform(0.50, 99999.99), 2)
    if rng.random() < 0.5:
        return f"${val:,.2f}"
    return f"{val:.2f}"


def gen_ipv4(rng: random.Random) -> str:
    return ".".join(str(rng.randint(1, 254)) for _ in range(4))


def gen_uuid(rng: random.Random) -> str:
    h = "0123456789abcdef"
    parts = [
        "".join(rng.choices(h, k=8)),
        "".join(rng.choices(h, k=4)),
        "4" + "".join(rng.choices(h, k=3)),
        rng.choice("89ab") + "".join(rng.choices(h, k=3)),
        "".join(rng.choices(h, k=12)),
    ]
    return "-".join(parts)


def gen_url(rng: random.Random) -> str:
    domain = rng.choice(["example.com", "acme.org", "data.io", "api.service.com"])
    path = "/".join(rng.choices(["users", "api", "v2", "data", "docs", "status"], k=rng.randint(1, 3)))
    return f"https://{domain}/{path}"


def gen_timestamp(rng: random.Random) -> str:
    year = rng.randint(2018, 2026)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    second = rng.randint(0, 59)
    if rng.random() < 0.5:
        return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}Z"
    return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"


def gen_record_id(rng: random.Random) -> str:
    if rng.random() < 0.6:
        return str(rng.randint(1, 9999999))
    prefix = rng.choice(["REC", "ID", "TXN", "ORD", "USR"])
    return f"{prefix}-{rng.randint(100000, 999999)}"


def gen_status(rng: random.Random) -> str:
    return rng.choice(STATUSES)


def gen_generic_string(rng: random.Random) -> str:
    templates = [
        lambda: "".join(rng.choices(string.ascii_letters, k=rng.randint(5, 20))),
        lambda: f"item_{rng.randint(1, 10000)}",
        lambda: rng.choice(["red", "blue", "green", "yellow", "black", "white"]),
        lambda: f"category_{rng.choice(string.ascii_uppercase)}",
        lambda: str(rng.randint(1, 100)),
    ]
    return rng.choice(templates)()


def gen_internal_code(rng: random.Random) -> str:
    prefixes = ["DEPT", "DIV", "SYS", "MOD", "GRP", "UNIT"]
    return f"{rng.choice(prefixes)}-{rng.randint(100, 999)}"


def gen_date(rng: random.Random) -> str:
    return _rng_date(rng).strftime("%Y-%m-%d")


def gen_datetime(rng: random.Random) -> str:
    return _rng_date(rng).strftime("%Y-%m-%dT%H:%M:%SZ")


def gen_time(rng: random.Random) -> str:
    return f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"


def gen_year(rng: random.Random) -> str:
    return str(rng.randint(1990, 2026))


def gen_month(rng: random.Random) -> str:
    return rng.choice(["January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December"])


def gen_country(rng: random.Random) -> str:
    return rng.choice(COUNTRIES)


def gen_country_code(rng: random.Random) -> str:
    return rng.choice(COUNTRY_CODES)


def gen_city(rng: random.Random) -> str:
    return rng.choice(CITIES)


def gen_region(rng: random.Random) -> str:
    return rng.choice(STATES)


def gen_org(rng: random.Random) -> str:
    return rng.choice(ORGS)


def gen_product(rng: random.Random) -> str:
    return rng.choice(PRODUCTS)


def gen_integer(rng: random.Random) -> str:
    return str(rng.randint(0, 10000))


def gen_decimal(rng: random.Random) -> str:
    return f"{round(rng.uniform(-1000, 10000), 4)}"


def gen_percentage(rng: random.Random) -> str:
    return f"{round(rng.uniform(0, 100), 2)}%"


def gen_score(rng: random.Random) -> str:
    return f"{round(rng.uniform(0, 10), 2)}"


def gen_count(rng: random.Random) -> str:
    return str(rng.randint(0, 100000))


def gen_price(rng: random.Random) -> str:
    return f"{round(rng.uniform(0.99, 9999.99), 2)}"


def gen_boolean(rng: random.Random) -> str:
    return rng.choice(["true", "false", "yes", "no", "1", "0"])


def gen_currency_code(rng: random.Random) -> str:
    return rng.choice(CURRENCIES)


def gen_language(rng: random.Random) -> str:
    return rng.choice(LANGUAGES)


def gen_lang_code(rng: random.Random) -> str:
    return rng.choice(LANG_CODES)


def gen_color(rng: random.Random) -> str:
    return rng.choice(COLORS)


def gen_size(rng: random.Random) -> str:
    return rng.choice(SIZES)


def gen_platform(rng: random.Random) -> str:
    return rng.choice(PLATFORMS)


def gen_browser(rng: random.Random) -> str:
    return rng.choice(BROWSERS)


def gen_device_type(rng: random.Random) -> str:
    return rng.choice(DEVICE_TYPES)


def gen_department(rng: random.Random) -> str:
    return rng.choice(DEPARTMENTS)


def gen_industry(rng: random.Random) -> str:
    return rng.choice(INDUSTRIES)


def gen_role(rng: random.Random) -> str:
    return rng.choice(ROLES)


def gen_severity(rng: random.Random) -> str:
    return rng.choice(SEVERITIES)


def gen_sentiment(rng: random.Random) -> str:
    return rng.choice(SENTIMENTS)


def gen_channel(rng: random.Random) -> str:
    return rng.choice(CHANNELS)


def gen_payment_method(rng: random.Random) -> str:
    return rng.choice(PAYMENT_METHODS)


def gen_shipping(rng: random.Random) -> str:
    return rng.choice(SHIPPING_METHODS)


def gen_mime(rng: random.Random) -> str:
    return rng.choice(MIME_TYPES)


def gen_protocol(rng: random.Random) -> str:
    return rng.choice(PROTOCOLS)


def gen_latitude(rng: random.Random) -> str:
    return f"{round(rng.uniform(-90, 90), 6)}"


def gen_longitude(rng: random.Random) -> str:
    return f"{round(rng.uniform(-180, 180), 6)}"


def gen_elevation(rng: random.Random) -> str:
    return f"{rng.randint(-100, 8848)}"


def gen_temperature(rng: random.Random) -> str:
    return f"{round(rng.uniform(-40, 50), 1)}"


def gen_weight(rng: random.Random) -> str:
    return f"{round(rng.uniform(0.1, 1000), 2)}"


def gen_distance(rng: random.Random) -> str:
    return f"{round(rng.uniform(0.1, 10000), 1)}"


def gen_duration(rng: random.Random) -> str:
    return f"{rng.randint(1, 86400)}"


def gen_age(rng: random.Random) -> str:
    return str(rng.randint(0, 120))


def gen_recid(rng: random.Random) -> str:
    return str(rng.randint(100000, 999999))


def gen_version(rng: random.Random) -> str:
    return f"{rng.randint(0, 5)}.{rng.randint(0, 20)}.{rng.randint(0, 99)}"


def gen_hash(rng: random.Random) -> str:
    return "".join(rng.choices("0123456789abcdef", k=64))


def gen_fax(rng: random.Random) -> str:
    return f"+1-{rng.randint(200, 999)}-{rng.randint(200, 999)}-{rng.randint(1000, 9999)}"


def gen_social(rng: random.Random) -> str:
    return f"@{rng.choice(FIRST_NAMES).lower()}{rng.choice(LAST_NAMES).lower()}{rng.randint(1, 999)}"


def gen_cvv(rng: random.Random) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(rng.choice([3, 4])))


def gen_expiry(rng: random.Random) -> str:
    return f"{rng.randint(1, 12):02d}/{rng.randint(24, 30)}"


def gen_salary(rng: random.Random) -> str:
    return f"${rng.randint(30000, 250000):,}"


def gen_hostname(rng: random.Random) -> str:
    return f"{rng.choice(['web', 'api', 'db', 'cache', 'app'])}-{rng.randint(1, 20)}.{rng.choice(['prod', 'staging', 'dev'])}.{rng.choice(['acme.com', 'example.net'])}"


def gen_api_key(rng: random.Random) -> str:
    return f"sk_{''.join(rng.choices(string.ascii_letters + string.digits, k=32))}"


def gen_session_token(rng: random.Random) -> str:
    return f"sess_{''.join(rng.choices(string.ascii_letters + string.digits, k=40))}"


def gen_regex(rng: random.Random) -> str:
    return rng.choice([r"^\d{3}-\d{2}-\d{4}$", r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+",
                        r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", r"^[A-Z]{2}\d{6}$"])


def gen_cron(rng: random.Random) -> str:
    return f"{rng.randint(0, 59)} {rng.randint(0, 23)} * * {rng.choice(['*', '1-5', '0,6'])}"


def gen_ticker(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_uppercase, k=rng.randint(2, 4)))


def gen_isbn(rng: random.Random) -> str:
    return f"978-{rng.randint(0, 9)}-{rng.randint(10000, 99999)}-{rng.randint(100, 999)}-{rng.randint(0, 9)}"


def gen_doi(rng: random.Random) -> str:
    return f"10.{rng.randint(1000, 9999)}/{rng.choice(string.ascii_lowercase)}{rng.randint(100, 999)}.{rng.randint(1, 50)}"


def gen_postal_code(rng: random.Random) -> str:
    if rng.random() < 0.5:
        return f"{rng.randint(10000, 99999)}"
    return f"{rng.choice(string.ascii_uppercase)}{rng.choice(string.ascii_uppercase)}{rng.randint(1, 9)} {rng.randint(1, 9)}{rng.choice(string.ascii_uppercase)}{rng.choice(string.ascii_uppercase)}"


def gen_iata(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_uppercase, k=3))


def gen_sku(rng: random.Random) -> str:
    return f"{''.join(rng.choices(string.ascii_uppercase, k=3))}-{rng.randint(1000, 9999)}"


def gen_coordinates(rng: random.Random) -> str:
    return f"{round(rng.uniform(-90, 90), 6)}, {round(rng.uniform(-180, 180), 6)}"


def gen_quarter(rng: random.Random) -> str:
    return f"Q{rng.randint(1, 4)} {rng.randint(2020, 2026)}"


def gen_day_of_week(rng: random.Random) -> str:
    return rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])


def gen_description(rng: random.Random) -> str:
    subjects = ["The system", "This record", "The product", "The service", "The customer account"]
    verbs = ["includes", "represents", "tracks", "monitors", "manages"]
    objects = ["data processing activities", "customer interactions", "operational metrics", "compliance records", "transaction histories"]
    return f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)}."


def gen_comment(rng: random.Random) -> str:
    comments = [
        "Reviewed and approved.", "Needs follow-up next quarter.",
        "Updated per client request.", "Flagged for review.",
        "Automated processing complete.", "Escalated to management.",
        "Resolved — no further action.", "Pending verification.",
    ]
    return rng.choice(comments)


def gen_keywords(rng: random.Random) -> str:
    words = ["data", "analytics", "cloud", "security", "compliance", "automation",
             "integration", "reporting", "performance", "monitoring", "scalability"]
    return ", ".join(rng.sample(words, rng.randint(2, 5)))


def gen_log_message(rng: random.Random) -> str:
    levels = ["INFO", "DEBUG", "WARN", "ERROR"]
    msgs = ["Request processed in {}ms", "Connection established to {}", "Cache miss for key {}",
            "Retry attempt {} of 3", "Health check passed", "Batch {} completed"]
    return f"[{rng.choice(levels)}] {rng.choice(msgs).format(rng.randint(1, 9999))}"


def gen_error_message(rng: random.Random) -> str:
    errors = [
        "NullPointerException at line {}", "Connection timeout after {}ms",
        "FileNotFoundError: {}", "ValueError: invalid literal for int()",
        "PermissionDenied: insufficient privileges", "HTTP 503 Service Unavailable",
    ]
    return rng.choice(errors).format(rng.randint(1, 9999))


def gen_filepath(rng: random.Random) -> str:
    dirs = ["data", "logs", "config", "exports", "uploads", "tmp"]
    exts = [".csv", ".json", ".parquet", ".log", ".yaml", ".xml"]
    return f"/var/{rng.choice(dirs)}/{rng.choice(['batch', 'daily', 'export'])}_{rng.randint(1, 999)}{rng.choice(exts)}"


# ── Full generator registry ───────────────────────────────────
#
# Maps every leaf ICE.* code to a generator function.
# This is the authoritative mapping used by both synth.py and
# generate_sample_source.py.

def build_generators() -> dict[str, Callable[[random.Random], str]]:
    """Build the full code → generator mapping."""
    g: dict[str, Callable[[random.Random], str]] = {}

    # ── Contact ──
    g["ICE.SENSITIVE.PID.CONTACT.EMAIL"] = gen_email
    g["ICE.SENSITIVE.PID.CONTACT.PHONE"] = gen_phone
    g["ICE.SENSITIVE.PID.CONTACT.ADDRESS"] = gen_address
    g["ICE.SENSITIVE.PID.CONTACT.FAX"] = gen_fax
    g["ICE.SENSITIVE.PID.CONTACT.SOCIAL_MEDIA"] = gen_social
    g["ICE.SENSITIVE.PID.CONTACT.MESSENGER"] = lambda rng: f"{rng.choice(FIRST_NAMES).lower()}_{rng.choice(LAST_NAMES).lower()}"

    # ── Identity ──
    g["ICE.SENSITIVE.PID.IDENTITY.NAME.FULLNAME"] = gen_fullname
    g["ICE.SENSITIVE.PID.IDENTITY.NAME.FIRST_NAME"] = gen_firstname
    g["ICE.SENSITIVE.PID.IDENTITY.NAME.LAST_NAME"] = gen_lastname
    g["ICE.SENSITIVE.PID.IDENTITY.NAME.MIDDLE_NAME"] = lambda rng: rng.choice(string.ascii_uppercase)
    g["ICE.SENSITIVE.PID.IDENTITY.NAME.MAIDEN_NAME"] = gen_lastname
    g["ICE.SENSITIVE.PID.IDENTITY.DOB"] = gen_dob
    g["ICE.SENSITIVE.PID.IDENTITY.GOVID.SSN"] = gen_ssn
    g["ICE.SENSITIVE.PID.IDENTITY.GENDER_PII"] = lambda rng: rng.choice(["Male", "Female", "Non-binary", "Other"])
    g["ICE.SENSITIVE.PID.IDENTITY.NATIONALITY"] = gen_country
    g["ICE.SENSITIVE.PID.IDENTITY.ETHNICITY"] = lambda rng: rng.choice(["Asian", "Black", "Hispanic", "White", "Mixed", "Other"])
    g["ICE.SENSITIVE.PID.IDENTITY.BIRTH_PLACE"] = gen_city
    g["ICE.SENSITIVE.PID.IDENTITY.MARITAL_STATUS"] = lambda rng: rng.choice(["Single", "Married", "Divorced", "Widowed"])
    g["ICE.SENSITIVE.PID.IDENTITY.BIOMETRIC.PHOTO"] = lambda rng: f"photo_{gen_uuid(rng)[:8]}.jpg"
    g["ICE.SENSITIVE.PID.IDENTITY.BIOMETRIC.SIGNATURE"] = lambda rng: f"sig_{gen_hash(rng)[:16]}"
    g["ICE.SENSITIVE.PID.IDENTITY.BIOMETRIC.VOICEPRINT"] = lambda rng: f"voice_{gen_hash(rng)[:16]}.wav"
    g["ICE.SENSITIVE.PID.IDENTITY.BIOMETRIC.FINGERPRINT"] = lambda rng: f"fp_{gen_hash(rng)[:16]}"
    g["ICE.SENSITIVE.PID.IDENTITY.GOVID.DLN"] = lambda rng: f"{rng.choice(STATES)}{rng.randint(1000000, 9999999)}"
    g["ICE.SENSITIVE.PID.IDENTITY.GOVID.PASSPORT"] = lambda rng: f"{rng.choice(string.ascii_uppercase)}{rng.randint(10000000, 99999999)}"
    g["ICE.SENSITIVE.PID.IDENTITY.GOVID.WORK_PERMIT"] = lambda rng: f"WP-{rng.choice(COUNTRY_CODES)}-{rng.randint(100000, 999999)}"
    g["ICE.SENSITIVE.PID.IDENTITY.NAME.TITLE_PREFIX"] = lambda rng: rng.choice(["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Rev.", "Hon."])
    g["ICE.SENSITIVE.PID.IDENTITY.PLATFORM_ID.USERNAME"] = lambda rng: f"{rng.choice(FIRST_NAMES).lower()}{rng.choice(['_', '.', ''])}{rng.choice(LAST_NAMES).lower()}{rng.randint(1, 999)}"
    g["ICE.SENSITIVE.PID.IDENTITY.PLATFORM_ID.EMPLOYEE_ID"] = lambda rng: f"EMP{rng.randint(10000, 99999)}"
    g["ICE.SENSITIVE.PID.IDENTITY.PLATFORM_ID.ADVERTISER_ID"] = lambda rng: f"ADV-{''.join(rng.choices(string.ascii_uppercase + string.digits, k=10))}"

    # ── Financial ──
    g["ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.PAN"] = gen_pan
    g["ICE.SENSITIVE.PID.FINANCIAL.ACCOUNT.BAN"] = gen_ban
    g["ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.TXNAMT"] = gen_amount
    g["ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.CVV"] = gen_cvv
    g["ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.EXPIRY"] = gen_expiry
    g["ICE.SENSITIVE.PID.IDENTITY.GOVID.TAX_ID"] = lambda rng: f"{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}"
    g["ICE.SENSITIVE.PID.FINANCIAL.INCOME.SALARY"] = gen_salary
    g["ICE.SENSITIVE.PID.FINANCIAL.CREDIT.CREDIT_SCORE"] = lambda rng: str(rng.randint(300, 850))
    g["ICE.SENSITIVE.PID.FINANCIAL.ACCOUNT.INVESTMENT"] = lambda rng: f"INV-{rng.randint(100000, 999999)}"
    g["ICE.SENSITIVE.PID.FINANCIAL.CREDIT.INSURANCE_ID"] = lambda rng: f"INS-{rng.randint(100000, 999999)}"
    g["ICE.SENSITIVE.PID.FINANCIAL.ACCOUNT.ROUTING_NUM"] = lambda rng: "".join(str(rng.randint(0, 9)) for _ in range(9))
    g["ICE.SENSITIVE.PID.FINANCIAL.ACCOUNT.CRYPTO_ADDR"] = lambda rng: f"0x{''.join(rng.choices('0123456789abcdef', k=40))}"
    g["ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.BIN"] = lambda rng: rng.choice(["4", "5", "37", "6011"]) + "".join(str(rng.randint(0, 9)) for _ in range(rng.choice([4, 5])))
    g["ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.LAST4"] = lambda rng: "".join(str(rng.randint(0, 9)) for _ in range(4))
    g["ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.MAGSTRIPE"] = lambda rng: f"%B{''.join(str(rng.randint(0, 9)) for _ in range(16))}^{rng.choice(LAST_NAMES).upper()}/{rng.choice(FIRST_NAMES).upper()}^{rng.randint(24, 30)}{rng.randint(1, 12):02d}"
    g["ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.BILLING.BILLING_ACCT"] = lambda rng: f"BILL-{rng.randint(100000, 999999)}"
    g["ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.BILLING.PAYPAL"] = lambda rng: f"{rng.choice(FIRST_NAMES).lower()}.{rng.choice(LAST_NAMES).lower()}@{rng.choice(['gmail.com', 'outlook.com', 'yahoo.com'])}"
    g["ICE.SENSITIVE.PID.FINANCIAL.INCOME.STOCK_RSU"] = lambda rng: f"{rng.randint(50, 5000)} shares @ ${round(rng.uniform(10, 500), 2)}"
    g["ICE.SENSITIVE.PID.FINANCIAL.INCOME.BONUS"] = lambda rng: f"${rng.randint(1000, 50000):,}"
    g["ICE.SENSITIVE.PID.FINANCIAL.CREDIT.FRAUD_SCORE"] = lambda rng: str(round(rng.uniform(0, 100), 1))

    # ── Health ──
    g["ICE.SENSITIVE.PID.HEALTH.DIAGNOSIS"] = lambda rng: rng.choice(["Hypertension", "Type 2 Diabetes", "Asthma", "Migraine", "Arthritis", "Anxiety", "Depression"])
    g["ICE.SENSITIVE.PID.HEALTH.PRESCRIPTION"] = lambda rng: rng.choice(["Lisinopril 10mg", "Metformin 500mg", "Albuterol 90mcg", "Ibuprofen 200mg", "Omeprazole 20mg"])
    g["ICE.SENSITIVE.PID.HEALTH.BLOOD_TYPE"] = lambda rng: rng.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
    g["ICE.SENSITIVE.PID.HEALTH.ALLERGY"] = lambda rng: rng.choice(["Penicillin", "Peanuts", "Latex", "Shellfish", "Pollen", "None"])
    g["ICE.SENSITIVE.PID.HEALTH.MEDICAL_RECORD"] = lambda rng: f"MRN-{rng.randint(100000, 999999)}"
    g["ICE.SENSITIVE.PID.HEALTH.INSURANCE_CLAIM"] = lambda rng: f"CLM-{rng.randint(100000, 999999)}"
    g["ICE.SENSITIVE.PID.HEALTH.LAB_RESULT"] = lambda rng: f"{rng.choice(['Glucose', 'Cholesterol', 'Hemoglobin', 'WBC'])}: {round(rng.uniform(50, 300), 1)} {rng.choice(['mg/dL', 'g/dL', 'K/uL'])}"
    g["ICE.SENSITIVE.PID.HEALTH.VITAL_SIGN"] = lambda rng: f"BP {rng.randint(90, 180)}/{rng.randint(60, 110)} HR {rng.randint(55, 120)}"
    g["ICE.SENSITIVE.PID.HEALTH.PROCEDURE"] = lambda rng: rng.choice(["Blood Draw", "X-Ray Chest", "MRI Brain", "ECG", "Colonoscopy", "Physical Exam"])
    g["ICE.SENSITIVE.PID.HEALTH.DISABILITY"] = lambda rng: rng.choice(["None", "Visual", "Hearing", "Mobility", "Cognitive"])

    # ── Technical ──
    g["ICE.SENSITIVE.TECHNICAL.IPADDR"] = gen_ipv4
    g["ICE.SENSITIVE.TECHNICAL.DEVID"] = gen_uuid
    g["ICE.SENSITIVE.TECHNICAL.URL_PII"] = lambda rng: f"https://example.com/users/{gen_uuid(rng)[:8]}/profile"
    g["ICE.SENSITIVE.TECHNICAL.HOSTNAME"] = gen_hostname
    g["ICE.SENSITIVE.TECHNICAL.API_KEY"] = gen_api_key
    g["ICE.SENSITIVE.TECHNICAL.PASSWORD_HASH"] = lambda rng: f"$2b$12${''.join(rng.choices(string.ascii_letters + string.digits, k=53))}"
    g["ICE.SENSITIVE.TECHNICAL.SESSION_TOKEN"] = gen_session_token
    g["ICE.SENSITIVE.TECHNICAL.COOKIE"] = lambda rng: f"_ga={''.join(rng.choices(string.digits, k=10))}.{rng.randint(1000000, 9999999)}"
    g["ICE.SENSITIVE.TECHNICAL.SSH_KEY"] = lambda rng: f"ssh-rsa {''.join(rng.choices(string.ascii_letters + string.digits + '+/', k=44))}="
    g["ICE.SENSITIVE.TECHNICAL.CERTIFICATE"] = lambda rng: f"-----BEGIN CERTIFICATE-----\n{''.join(rng.choices(string.ascii_letters + string.digits + '+/', k=64))}\n-----END CERTIFICATE-----"
    g["ICE.SENSITIVE.TECHNICAL.DATABASE_CONN"] = lambda rng: f"postgresql://user:****@{rng.choice(['db-prod', 'db-staging'])}.internal:5432/{rng.choice(['main', 'analytics'])}"

    # ── Business ──
    g["ICE.SENSITIVE.BUSINESS.TRADE_SECRET"] = lambda rng: f"[CONFIDENTIAL] Formula #{rng.randint(100, 999)}"
    g["ICE.SENSITIVE.BUSINESS.CONTRACT_VALUE"] = lambda rng: f"${rng.randint(10000, 10000000):,}"
    g["ICE.SENSITIVE.BUSINESS.CUSTOMER_LIST"] = lambda rng: ", ".join(rng.sample(ORGS, rng.randint(2, 5)))
    g["ICE.SENSITIVE.BUSINESS.PRICING_STRATEGY"] = lambda rng: rng.choice(["Cost-plus 25%", "Market-based", "Value pricing", "Penetration", "Premium"])
    g["ICE.SENSITIVE.BUSINESS.MARKET_INTEL"] = lambda rng: f"Competitor {rng.choice(ORGS)} launched {rng.choice(PRODUCTS)}"
    g["ICE.SENSITIVE.BUSINESS.INTERNAL_MEMO"] = lambda rng: f"[INTERNAL] Q{rng.randint(1, 4)} strategy review pending board approval"
    g["ICE.SENSITIVE.BUSINESS.LEGAL_HOLD"] = lambda rng: f"HOLD-{rng.randint(2020, 2026)}-{rng.randint(100, 999)}"
    g["ICE.SENSITIVE.BUSINESS.AUDIT_FINDING"] = lambda rng: rng.choice(["Non-conformance: access control", "Observation: backup schedule", "Major finding: data retention"])
    g["ICE.SENSITIVE.BUSINESS.FINANCIAL_FORECAST"] = lambda rng: f"Q{rng.randint(1, 4)} {rng.randint(2024, 2027)}: ${rng.randint(1, 100)}M projected"
    g["ICE.SENSITIVE.BUSINESS.STRATEGIC_PLAN"] = lambda rng: rng.choice(["Expand APAC market", "Launch SaaS tier", "Acquire competitor", "IPO readiness"])
    g["ICE.SENSITIVE.BUSINESS.EMPLOYEE_RECORD"] = lambda rng: f"EMP-{rng.randint(10000, 99999)}"
    g["ICE.SENSITIVE.BUSINESS.PERFORMANCE_REVIEW"] = lambda rng: rng.choice(["Exceeds expectations", "Meets expectations", "Needs improvement", "Outstanding"])

    # ── Designative: Names ──
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.PERSON"] = gen_fullname
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.ORG"] = gen_org
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.PRODUCT"] = gen_product
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.SCIENTIFIC"] = lambda rng: rng.choice(["Homo sapiens", "E. coli", "NaCl", "H2O", "C6H12O6", "Fe2O3"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.GEO"] = gen_country
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.EVENT"] = lambda rng: f"{rng.choice(['Annual', 'Global', 'Tech', 'Data'])} {rng.choice(['Summit', 'Conference', 'Forum', 'Expo'])} {rng.randint(2020, 2026)}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.PROJECT"] = lambda rng: f"Project {rng.choice(['Alpha', 'Beta', 'Phoenix', 'Atlas', 'Horizon', 'Zenith'])}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.MATERIAL"] = lambda rng: rng.choice(["Steel", "Aluminum", "Carbon Fiber", "Titanium", "Polyethylene", "Glass"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.AWARD"] = lambda rng: rng.choice(["Employee of the Month", "Innovation Award", "Best Paper", "Gold Medal", "Excellence Award"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.BRAND"] = lambda rng: rng.choice(["TechBrand", "EcoLine", "ProSeries", "SmartChoice", "ValueMax"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.VENUE"] = lambda rng: rng.choice(["Convention Center", "Grand Ballroom", "Stadium", "Amphitheater", "Conference Hall"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.ARTWORK"] = lambda rng: rng.choice(["Starry Night", "The Persistence of Memory", "Water Lilies", "The Scream"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.SONG"] = lambda rng: rng.choice(["Bohemian Rhapsody", "Imagine", "Yesterday", "Hotel California", "Stairway to Heaven"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.MOVIE"] = lambda rng: rng.choice(["The Matrix", "Inception", "Interstellar", "Blade Runner", "The Godfather"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.BOOK"] = lambda rng: rng.choice(["1984", "Dune", "The Great Gatsby", "To Kill a Mockingbird", "Brave New World"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.DISEASE"] = lambda rng: rng.choice(["Influenza", "Malaria", "Tuberculosis", "COVID-19", "Diabetes Mellitus"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.DRUG"] = lambda rng: rng.choice(["Aspirin", "Metformin", "Amoxicillin", "Ibuprofen", "Lisinopril"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.FOOD"] = lambda rng: rng.choice(["Pasta", "Sushi", "Tacos", "Pizza", "Curry", "Steak", "Salad"])

    # ── Designative: Codes ──
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.ID"] = gen_uuid
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.ABBREV"] = lambda rng: "".join(rng.choices(string.ascii_uppercase, k=rng.randint(2, 5)))
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.POSTAL"] = gen_postal_code
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.ISO_COUNTRY"] = gen_country_code
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.ISO_CURRENCY"] = gen_currency_code
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.ISO_LANGUAGE"] = gen_lang_code
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.IATA"] = gen_iata
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.NAICS"] = lambda rng: str(rng.randint(110000, 999999))
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.TICKER"] = gen_ticker
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.DOI"] = gen_doi
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.ISBN"] = gen_isbn
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.ISSN"] = lambda rng: f"{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.SKU"] = gen_sku
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.FIPS"] = lambda rng: f"{rng.randint(1, 56):02d}{rng.randint(1, 999):03d}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.LEI"] = lambda rng: "".join(rng.choices(string.ascii_uppercase + string.digits, k=20))
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.DUNS"] = lambda rng: "".join(str(rng.randint(0, 9)) for _ in range(9))
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.MIME_TYPE"] = gen_mime
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.PHONE_CODE"] = lambda rng: f"+{rng.choice([1, 44, 49, 33, 81, 86, 91, 61, 55])}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.CAS"] = lambda rng: f"{rng.randint(50, 99999)}-{rng.randint(10, 99)}-{rng.randint(0, 9)}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.VIN"] = lambda rng: "".join(rng.choices(string.ascii_uppercase + string.digits, k=17))
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.CUSIP"] = lambda rng: "".join(rng.choices(string.ascii_uppercase + string.digits, k=9))
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.CIK"] = lambda rng: str(rng.randint(100000, 9999999)).zfill(10)
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.GTIN"] = lambda rng: "".join(str(rng.randint(0, 9)) for _ in range(13))
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.HASH_ID"] = gen_hash
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.SEMANTIC_VERSION"] = gen_version
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.PLATE_NUMBER"] = lambda rng: f"{''.join(rng.choices(string.ascii_uppercase, k=3))}-{rng.randint(1000, 9999)}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.FLIGHT_NUMBER"] = lambda rng: f"{''.join(rng.choices(string.ascii_uppercase, k=2))}{rng.randint(100, 9999)}"

    # ── Designative: Geographic ──
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.COUNTRY"] = gen_country
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.REGION"] = gen_region
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.CITY"] = gen_city
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.LOCATION"] = lambda rng: f"{rng.choice(CITIES)}, {rng.choice(COUNTRIES)}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.COORDINATES"] = gen_coordinates
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.CONTINENT"] = lambda rng: rng.choice(["North America", "Europe", "Asia", "Africa", "South America", "Oceania", "Antarctica"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.DISTRICT"] = lambda rng: rng.choice(["Downtown", "Midtown", "West End", "East Side", "North Shore", "South Bay"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.TIMEZONE"] = lambda rng: rng.choice(["UTC", "EST", "PST", "CET", "JST", "America/New_York", "Europe/London"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.ADDRESS_LINE"] = lambda rng: f"{rng.randint(1, 9999)} {rng.choice(STREETS)}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.LANDMARK"] = lambda rng: rng.choice(["Central Park", "Eiffel Tower", "Golden Gate Bridge", "Big Ben", "Statue of Liberty"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.AIRPORT"] = lambda rng: rng.choice(["JFK", "LAX", "ORD", "LHR", "NRT", "SFO", "CDG", "SIN"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.PORT"] = lambda rng: rng.choice(["Port of LA", "Rotterdam", "Shanghai", "Singapore", "Hamburg", "Busan"])

    # ── Designative: References ──
    g["ICE.NONSENSITIVE.DESIGNATIVE.REF.CITATION"] = lambda rng: f"{rng.choice(LAST_NAMES)} et al. ({rng.randint(2000, 2026)})"
    g["ICE.NONSENSITIVE.DESIGNATIVE.REF.VERSION"] = gen_version
    g["ICE.NONSENSITIVE.DESIGNATIVE.REF.SOURCE"] = lambda rng: rng.choice(["Bloomberg", "Reuters", "Census Bureau", "WHO", "CDC", "Eurostat"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.REF.URL"] = gen_url
    g["ICE.NONSENSITIVE.DESIGNATIVE.REF.FILEPATH"] = gen_filepath

    # ── Designative: Other ──
    g["ICE.NONSENSITIVE.DESIGNATIVE.TITLE"] = lambda rng: f"{rng.choice(['Annual', 'Quarterly', 'Technical', 'Executive'])} {rng.choice(['Report', 'Review', 'Summary', 'Analysis'])} {rng.randint(2020, 2026)}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.LABEL"] = lambda rng: rng.choice(["high-priority", "needs-review", "approved", "deprecated", "experimental", "stable"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.BOOLEAN"] = gen_boolean
    g["ICE.NONSENSITIVE.DESIGNATIVE.EMAIL_DOMAIN"] = lambda rng: rng.choice(DOMAINS)
    g["ICE.NONSENSITIVE.DESIGNATIVE.DOMAIN_NAME"] = lambda rng: rng.choice(["example.com", "acme.org", "data.io", "company.net", "service.cloud"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.PHONE_FORMAT"] = lambda rng: rng.choice(["+1-XXX-XXX-XXXX", "+44-XXXX-XXXXXX", "+81-XX-XXXX-XXXX"])

    # ── Descriptive: Text ──
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.DESCRIPTION"] = gen_description
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.COMMENT"] = gen_comment
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.ABSTRACT"] = lambda rng: f"This paper presents {rng.choice(['a novel approach', 'an analysis', 'a framework'])} for {rng.choice(['data classification', 'entity resolution', 'anomaly detection'])}."
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.DEFINITION"] = lambda rng: f"A {rng.choice(['systematic', 'formal', 'operational'])} definition of {rng.choice(['compliance', 'risk', 'data quality'])}."
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.BODY"] = lambda rng: "  ".join(gen_description(rng) for _ in range(rng.randint(2, 4)))
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.KEYWORDS"] = gen_keywords
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.INSTRUCTION"] = lambda rng: f"Step {rng.randint(1, 5)}: {rng.choice(['Configure', 'Verify', 'Initialize', 'Deploy', 'Monitor'])} the {rng.choice(['system', 'service', 'pipeline', 'module'])}."
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.REVIEW"] = lambda rng: rng.choice(["Great product!", "Needs improvement.", "Average experience.", "Excellent service.", "Not recommended."])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.LOG_MESSAGE"] = gen_log_message
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.ERROR_MESSAGE"] = gen_error_message
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.QUERY"] = lambda rng: f"SELECT * FROM {rng.choice(['users', 'orders', 'products'])} WHERE {rng.choice(['status', 'type', 'id'])} = '{rng.choice(['active', 'pending', '1'])}'"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.TRANSLATION"] = lambda rng: rng.choice(["Bonjour", "Hola", "Guten Tag", "Konnichiwa", "Namaste"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.HEADLINE"] = lambda rng: f"{rng.choice(['Breaking', 'New', 'Updated'])}: {rng.choice(['Product Launch', 'Policy Change', 'Partnership'])}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.CAPTION"] = lambda rng: f"Figure {rng.randint(1, 20)}: {rng.choice(['Overview', 'Comparison', 'Distribution', 'Timeline'])}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.ADDRESS_TEXT"] = lambda rng: f"{rng.randint(1, 9999)} {rng.choice(STREETS)}, {rng.choice(CITIES)}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.BIOGRAPHY"] = lambda rng: f"{rng.choice(FIRST_NAMES)} is a {rng.choice(ROLES)} with {rng.randint(1, 30)} years of experience."
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.CHANGELOG"] = lambda rng: f"v{gen_version(rng)}: {rng.choice(['Fixed bug in', 'Added support for', 'Improved', 'Removed'])} {rng.choice(['authentication', 'caching', 'logging', 'API'])}."
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.RECIPE"] = lambda rng: f"Mix {rng.randint(1, 3)} cups {rng.choice(['flour', 'sugar', 'rice'])} with {rng.randint(1, 2)} tbsp {rng.choice(['oil', 'butter', 'water'])}."

    # ── Descriptive: Categorical ──
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.TYPE"] = lambda rng: rng.choice(["Standard", "Premium", "Enterprise", "Free", "Trial", "Basic"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.CATEGORY"] = lambda rng: rng.choice(["Electronics", "Clothing", "Food", "Services", "Software", "Hardware"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.RANK"] = lambda rng: str(rng.randint(1, 100))
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.LANGUAGE"] = gen_language
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.GENDER"] = lambda rng: rng.choice(["Male", "Female", "Unisex"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.COLOR"] = gen_color
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.SIZE"] = gen_size
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.STATUS"] = gen_status
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.INDUSTRY"] = gen_industry
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.DEPARTMENT"] = gen_department
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.ROLE"] = gen_role
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.FORMAT"] = lambda rng: rng.choice(["CSV", "JSON", "XML", "Parquet", "Avro", "YAML"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.PROTOCOL"] = gen_protocol
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.BOOLEAN_ENUM"] = lambda rng: rng.choice(["Active", "Inactive", "Enabled", "Disabled"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.SEVERITY"] = gen_severity
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.SENTIMENT"] = gen_sentiment
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.LIFECYCLE"] = lambda rng: rng.choice(["Draft", "In Review", "Published", "Archived", "Deprecated"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.CHANNEL"] = gen_channel
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.PLATFORM"] = gen_platform
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.BROWSER"] = gen_browser
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.DEVICE_TYPE"] = gen_device_type
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.CONTENT_TYPE"] = lambda rng: rng.choice(["Article", "Video", "Podcast", "Image", "Infographic"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.PAYMENT_METHOD"] = gen_payment_method
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.SHIPPING_METHOD"] = gen_shipping
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.EDUCATION_LEVEL"] = lambda rng: rng.choice(["High School", "Bachelor's", "Master's", "PhD", "Associate's"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.RELATIONSHIP"] = lambda rng: rng.choice(["Parent", "Child", "Sibling", "Spouse", "Manager", "Peer"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.PERMISSION_LEVEL"] = lambda rng: rng.choice(["Admin", "Editor", "Viewer", "Owner", "Guest"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.RISK_LEVEL"] = lambda rng: rng.choice(["Critical", "High", "Medium", "Low", "Negligible"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.APPROVAL_STATUS"] = lambda rng: rng.choice(["Approved", "Pending", "Rejected", "In Review"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.CURRENCY_NAME"] = lambda rng: rng.choice(["US Dollar", "Euro", "British Pound", "Japanese Yen", "Swiss Franc"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.UNIT"] = lambda rng: rng.choice(["kg", "lb", "m", "ft", "L", "gal", "\u00b0C", "\u00b0F"])

    # ── Descriptive: Measurement ──
    _measurement_gens: list[tuple[str, Callable]] = [
        ("LENGTH", lambda rng: f"{round(rng.uniform(0.1, 1000), 2)}"),
        ("AREA", lambda rng: f"{round(rng.uniform(1, 100000), 1)}"),
        ("VOLUME", lambda rng: f"{round(rng.uniform(0.1, 10000), 2)}"),
        ("WEIGHT", gen_weight),
        ("TEMPERATURE", gen_temperature),
        ("SPEED", lambda rng: f"{round(rng.uniform(0, 300), 1)}"),
        ("PRESSURE", lambda rng: f"{round(rng.uniform(0, 2000), 1)}"),
        ("FREQUENCY", lambda rng: f"{round(rng.uniform(0.1, 10000), 2)}"),
        ("ENERGY", lambda rng: f"{round(rng.uniform(0, 100000), 1)}"),
        ("POWER", lambda rng: f"{round(rng.uniform(0, 50000), 1)}"),
        ("DENSITY", lambda rng: f"{round(rng.uniform(0.1, 20), 3)}"),
        ("CONCENTRATION", lambda rng: f"{round(rng.uniform(0, 1000), 2)}"),
        ("PERCENTAGE", gen_percentage),
        ("SCORE", gen_score),
        ("COUNT", gen_count),
        ("DURATION", gen_duration),
        ("AGE", gen_age),
        ("PRICE", gen_price),
        ("REVENUE", lambda rng: f"${rng.randint(1000, 10000000):,}"),
        ("BUDGET", lambda rng: f"${rng.randint(1000, 5000000):,}"),
        ("MARKET_VALUE", lambda rng: f"${rng.randint(100000, 1000000000):,}"),
        ("EXCHANGE_RATE", lambda rng: f"{round(rng.uniform(0.5, 150), 4)}"),
        ("LATITUDE", gen_latitude),
        ("LONGITUDE", gen_longitude),
        ("ELEVATION", gen_elevation),
        ("DISTANCE", gen_distance),
        ("BANDWIDTH", lambda rng: f"{rng.randint(1, 10000)}"),
        ("STORAGE", lambda rng: f"{rng.randint(1, 1000000)}"),
        ("LATENCY", lambda rng: f"{rng.randint(1, 5000)}"),
        ("UPTIME", lambda rng: f"{round(rng.uniform(95, 100), 4)}"),
        ("CPU_USAGE", lambda rng: f"{round(rng.uniform(0, 100), 1)}"),
        ("MEMORY_USAGE", lambda rng: f"{round(rng.uniform(0, 100), 1)}"),
        ("VOLTAGE", lambda rng: f"{round(rng.uniform(0, 480), 1)}"),
        ("CURRENT", lambda rng: f"{round(rng.uniform(0, 100), 2)}"),
        ("ANGLE", lambda rng: f"{round(rng.uniform(0, 360), 2)}"),
        ("FLOW_RATE", lambda rng: f"{round(rng.uniform(0, 1000), 2)}"),
        ("HUMIDITY", lambda rng: f"{round(rng.uniform(0, 100), 1)}"),
        ("LUMINOSITY", lambda rng: f"{round(rng.uniform(0, 100000), 1)}"),
        ("DECIBEL", lambda rng: f"{round(rng.uniform(0, 130), 1)}"),
        ("PH", lambda rng: f"{round(rng.uniform(0, 14), 2)}"),
        ("RESOLUTION", lambda rng: rng.choice(["1920x1080", "3840x2160", "1280x720", "2560x1440"])),
        ("RATIO", lambda rng: f"{round(rng.uniform(0, 10), 4)}"),
        ("POPULATION", lambda rng: f"{rng.randint(100, 10000000)}"),
    ]
    for code_suffix, gen in _measurement_gens:
        g[f"ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.{code_suffix}"] = gen

    # ── Descriptive: Temporal ──
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.DATE"] = gen_date
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.DATETIME"] = gen_datetime
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.TIME"] = gen_time
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.YEAR"] = gen_year
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.MONTH"] = gen_month
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.DAY_OF_WEEK"] = gen_day_of_week
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.QUARTER"] = gen_quarter
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.PERIOD"] = lambda rng: rng.choice(["Q1 2024", "FY 2025", "H2 2023", "2020-2025", "Jan-Mar"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.START_DATE"] = gen_date
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.END_DATE"] = gen_date
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.CRON"] = gen_cron

    # ── Descriptive: Numeric ──
    g["ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC.INTEGER"] = gen_integer
    g["ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC.DECIMAL"] = gen_decimal
    g["ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC.ORDINAL"] = lambda rng: str(rng.randint(1, 1000))
    g["ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC.PROBABILITY"] = lambda rng: f"{round(rng.uniform(0, 1), 4)}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC.BINARY"] = lambda rng: str(rng.choice([0, 1]))
    g["ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC.RANGE"] = lambda rng: f"{rng.randint(0, 50)}-{rng.randint(51, 100)}"

    # ── Descriptive: Embedding ──
    g["ICE.NONSENSITIVE.DESCRIPTIVE.EMBEDDING.WORD_EMBEDDING"] = lambda rng: str([round(rng.gauss(0, 1), 4) for _ in range(8)])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.EMBEDDING.FEATURE_VECTOR"] = lambda rng: str([round(rng.uniform(0, 1), 4) for _ in range(8)])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.EMBEDDING.IMAGE_EMBEDDING"] = lambda rng: str([round(rng.gauss(0, 0.5), 4) for _ in range(8)])

    # ── Descriptive: Statistical ──
    g["ICE.NONSENSITIVE.DESCRIPTIVE.STATISTICAL.MEAN"] = lambda rng: f"{round(rng.uniform(-100, 100), 4)}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.STATISTICAL.MEDIAN"] = lambda rng: f"{round(rng.uniform(-100, 100), 4)}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.STATISTICAL.STD_DEV"] = lambda rng: f"{round(rng.uniform(0, 50), 4)}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.STATISTICAL.PERCENTILE"] = lambda rng: f"{round(rng.uniform(0, 100), 2)}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.STATISTICAL.CORRELATION"] = lambda rng: f"{round(rng.uniform(-1, 1), 4)}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.STATISTICAL.TREND"] = lambda rng: f"{round(rng.uniform(-50, 50), 2)}%"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.STATISTICAL.HISTOGRAM_BIN"] = lambda rng: f"{round(rng.uniform(0, 100), 2)}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.STATISTICAL.CONFIDENCE_INTERVAL"] = lambda rng: f"[{round(rng.uniform(0, 40), 2)}, {round(rng.uniform(60, 100), 2)}]"

    # ── Prescriptive ──
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.FORMAT_SPEC"] = lambda rng: rng.choice(["RFC 3339", "ISO 8601", "POSIX", "IEEE 754", "W3C Date"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.FORMULA"] = lambda rng: rng.choice(["E = mc\u00b2", "F = ma", "PV = nRT", "A = \u03c0r\u00b2", "BMI = kg/m\u00b2"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.RULE"] = lambda rng: rng.choice(["age >= 18", "amount <= 10000", "status IN ('active','pending')", "NOT NULL"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.CONFIG"] = lambda rng: f"{rng.choice(['max_retries', 'timeout_ms', 'batch_size', 'pool_size'])}={rng.randint(1, 1000)}"
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.PERMISSION"] = lambda rng: rng.choice(["read", "write", "admin", "execute", "rw", "r-x", "rwx"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.SCHEDULE"] = lambda rng: rng.choice(["Daily at 2am", "Every 15 minutes", "Weekly on Monday", "Monthly on 1st"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.COMMAND"] = lambda rng: rng.choice(["GET /api/users", "POST /api/orders", "kubectl apply", "docker run", "npm install"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.TEMPLATE"] = lambda rng: rng.choice(["Hello {{name}}", "Order #{{id}}", "Dear {{customer}}", "{{date}} - {{event}}"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.THRESHOLD"] = lambda rng: f"{rng.choice(['max', 'min', 'warn', 'crit'])}={rng.randint(1, 10000)}"
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.LICENSE"] = lambda rng: rng.choice(["MIT", "Apache-2.0", "GPL-3.0", "BSD-3-Clause", "Proprietary", "CC BY 4.0"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.SLA"] = lambda rng: f"{round(rng.uniform(99, 100), 3)}% uptime"
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.POLICY"] = lambda rng: rng.choice(["Data retained for 7 years", "PII encrypted at rest", "MFA required for admin"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.API_SPEC"] = lambda rng: rng.choice(["OpenAPI 3.0", "GraphQL", "gRPC/protobuf", "REST/JSON", "SOAP/XML"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.FILTER"] = lambda rng: f"{rng.choice(['status', 'type', 'date'])} {rng.choice(['=', '!=', '>', '<', 'IN'])} '{rng.choice(['active', '2024', 'prod'])}'"
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.MAPPING"] = lambda rng: f"{rng.choice(['source_col', 'field_a', 'input'])} -> {rng.choice(['target_col', 'field_b', 'output'])}"
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.SCHEMA_DEF"] = lambda rng: rng.choice(["CREATE TABLE t (id INT PRIMARY KEY)", '{"type": "object", "properties": {}}', "message Msg { string id = 1; }"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.CRON_SPEC"] = gen_cron
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.REGEX"] = gen_regex
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.RETENTION_POLICY"] = lambda rng: rng.choice(["7 days", "30 days", "1 year", "7 years", "indefinite"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.VALIDATION_RULE"] = lambda rng: rng.choice(["NOT NULL", "UNIQUE", "CHECK (val > 0)", "BETWEEN 1 AND 100", "MATCHES '^[A-Z]+'"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.DEFAULT_VALUE"] = lambda rng: rng.choice(["0", "null", "''", "false", "CURRENT_TIMESTAMP", "1"])

    # ── Metadata ──
    g["ICE.METADATA.TIMESTAMP"] = gen_timestamp
    g["ICE.METADATA.RECID"] = gen_recid
    g["ICE.METADATA.STATUS"] = gen_status
    g["ICE.METADATA.CREATED_BY"] = lambda rng: f"{rng.choice(FIRST_NAMES).lower()}.{rng.choice(LAST_NAMES).lower()}"
    g["ICE.METADATA.MODIFIED_BY"] = lambda rng: f"{rng.choice(FIRST_NAMES).lower()}.{rng.choice(LAST_NAMES).lower()}"
    g["ICE.METADATA.CREATED_AT"] = gen_timestamp
    g["ICE.METADATA.MODIFIED_AT"] = gen_timestamp
    g["ICE.METADATA.DELETED_AT"] = lambda rng: gen_timestamp(rng) if rng.random() < 0.2 else ""
    g["ICE.METADATA.VERSION"] = gen_version
    g["ICE.METADATA.SCHEMA"] = lambda rng: rng.choice(["public.users", "analytics.events", "sales.orders", "hr.employees"])
    g["ICE.METADATA.COLUMN_NAME"] = lambda rng: rng.choice(["id", "name", "email", "status", "created_at", "amount", "type"])
    g["ICE.METADATA.DATA_TYPE"] = lambda rng: rng.choice(["VARCHAR", "INTEGER", "BOOLEAN", "TIMESTAMP", "FLOAT", "TEXT", "JSON"])
    g["ICE.METADATA.ENCODING"] = lambda rng: rng.choice(["UTF-8", "ASCII", "Latin-1", "UTF-16", "ISO-8859-1"])
    g["ICE.METADATA.ROW_COUNT"] = gen_count
    g["ICE.METADATA.CHECKSUM"] = gen_hash
    g["ICE.METADATA.PARTITION"] = lambda rng: f"{rng.choice(['dt', 'region', 'tenant'])}={rng.choice(['2024-01', 'us-east', 'acme'])}"
    g["ICE.METADATA.TTL"] = lambda rng: f"{rng.randint(1, 365)} days"
    g["ICE.METADATA.ETL_BATCH"] = lambda rng: f"batch-{_rng_date(rng).strftime('%Y%m%d')}-{rng.randint(1, 99):02d}"
    g["ICE.METADATA.LINEAGE"] = lambda rng: f"{rng.choice(['raw', 'staging', 'curated'])}.{rng.choice(['users', 'orders', 'events'])} -> {rng.choice(['analytics', 'reports', 'exports'])}"
    g["ICE.METADATA.NULLABLE"] = lambda rng: rng.choice(["true", "false", "YES", "NO"])
    g["ICE.METADATA.SOURCE_SYSTEM"] = lambda rng: rng.choice(["Salesforce", "SAP", "Workday", "Snowflake", "Kafka", "PostgreSQL"])
    g["ICE.METADATA.LOAD_TIMESTAMP"] = gen_timestamp
    g["ICE.METADATA.RECORD_TYPE"] = lambda rng: rng.choice(["insert", "update", "delete", "upsert", "snapshot"])
    g["ICE.METADATA.IS_DELETED"] = lambda rng: rng.choice(["false", "false", "false", "true"])
    g["ICE.METADATA.TENANT_ID"] = lambda rng: f"tenant-{rng.randint(1, 50):03d}"

    # ── Contact subtypes (phone, address) ──
    g["ICE.SENSITIVE.PID.CONTACT.PHONE.MOBILE"] = lambda rng: gen_phone(rng)
    g["ICE.SENSITIVE.PID.CONTACT.PHONE.HOME"] = lambda rng: gen_phone(rng)
    g["ICE.SENSITIVE.PID.CONTACT.PHONE.WORK"] = lambda rng: gen_phone(rng)
    g["ICE.SENSITIVE.PID.CONTACT.PHONE.OTHER"] = lambda rng: gen_phone(rng)
    g["ICE.SENSITIVE.PID.CONTACT.PHONE.EXTENSION"] = lambda rng: str(rng.randint(100, 99999))
    g["ICE.SENSITIVE.PID.CONTACT.PHONE.SUBSCRIBER"] = lambda rng: f"{rng.randint(200, 999)}-{rng.randint(1000, 9999)}"
    g["ICE.SENSITIVE.PID.CONTACT.PHONE.PAGER"] = lambda rng: gen_phone(rng)
    g["ICE.SENSITIVE.PID.CONTACT.PHONE.EMERGENCY"] = lambda rng: gen_phone(rng)
    g["ICE.SENSITIVE.PID.CONTACT.ADDRESS.BILLING"] = gen_address
    g["ICE.SENSITIVE.PID.CONTACT.ADDRESS.SHIPPING"] = gen_address
    g["ICE.SENSITIVE.PID.CONTACT.ADDRESS.HOME"] = gen_address
    g["ICE.SENSITIVE.PID.CONTACT.ADDRESS.OFFICE"] = gen_address

    # ── Geographic subtypes ──
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.COUNTRY.BILLING"] = gen_country
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.COUNTRY.SHIPPING"] = gen_country
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.COUNTRY.RESIDENCE"] = gen_country
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.COUNTRY.CITIZENSHIP"] = gen_country
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.REGION.BILLING"] = gen_region
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.REGION.SHIPPING"] = gen_region

    # ── Identity subtypes ──
    g["ICE.SENSITIVE.PID.IDENTITY.NAME.ALIAS"] = lambda rng: rng.choice(FIRST_NAMES) + " " + rng.choice(LAST_NAMES)
    g["ICE.SENSITIVE.PID.IDENTITY.PLATFORM_ID.ORGANIZATION_ID"] = lambda rng: f"ORG-{rng.randint(100000, 999999)}"
    g["ICE.SENSITIVE.PID.IDENTITY.GOVID.TAX_JURISDICTION"] = lambda rng: rng.choice(["US-CA", "US-NY", "US-TX", "US-FL", "DE-BY", "GB-ENG", "FR-IDF", "JP-13"])
    g["ICE.SENSITIVE.PID.IDENTITY.PLATFORM_ID.CONTRACTOR_ID"] = lambda rng: f"CTR-{rng.randint(10000, 99999)}"

    # ── Device identifiers ──
    g["ICE.SENSITIVE.TECHNICAL.DEVID.IMEI"] = lambda rng: "".join(str(rng.randint(0, 9)) for _ in range(15))
    g["ICE.SENSITIVE.TECHNICAL.DEVID.MAC"] = lambda rng: ":".join(f"{rng.randint(0, 255):02x}" for _ in range(6))

    # ── Data lifecycle / quality ──
    g["ICE.METADATA.LIFECYCLE.NULLIFIED"] = lambda rng: rng.choice(["NULL", "VOID", "NULLIFIED", "N/A", "CANCELLED", "INVALID"])
    g["ICE.METADATA.LIFECYCLE.DEPRECATED"] = lambda rng: rng.choice(["DEPRECATED", "OBSOLETE", "RETIRED", "LEGACY", "EOL", "SUPERSEDED"])
    g["ICE.METADATA.LIFECYCLE.MASKED"] = lambda rng: rng.choice(["***MASKED***", "XXXX-XXXX", "[REDACTED]", "****", "##MASKED##"])
    g["ICE.METADATA.LIFECYCLE.HASHED"] = lambda rng: gen_hash(rng)

    # ── Encryption methods ──
    g["ICE.METADATA.ENCRYPTION.E2E"] = lambda rng: rng.choice(["AES-256-GCM", "ChaCha20-Poly1305", "RSA-OAEP-256", "X25519"])
    g["ICE.METADATA.ENCRYPTION.AT_REST"] = lambda rng: rng.choice(["AES-256", "TDE", "LUKS", "BitLocker", "FileVault"])
    g["ICE.METADATA.ENCRYPTION.IN_TRANSIT"] = lambda rng: rng.choice(["TLS 1.3", "TLS 1.2", "mTLS", "IPsec", "WireGuard"])

    # ── Runtime / execution ──
    g["ICE.NONSENSITIVE.DESIGNATIVE.REF.EXECUTABLE"] = lambda rng: f"/usr/{rng.choice(['bin', 'local/bin', 'sbin'])}/{rng.choice(['python3', 'java', 'node', 'nginx', 'postgres'])}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.RUNTIME"] = lambda rng: f"RUN-{rng.randint(10000, 99999)}-{rng.choice(string.ascii_uppercase)}"

    # ── Access control ──
    g["ICE.SENSITIVE.TECHNICAL.ACCESS.TRUSTED_IP"] = gen_ipv4
    g["ICE.SENSITIVE.TECHNICAL.ACCESS.GROUP_ID"] = lambda rng: rng.choice(["CN=Domain Admins", "CN=Engineering", "role-admin", "sg-prod-readers", "arn:aws:iam::123:group/devops"])
    g["ICE.SENSITIVE.TECHNICAL.SECRET"] = lambda rng: f"vault:secret/{rng.choice(['db', 'api', 'service'])}/{rng.choice(['password', 'key', 'token'])}"

    # ── Transaction identifiers ──
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.SUBSCRIPTION_ID"] = lambda rng: f"SUB-{rng.randint(100000, 999999)}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.ORDER_ID"] = lambda rng: f"ORD-{rng.randint(100000, 999999)}"

    # ── User-generated content ──
    g["ICE.SENSITIVE.PID.UGC.FREE_TEXT"] = gen_comment
    g["ICE.SENSITIVE.PID.UGC.BUG_REPORT"] = lambda rng: f"Bug: {rng.choice(['Crash on', 'Error in', 'Slow', 'Missing'])} {rng.choice(['login', 'search', 'checkout', 'upload'])} - {gen_error_message(rng)}"

    # ── OS type ──
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.OS_TYPE"] = lambda rng: rng.choice(["iOS", "Android", "Windows", "macOS", "Linux", "ChromeOS"])

    # ── Legacy compatibility (synth.py core codes) ──
    g["ICE.NONSENSITIVE"] = gen_internal_code
    g["ICE.SENSITIVE.TECHNICAL.URL"] = gen_url
    g["ICE.METADATA.RECID"] = gen_record_id

    return g


# Module-level singleton — built once on import
GENERATORS: dict[str, Callable[[random.Random], str]] = build_generators()
