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
    data/sample/tables/*.csv       — ~25 mixed-domain tables, 100 rows each
    data/sample/ground_truth.json  — column → category mapping
"""

from __future__ import annotations

import csv
import json
import random
import string
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY_PATH = PROJECT_ROOT / "data" / "sample" / "ontology.json"
TABLES_DIR = PROJECT_ROOT / "data" / "sample" / "tables"
GROUND_TRUTH_PATH = PROJECT_ROOT / "data" / "sample" / "ground_truth.json"

ROWS_PER_TABLE = 100
SEED = 42

# ── Value generator registry ─────────────────────────────────────
#
# Each generator takes (rng: random.Random) → str.
# Categories are mapped to generators by code suffix or subtree.

_FIRST_NAMES = [
    "James", "John", "Robert", "Michael", "William", "David", "Richard",
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Susan", "Jessica",
    "Sarah", "Emily", "Amy", "Anna", "Emma", "Sophia", "Olivia", "Liam",
    "Noah", "Ethan", "Mason", "Logan", "Aiden", "Oliver", "Lucas", "Harper",
]
_LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Anderson", "Taylor", "Thomas",
    "Jackson", "White", "Harris", "Martin", "Thompson", "Moore", "Lee",
    "Clark", "Lewis", "Robinson", "Walker", "Young", "King", "Wright",
]
_CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Seattle",
    "Denver", "Boston", "Austin", "Portland", "Miami", "Dallas", "Atlanta",
    "San Francisco", "Toronto", "London", "Paris", "Berlin", "Tokyo", "Sydney",
]
_COUNTRIES = [
    "United States", "Canada", "United Kingdom", "Germany", "France", "Japan",
    "Australia", "Brazil", "India", "Mexico", "Italy", "Spain", "Netherlands",
    "South Korea", "Sweden", "Norway", "Switzerland", "Singapore", "Israel",
]
_COUNTRY_CODES = ["US", "CA", "GB", "DE", "FR", "JP", "AU", "BR", "IN", "MX"]
_DOMAINS = ["gmail.com", "outlook.com", "yahoo.com", "company.com", "acme.org", "example.com"]
_STATES = ["CA", "TX", "NY", "FL", "WA", "IL", "CO", "MA", "GA", "PA", "OH", "NC"]
_STREETS = ["Main St", "Oak Ave", "Maple Dr", "Cedar Ln", "Pine Rd", "Elm St", "Park Blvd"]
_ORGS = [
    "Acme Corp", "TechCo Industries", "Global Solutions", "DataCraft Inc",
    "Quantum Dynamics", "Pacific Research Group", "Alpine Systems",
    "Skyline Partners", "Beacon Analytics", "Vertex Technologies",
    "Meridian Health", "Catalyst Ventures", "Pioneer Software",
]
_PRODUCTS = [
    "Widget Pro", "DataSync 3000", "CloudGuard", "SmartHub X1",
    "FlexPay Platform", "NanoCore SDK", "AuditTrail Plus",
    "Sentinel Monitor", "StreamLine ERP", "VaultKeeper",
]
_STATUSES = ["active", "inactive", "pending", "completed", "failed", "cancelled", "approved", "rejected"]
_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "INR", "BRL"]
_LANGUAGES = ["English", "Spanish", "French", "German", "Japanese", "Chinese", "Portuguese", "Korean", "Arabic", "Hindi"]
_LANG_CODES = ["en", "es", "fr", "de", "ja", "zh", "pt", "ko", "ar", "hi"]
_COLORS = ["Red", "Blue", "Green", "Black", "White", "Yellow", "Purple", "Orange", "Gray", "Brown"]
_SIZES = ["XS", "S", "M", "L", "XL", "XXL"]
_PLATFORMS = ["Windows", "macOS", "Linux", "iOS", "Android", "Chrome OS"]
_BROWSERS = ["Chrome", "Firefox", "Safari", "Edge", "Opera", "Brave"]
_DEVICE_TYPES = ["Desktop", "Mobile", "Tablet", "IoT", "Wearable", "Smart TV"]
_DEPARTMENTS = ["Engineering", "Sales", "Marketing", "Finance", "HR", "Operations", "Legal", "Support"]
_INDUSTRIES = ["Technology", "Healthcare", "Finance", "Retail", "Manufacturing", "Energy", "Education"]
_ROLES = ["Software Engineer", "Product Manager", "Data Analyst", "Designer", "VP Engineering", "CTO", "Sales Rep"]
_SEVERITIES = ["Critical", "High", "Medium", "Low", "Info"]
_SENTIMENTS = ["Positive", "Negative", "Neutral", "Mixed"]
_CHANNELS = ["Web", "Mobile", "Email", "Phone", "In-Store", "Social Media", "Chat"]
_PAYMENT_METHODS = ["Credit Card", "Debit Card", "Wire Transfer", "ACH", "PayPal", "Cash", "Check"]
_SHIPPING_METHODS = ["Standard", "Express", "Overnight", "Freight", "Same Day", "Economy"]
_MIME_TYPES = ["application/json", "text/csv", "text/html", "image/png", "application/pdf", "application/xml"]
_PROTOCOLS = ["HTTP", "HTTPS", "FTP", "SSH", "SMTP", "gRPC", "WebSocket", "MQTT"]


def _rng_date(rng: random.Random, start_year: int = 2018, end_year: int = 2026) -> datetime:
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 28)
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def _gen_email(rng): return f"{rng.choice(_FIRST_NAMES).lower()}.{rng.choice(_LAST_NAMES).lower()}{rng.randint(1,99) if rng.random()<0.3 else ''}@{rng.choice(_DOMAINS)}"
def _gen_phone(rng): return f"+1-{rng.randint(200,999)}-{rng.randint(200,999)}-{rng.randint(1000,9999)}"
def _gen_address(rng): return f"{rng.randint(100,9999)} {rng.choice(_STREETS)}, {rng.choice(_CITIES)}, {rng.choice(_STATES)} {rng.randint(10000,99999)}"
def _gen_fullname(rng): return f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
def _gen_firstname(rng): return rng.choice(_FIRST_NAMES)
def _gen_lastname(rng): return rng.choice(_LAST_NAMES)
def _gen_dob(rng): d = _rng_date(rng, 1950, 2005); return d.strftime("%Y-%m-%d")
def _gen_ssn(rng): return f"{rng.randint(100,899)}-{rng.randint(10,99)}-{rng.randint(1000,9999)}"
def _gen_pan(rng): d = "".join(str(rng.randint(0,9)) for _ in range(16)); return f"{d[:4]}-{d[4:8]}-{d[8:12]}-{d[12:]}"
def _gen_ban(rng): return "".join(str(rng.randint(0,9)) for _ in range(rng.choice([8,10,12])))
def _gen_amount(rng): return f"${round(rng.uniform(0.50, 99999.99), 2):,.2f}"
def _gen_ipv4(rng): return ".".join(str(rng.randint(1,254)) for _ in range(4))
def _gen_uuid(rng): h = "0123456789abcdef"; return "-".join(["".join(rng.choices(h,k=8)),"".join(rng.choices(h,k=4)),"4"+"".join(rng.choices(h,k=3)),rng.choice("89ab")+"".join(rng.choices(h,k=3)),"".join(rng.choices(h,k=12))])
def _gen_url(rng): return f"https://{rng.choice(['example.com','acme.org','data.io','api.svc.com'])}/{'/' .join(rng.choices(['users','api','v2','data','docs'],k=rng.randint(1,3)))}"
def _gen_timestamp(rng): d = _rng_date(rng); return d.strftime("%Y-%m-%d %H:%M:%S")
def _gen_date(rng): return _rng_date(rng).strftime("%Y-%m-%d")
def _gen_datetime(rng): d = _rng_date(rng); return d.strftime("%Y-%m-%dT%H:%M:%SZ")
def _gen_time(rng): return f"{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:{rng.randint(0,59):02d}"
def _gen_year(rng): return str(rng.randint(1990, 2026))
def _gen_month(rng): return rng.choice(["January","February","March","April","May","June","July","August","September","October","November","December"])
def _gen_country(rng): return rng.choice(_COUNTRIES)
def _gen_country_code(rng): return rng.choice(_COUNTRY_CODES)
def _gen_city(rng): return rng.choice(_CITIES)
def _gen_region(rng): return rng.choice(_STATES)
def _gen_org(rng): return rng.choice(_ORGS)
def _gen_product(rng): return rng.choice(_PRODUCTS)
def _gen_status(rng): return rng.choice(_STATUSES)
def _gen_integer(rng): return str(rng.randint(0, 10000))
def _gen_decimal(rng): return f"{round(rng.uniform(-1000, 10000), 4)}"
def _gen_percentage(rng): return f"{round(rng.uniform(0, 100), 2)}%"
def _gen_score(rng): return f"{round(rng.uniform(0, 10), 2)}"
def _gen_count(rng): return str(rng.randint(0, 100000))
def _gen_price(rng): return f"{round(rng.uniform(0.99, 9999.99), 2)}"
def _gen_boolean(rng): return rng.choice(["true", "false", "yes", "no", "1", "0"])
def _gen_currency_code(rng): return rng.choice(_CURRENCIES)
def _gen_language(rng): return rng.choice(_LANGUAGES)
def _gen_lang_code(rng): return rng.choice(_LANG_CODES)
def _gen_color(rng): return rng.choice(_COLORS)
def _gen_size(rng): return rng.choice(_SIZES)
def _gen_platform(rng): return rng.choice(_PLATFORMS)
def _gen_browser(rng): return rng.choice(_BROWSERS)
def _gen_device_type(rng): return rng.choice(_DEVICE_TYPES)
def _gen_department(rng): return rng.choice(_DEPARTMENTS)
def _gen_industry(rng): return rng.choice(_INDUSTRIES)
def _gen_role(rng): return rng.choice(_ROLES)
def _gen_severity(rng): return rng.choice(_SEVERITIES)
def _gen_sentiment(rng): return rng.choice(_SENTIMENTS)
def _gen_channel(rng): return rng.choice(_CHANNELS)
def _gen_payment_method(rng): return rng.choice(_PAYMENT_METHODS)
def _gen_shipping(rng): return rng.choice(_SHIPPING_METHODS)
def _gen_mime(rng): return rng.choice(_MIME_TYPES)
def _gen_protocol(rng): return rng.choice(_PROTOCOLS)
def _gen_latitude(rng): return f"{round(rng.uniform(-90, 90), 6)}"
def _gen_longitude(rng): return f"{round(rng.uniform(-180, 180), 6)}"
def _gen_elevation(rng): return f"{rng.randint(-100, 8848)}"
def _gen_temperature(rng): return f"{round(rng.uniform(-40, 50), 1)}"
def _gen_weight(rng): return f"{round(rng.uniform(0.1, 1000), 2)}"
def _gen_distance(rng): return f"{round(rng.uniform(0.1, 10000), 1)}"
def _gen_duration(rng): return f"{rng.randint(1, 86400)}"
def _gen_age(rng): return str(rng.randint(0, 120))
def _gen_recid(rng): return str(rng.randint(100000, 999999))
def _gen_version(rng): return f"{rng.randint(0,5)}.{rng.randint(0,20)}.{rng.randint(0,99)}"
def _gen_hash(rng): return "".join(rng.choices("0123456789abcdef", k=64))
def _gen_fax(rng): return f"+1-{rng.randint(200,999)}-{rng.randint(200,999)}-{rng.randint(1000,9999)}"
def _gen_social(rng): return f"@{rng.choice(_FIRST_NAMES).lower()}{rng.choice(_LAST_NAMES).lower()}{rng.randint(1,999)}"
def _gen_cvv(rng): return "".join(str(rng.randint(0,9)) for _ in range(rng.choice([3,4])))
def _gen_expiry(rng): return f"{rng.randint(1,12):02d}/{rng.randint(24,30)}"
def _gen_salary(rng): return f"${rng.randint(30000, 250000):,}"
def _gen_hostname(rng): return f"{rng.choice(['web','api','db','cache','app'])}-{rng.randint(1,20)}.{rng.choice(['prod','staging','dev'])}.{rng.choice(['acme.com','example.net'])}"
def _gen_api_key(rng): return f"sk_{''.join(rng.choices(string.ascii_letters + string.digits, k=32))}"
def _gen_session_token(rng): return f"sess_{''.join(rng.choices(string.ascii_letters + string.digits, k=40))}"
def _gen_regex(rng): return rng.choice([r"^\d{3}-\d{2}-\d{4}$", r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", r"^[A-Z]{2}\d{6}$"])
def _gen_cron(rng): return f"{rng.randint(0,59)} {rng.randint(0,23)} * * {rng.choice(['*','1-5','0,6'])}"
def _gen_ticker(rng): return "".join(rng.choices(string.ascii_uppercase, k=rng.randint(2,4)))
def _gen_isbn(rng): return f"978-{rng.randint(0,9)}-{rng.randint(10000,99999)}-{rng.randint(100,999)}-{rng.randint(0,9)}"
def _gen_doi(rng): return f"10.{rng.randint(1000,9999)}/{rng.choice(string.ascii_lowercase)}{rng.randint(100,999)}.{rng.randint(1,50)}"
def _gen_postal_code(rng): return f"{rng.randint(10000,99999)}" if rng.random()<0.5 else f"{rng.choice(string.ascii_uppercase)}{rng.choice(string.ascii_uppercase)}{rng.randint(1,9)} {rng.randint(1,9)}{rng.choice(string.ascii_uppercase)}{rng.choice(string.ascii_uppercase)}"
def _gen_iata(rng): return "".join(rng.choices(string.ascii_uppercase, k=3))
def _gen_sku(rng): return f"{''.join(rng.choices(string.ascii_uppercase, k=3))}-{rng.randint(1000,9999)}"
def _gen_coordinates(rng): return f"{round(rng.uniform(-90,90),6)}, {round(rng.uniform(-180,180),6)}"
def _gen_quarter(rng): return f"Q{rng.randint(1,4)} {rng.randint(2020,2026)}"
def _gen_day_of_week(rng): return rng.choice(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])


def _gen_description(rng):
    subjects = ["The system", "This record", "The product", "The service", "The customer account"]
    verbs = ["includes", "represents", "tracks", "monitors", "manages"]
    objects = ["data processing activities", "customer interactions", "operational metrics", "compliance records", "transaction histories"]
    return f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)}."


def _gen_comment(rng):
    comments = [
        "Reviewed and approved.", "Needs follow-up next quarter.",
        "Updated per client request.", "Flagged for review.",
        "Automated processing complete.", "Escalated to management.",
        "Resolved — no further action.", "Pending verification.",
    ]
    return rng.choice(comments)


def _gen_keywords(rng):
    words = ["data", "analytics", "cloud", "security", "compliance", "automation",
             "integration", "reporting", "performance", "monitoring", "scalability"]
    return ", ".join(rng.sample(words, rng.randint(2, 5)))


def _gen_log_message(rng):
    levels = ["INFO", "DEBUG", "WARN", "ERROR"]
    msgs = ["Request processed in {}ms", "Connection established to {}", "Cache miss for key {}",
            "Retry attempt {} of 3", "Health check passed", "Batch {} completed"]
    return f"[{rng.choice(levels)}] {rng.choice(msgs).format(rng.randint(1,9999))}"


def _gen_error_message(rng):
    errors = [
        "NullPointerException at line {}", "Connection timeout after {}ms",
        "FileNotFoundError: {}", "ValueError: invalid literal for int()",
        "PermissionDenied: insufficient privileges", "HTTP 503 Service Unavailable",
    ]
    return rng.choice(errors).format(rng.randint(1, 9999))


def _gen_filepath(rng):
    dirs = ["data", "logs", "config", "exports", "uploads", "tmp"]
    exts = [".csv", ".json", ".parquet", ".log", ".yaml", ".xml"]
    return f"/var/{rng.choice(dirs)}/{rng.choice(['batch','daily','export'])}_{rng.randint(1,999)}{rng.choice(exts)}"


# ── Generator dispatch ───────────────────────────────────────────
#
# Maps leaf category codes to generator functions. Uses suffix matching
# for broad coverage — specific codes override when needed.

_GENERATORS: dict[str, callable] = {}


def _register_generators():
    """Build the code → generator mapping."""
    g = _GENERATORS

    # ── Contact ──
    g["ICE.SENSITIVE.PID.CONTACT.EMAIL"] = _gen_email
    g["ICE.SENSITIVE.PID.CONTACT.PHONE"] = _gen_phone
    g["ICE.SENSITIVE.PID.CONTACT.ADDRESS"] = _gen_address
    g["ICE.SENSITIVE.PID.CONTACT.FAX"] = _gen_fax
    g["ICE.SENSITIVE.PID.CONTACT.SOCIAL_MEDIA"] = _gen_social
    g["ICE.SENSITIVE.PID.CONTACT.MESSENGER"] = lambda rng: f"{rng.choice(_FIRST_NAMES).lower()}_{rng.choice(_LAST_NAMES).lower()}"

    # ── Identity ──
    g["ICE.SENSITIVE.PID.IDENTITY.FULLNAME"] = _gen_fullname
    g["ICE.SENSITIVE.PID.IDENTITY.FIRST_NAME"] = _gen_firstname
    g["ICE.SENSITIVE.PID.IDENTITY.LAST_NAME"] = _gen_lastname
    g["ICE.SENSITIVE.PID.IDENTITY.MIDDLE_NAME"] = lambda rng: rng.choice(string.ascii_uppercase)
    g["ICE.SENSITIVE.PID.IDENTITY.MAIDEN_NAME"] = _gen_lastname
    g["ICE.SENSITIVE.PID.IDENTITY.DOB"] = _gen_dob
    g["ICE.SENSITIVE.PID.IDENTITY.GOVID"] = _gen_ssn
    g["ICE.SENSITIVE.PID.IDENTITY.GENDER_PII"] = lambda rng: rng.choice(["Male", "Female", "Non-binary", "Other"])
    g["ICE.SENSITIVE.PID.IDENTITY.NATIONALITY"] = _gen_country
    g["ICE.SENSITIVE.PID.IDENTITY.ETHNICITY"] = lambda rng: rng.choice(["Asian", "Black", "Hispanic", "White", "Mixed", "Other"])
    g["ICE.SENSITIVE.PID.IDENTITY.BIRTH_PLACE"] = _gen_city
    g["ICE.SENSITIVE.PID.IDENTITY.MARITAL_STATUS"] = lambda rng: rng.choice(["Single", "Married", "Divorced", "Widowed"])
    g["ICE.SENSITIVE.PID.IDENTITY.PHOTO"] = lambda rng: f"photo_{_gen_uuid(rng)[:8]}.jpg"
    g["ICE.SENSITIVE.PID.IDENTITY.SIGNATURE"] = lambda rng: f"sig_{_gen_hash(rng)[:16]}"
    g["ICE.SENSITIVE.PID.IDENTITY.VOICEPRINT"] = lambda rng: f"voice_{_gen_hash(rng)[:16]}.wav"

    # ── Financial ──
    g["ICE.SENSITIVE.PID.FINANCIAL.PAN"] = _gen_pan
    g["ICE.SENSITIVE.PID.FINANCIAL.BAN"] = _gen_ban
    g["ICE.SENSITIVE.PID.FINANCIAL.TXNAMT"] = _gen_amount
    g["ICE.SENSITIVE.PID.FINANCIAL.CVV"] = _gen_cvv
    g["ICE.SENSITIVE.PID.FINANCIAL.EXPIRY"] = _gen_expiry
    g["ICE.SENSITIVE.PID.FINANCIAL.TAX_ID"] = lambda rng: f"{rng.randint(10,99)}-{rng.randint(1000000,9999999)}"
    g["ICE.SENSITIVE.PID.FINANCIAL.SALARY"] = _gen_salary
    g["ICE.SENSITIVE.PID.FINANCIAL.CREDIT_SCORE"] = lambda rng: str(rng.randint(300, 850))
    g["ICE.SENSITIVE.PID.FINANCIAL.INVESTMENT"] = lambda rng: f"INV-{rng.randint(100000,999999)}"
    g["ICE.SENSITIVE.PID.FINANCIAL.INSURANCE_ID"] = lambda rng: f"INS-{rng.randint(100000,999999)}"
    g["ICE.SENSITIVE.PID.FINANCIAL.ROUTING_NUM"] = lambda rng: "".join(str(rng.randint(0,9)) for _ in range(9))
    g["ICE.SENSITIVE.PID.FINANCIAL.CRYPTO_ADDR"] = lambda rng: f"0x{''.join(rng.choices('0123456789abcdef', k=40))}"

    # ── Health ──
    g["ICE.SENSITIVE.PID.HEALTH.DIAGNOSIS"] = lambda rng: rng.choice(["Hypertension", "Type 2 Diabetes", "Asthma", "Migraine", "Arthritis", "Anxiety", "Depression"])
    g["ICE.SENSITIVE.PID.HEALTH.PRESCRIPTION"] = lambda rng: rng.choice(["Lisinopril 10mg", "Metformin 500mg", "Albuterol 90mcg", "Ibuprofen 200mg", "Omeprazole 20mg"])
    g["ICE.SENSITIVE.PID.HEALTH.BLOOD_TYPE"] = lambda rng: rng.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
    g["ICE.SENSITIVE.PID.HEALTH.ALLERGY"] = lambda rng: rng.choice(["Penicillin", "Peanuts", "Latex", "Shellfish", "Pollen", "None"])
    g["ICE.SENSITIVE.PID.HEALTH.MEDICAL_RECORD"] = lambda rng: f"MRN-{rng.randint(100000,999999)}"
    g["ICE.SENSITIVE.PID.HEALTH.INSURANCE_CLAIM"] = lambda rng: f"CLM-{rng.randint(100000,999999)}"
    g["ICE.SENSITIVE.PID.HEALTH.LAB_RESULT"] = lambda rng: f"{rng.choice(['Glucose','Cholesterol','Hemoglobin','WBC'])}: {round(rng.uniform(50,300),1)} {rng.choice(['mg/dL','g/dL','K/uL'])}"
    g["ICE.SENSITIVE.PID.HEALTH.VITAL_SIGN"] = lambda rng: f"BP {rng.randint(90,180)}/{rng.randint(60,110)} HR {rng.randint(55,120)}"
    g["ICE.SENSITIVE.PID.HEALTH.PROCEDURE"] = lambda rng: rng.choice(["Blood Draw", "X-Ray Chest", "MRI Brain", "ECG", "Colonoscopy", "Physical Exam"])
    g["ICE.SENSITIVE.PID.HEALTH.DISABILITY"] = lambda rng: rng.choice(["None", "Visual", "Hearing", "Mobility", "Cognitive"])

    # ── Technical ──
    g["ICE.SENSITIVE.TECHNICAL.IPADDR"] = _gen_ipv4
    g["ICE.SENSITIVE.TECHNICAL.DEVID"] = _gen_uuid
    g["ICE.SENSITIVE.TECHNICAL.URL_PII"] = lambda rng: f"https://example.com/users/{_gen_uuid(rng)[:8]}/profile"
    g["ICE.SENSITIVE.TECHNICAL.HOSTNAME"] = _gen_hostname
    g["ICE.SENSITIVE.TECHNICAL.API_KEY"] = _gen_api_key
    g["ICE.SENSITIVE.TECHNICAL.PASSWORD_HASH"] = lambda rng: f"$2b$12${''.join(rng.choices(string.ascii_letters+string.digits, k=53))}"
    g["ICE.SENSITIVE.TECHNICAL.SESSION_TOKEN"] = _gen_session_token
    g["ICE.SENSITIVE.TECHNICAL.COOKIE"] = lambda rng: f"_ga={''.join(rng.choices(string.digits, k=10))}.{rng.randint(1000000,9999999)}"
    g["ICE.SENSITIVE.TECHNICAL.SSH_KEY"] = lambda rng: f"ssh-rsa {''.join(rng.choices(string.ascii_letters+string.digits+'+/', k=44))}="
    g["ICE.SENSITIVE.TECHNICAL.CERTIFICATE"] = lambda rng: f"-----BEGIN CERTIFICATE-----\n{''.join(rng.choices(string.ascii_letters+string.digits+'+/', k=64))}\n-----END CERTIFICATE-----"
    g["ICE.SENSITIVE.TECHNICAL.DATABASE_CONN"] = lambda rng: f"postgresql://user:****@{rng.choice(['db-prod','db-staging'])}.internal:5432/{rng.choice(['main','analytics'])}"

    # ── Business ──
    g["ICE.SENSITIVE.BUSINESS.TRADE_SECRET"] = lambda rng: f"[CONFIDENTIAL] Formula #{rng.randint(100,999)}"
    g["ICE.SENSITIVE.BUSINESS.CONTRACT_VALUE"] = lambda rng: f"${rng.randint(10000, 10000000):,}"
    g["ICE.SENSITIVE.BUSINESS.CUSTOMER_LIST"] = lambda rng: ", ".join(rng.sample(_ORGS, rng.randint(2,5)))
    g["ICE.SENSITIVE.BUSINESS.PRICING_STRATEGY"] = lambda rng: rng.choice(["Cost-plus 25%", "Market-based", "Value pricing", "Penetration", "Premium"])
    g["ICE.SENSITIVE.BUSINESS.MARKET_INTEL"] = lambda rng: f"Competitor {rng.choice(_ORGS)} launched {rng.choice(_PRODUCTS)}"
    g["ICE.SENSITIVE.BUSINESS.INTERNAL_MEMO"] = lambda rng: f"[INTERNAL] Q{rng.randint(1,4)} strategy review pending board approval"
    g["ICE.SENSITIVE.BUSINESS.LEGAL_HOLD"] = lambda rng: f"HOLD-{rng.randint(2020,2026)}-{rng.randint(100,999)}"
    g["ICE.SENSITIVE.BUSINESS.AUDIT_FINDING"] = lambda rng: rng.choice(["Non-conformance: access control", "Observation: backup schedule", "Major finding: data retention"])
    g["ICE.SENSITIVE.BUSINESS.FINANCIAL_FORECAST"] = lambda rng: f"Q{rng.randint(1,4)} {rng.randint(2024,2027)}: ${rng.randint(1,100)}M projected"
    g["ICE.SENSITIVE.BUSINESS.STRATEGIC_PLAN"] = lambda rng: rng.choice(["Expand APAC market", "Launch SaaS tier", "Acquire competitor", "IPO readiness"])
    g["ICE.SENSITIVE.BUSINESS.EMPLOYEE_RECORD"] = lambda rng: f"EMP-{rng.randint(10000,99999)}"
    g["ICE.SENSITIVE.BUSINESS.PERFORMANCE_REVIEW"] = lambda rng: rng.choice(["Exceeds expectations", "Meets expectations", "Needs improvement", "Outstanding"])

    # ── Designative ──
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.PERSON"] = _gen_fullname
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.ORG"] = _gen_org
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.PRODUCT"] = _gen_product
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.SCIENTIFIC"] = lambda rng: rng.choice(["Homo sapiens", "E. coli", "NaCl", "H2O", "C6H12O6", "Fe2O3"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.GEO"] = _gen_country
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.EVENT"] = lambda rng: f"{rng.choice(['Annual','Global','Tech','Data'])} {rng.choice(['Summit','Conference','Forum','Expo'])} {rng.randint(2020,2026)}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.NAME.PROJECT"] = lambda rng: f"Project {rng.choice(['Alpha','Beta','Phoenix','Atlas','Horizon','Zenith'])}"
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

    # ── Codes ──
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.ID"] = _gen_uuid
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.ABBREV"] = lambda rng: "".join(rng.choices(string.ascii_uppercase, k=rng.randint(2,5)))
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.POSTAL"] = _gen_postal_code
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.ISO_COUNTRY"] = _gen_country_code
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.ISO_CURRENCY"] = _gen_currency_code
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.ISO_LANGUAGE"] = _gen_lang_code
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.IATA"] = _gen_iata
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.NAICS"] = lambda rng: str(rng.randint(110000, 999999))
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.TICKER"] = _gen_ticker
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.DOI"] = _gen_doi
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.ISBN"] = _gen_isbn
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.ISSN"] = lambda rng: f"{rng.randint(1000,9999)}-{rng.randint(1000,9999)}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.SKU"] = _gen_sku
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.FIPS"] = lambda rng: f"{rng.randint(1,56):02d}{rng.randint(1,999):03d}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.LEI"] = lambda rng: "".join(rng.choices(string.ascii_uppercase + string.digits, k=20))
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.DUNS"] = lambda rng: "".join(str(rng.randint(0,9)) for _ in range(9))
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.MIME_TYPE"] = _gen_mime
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.PHONE_CODE"] = lambda rng: f"+{rng.choice([1,44,49,33,81,86,91,61,55])}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.CAS"] = lambda rng: f"{rng.randint(50,99999)}-{rng.randint(10,99)}-{rng.randint(0,9)}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.VIN"] = lambda rng: "".join(rng.choices(string.ascii_uppercase + string.digits, k=17))
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.CUSIP"] = lambda rng: "".join(rng.choices(string.ascii_uppercase + string.digits, k=9))
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.CIK"] = lambda rng: str(rng.randint(100000, 9999999)).zfill(10)
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.GTIN"] = lambda rng: "".join(str(rng.randint(0,9)) for _ in range(13))
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.HASH_ID"] = _gen_hash
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.SEMANTIC_VERSION"] = _gen_version
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.PLATE_NUMBER"] = lambda rng: f"{''.join(rng.choices(string.ascii_uppercase, k=3))}-{rng.randint(1000,9999)}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.CODE.FLIGHT_NUMBER"] = lambda rng: f"{''.join(rng.choices(string.ascii_uppercase, k=2))}{rng.randint(100,9999)}"

    # ── Geographic ──
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.COUNTRY"] = _gen_country
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.REGION"] = _gen_region
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.CITY"] = _gen_city
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.LOCATION"] = lambda rng: f"{rng.choice(_CITIES)}, {rng.choice(_COUNTRIES)}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.COORDINATES"] = _gen_coordinates
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.CONTINENT"] = lambda rng: rng.choice(["North America", "Europe", "Asia", "Africa", "South America", "Oceania", "Antarctica"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.DISTRICT"] = lambda rng: rng.choice(["Downtown", "Midtown", "West End", "East Side", "North Shore", "South Bay"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.TIMEZONE"] = lambda rng: rng.choice(["UTC", "EST", "PST", "CET", "JST", "America/New_York", "Europe/London"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.ADDRESS_LINE"] = lambda rng: f"{rng.randint(1,9999)} {rng.choice(_STREETS)}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.LANDMARK"] = lambda rng: rng.choice(["Central Park", "Eiffel Tower", "Golden Gate Bridge", "Big Ben", "Statue of Liberty"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.AIRPORT"] = lambda rng: rng.choice(["JFK", "LAX", "ORD", "LHR", "NRT", "SFO", "CDG", "SIN"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.GEO.PORT"] = lambda rng: rng.choice(["Port of LA", "Rotterdam", "Shanghai", "Singapore", "Hamburg", "Busan"])

    # ── References ──
    g["ICE.NONSENSITIVE.DESIGNATIVE.REF.CITATION"] = lambda rng: f"{rng.choice(_LAST_NAMES)} et al. ({rng.randint(2000,2026)})"
    g["ICE.NONSENSITIVE.DESIGNATIVE.REF.VERSION"] = _gen_version
    g["ICE.NONSENSITIVE.DESIGNATIVE.REF.SOURCE"] = lambda rng: rng.choice(["Bloomberg", "Reuters", "Census Bureau", "WHO", "CDC", "Eurostat"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.REF.URL"] = _gen_url
    g["ICE.NONSENSITIVE.DESIGNATIVE.REF.FILEPATH"] = _gen_filepath

    # ── Other designative ──
    g["ICE.NONSENSITIVE.DESIGNATIVE.TITLE"] = lambda rng: f"{rng.choice(['Annual','Quarterly','Technical','Executive'])} {rng.choice(['Report','Review','Summary','Analysis'])} {rng.randint(2020,2026)}"
    g["ICE.NONSENSITIVE.DESIGNATIVE.LABEL"] = lambda rng: rng.choice(["high-priority", "needs-review", "approved", "deprecated", "experimental", "stable"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.BOOLEAN"] = _gen_boolean
    g["ICE.NONSENSITIVE.DESIGNATIVE.EMAIL_DOMAIN"] = lambda rng: rng.choice(_DOMAINS)
    g["ICE.NONSENSITIVE.DESIGNATIVE.DOMAIN_NAME"] = lambda rng: rng.choice(["example.com", "acme.org", "data.io", "company.net", "service.cloud"])
    g["ICE.NONSENSITIVE.DESIGNATIVE.PHONE_FORMAT"] = lambda rng: rng.choice(["+1-XXX-XXX-XXXX", "+44-XXXX-XXXXXX", "+81-XX-XXXX-XXXX"])

    # ── Descriptive: Text ──
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.DESCRIPTION"] = _gen_description
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.COMMENT"] = _gen_comment
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.ABSTRACT"] = lambda rng: f"This paper presents {rng.choice(['a novel approach','an analysis','a framework'])} for {rng.choice(['data classification','entity resolution','anomaly detection'])}."
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.DEFINITION"] = lambda rng: f"A {rng.choice(['systematic','formal','operational'])} definition of {rng.choice(['compliance','risk','data quality'])}."
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.BODY"] = lambda rng: f"{'  '.join(_gen_description(rng) for _ in range(rng.randint(2,4)))}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.KEYWORDS"] = _gen_keywords
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.INSTRUCTION"] = lambda rng: f"Step {rng.randint(1,5)}: {rng.choice(['Configure','Verify','Initialize','Deploy','Monitor'])} the {rng.choice(['system','service','pipeline','module'])}."
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.REVIEW"] = lambda rng: rng.choice(["Great product!", "Needs improvement.", "Average experience.", "Excellent service.", "Not recommended."])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.LOG_MESSAGE"] = _gen_log_message
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.ERROR_MESSAGE"] = _gen_error_message
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.QUERY"] = lambda rng: f"SELECT * FROM {rng.choice(['users','orders','products'])} WHERE {rng.choice(['status','type','id'])} = '{rng.choice(['active','pending','1'])}'"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.TRANSLATION"] = lambda rng: rng.choice(["Bonjour", "Hola", "Guten Tag", "Konnichiwa", "Namaste"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.HEADLINE"] = lambda rng: f"{rng.choice(['Breaking','New','Updated'])}: {rng.choice(['Product Launch','Policy Change','Partnership'])}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.CAPTION"] = lambda rng: f"Figure {rng.randint(1,20)}: {rng.choice(['Overview','Comparison','Distribution','Timeline'])}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.ADDRESS_TEXT"] = lambda rng: f"{rng.randint(1,9999)} {rng.choice(_STREETS)}, {rng.choice(_CITIES)}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.BIOGRAPHY"] = lambda rng: f"{rng.choice(_FIRST_NAMES)} is a {rng.choice(_ROLES)} with {rng.randint(1,30)} years of experience."
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.CHANGELOG"] = lambda rng: f"v{_gen_version(rng)}: {rng.choice(['Fixed bug in','Added support for','Improved','Removed'])} {rng.choice(['authentication','caching','logging','API'])}."
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.RECIPE"] = lambda rng: f"Mix {rng.randint(1,3)} cups {rng.choice(['flour','sugar','rice'])} with {rng.randint(1,2)} tbsp {rng.choice(['oil','butter','water'])}."

    # ── Descriptive: Categorical ──
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.TYPE"] = lambda rng: rng.choice(["Standard", "Premium", "Enterprise", "Free", "Trial", "Basic"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.CATEGORY"] = lambda rng: rng.choice(["Electronics", "Clothing", "Food", "Services", "Software", "Hardware"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.RANK"] = lambda rng: str(rng.randint(1, 100))
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.LANGUAGE"] = _gen_language
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.GENDER"] = lambda rng: rng.choice(["Male", "Female", "Unisex"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.COLOR"] = _gen_color
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.SIZE"] = _gen_size
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.STATUS"] = _gen_status
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.INDUSTRY"] = _gen_industry
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.DEPARTMENT"] = _gen_department
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.ROLE"] = _gen_role
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.FORMAT"] = lambda rng: rng.choice(["CSV", "JSON", "XML", "Parquet", "Avro", "YAML"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.PROTOCOL"] = _gen_protocol
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.BOOLEAN_ENUM"] = lambda rng: rng.choice(["Active", "Inactive", "Enabled", "Disabled"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.SEVERITY"] = _gen_severity
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.SENTIMENT"] = _gen_sentiment
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.LIFECYCLE"] = lambda rng: rng.choice(["Draft", "In Review", "Published", "Archived", "Deprecated"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.CHANNEL"] = _gen_channel
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.PLATFORM"] = _gen_platform
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.BROWSER"] = _gen_browser
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.DEVICE_TYPE"] = _gen_device_type
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.CONTENT_TYPE"] = lambda rng: rng.choice(["Article", "Video", "Podcast", "Image", "Infographic"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.PAYMENT_METHOD"] = _gen_payment_method
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.SHIPPING_METHOD"] = _gen_shipping
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.EDUCATION_LEVEL"] = lambda rng: rng.choice(["High School", "Bachelor's", "Master's", "PhD", "Associate's"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.RELATIONSHIP"] = lambda rng: rng.choice(["Parent", "Child", "Sibling", "Spouse", "Manager", "Peer"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.PERMISSION_LEVEL"] = lambda rng: rng.choice(["Admin", "Editor", "Viewer", "Owner", "Guest"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.RISK_LEVEL"] = lambda rng: rng.choice(["Critical", "High", "Medium", "Low", "Negligible"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.APPROVAL_STATUS"] = lambda rng: rng.choice(["Approved", "Pending", "Rejected", "In Review"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.CURRENCY_NAME"] = lambda rng: rng.choice(["US Dollar", "Euro", "British Pound", "Japanese Yen", "Swiss Franc"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.UNIT"] = lambda rng: rng.choice(["kg", "lb", "m", "ft", "L", "gal", "°C", "°F"])

    # ── Descriptive: Measurement ──
    for code, gen in [
        ("LENGTH", lambda rng: f"{round(rng.uniform(0.1,1000),2)}"),
        ("AREA", lambda rng: f"{round(rng.uniform(1,100000),1)}"),
        ("VOLUME", lambda rng: f"{round(rng.uniform(0.1,10000),2)}"),
        ("WEIGHT", _gen_weight),
        ("TEMPERATURE", _gen_temperature),
        ("SPEED", lambda rng: f"{round(rng.uniform(0,300),1)}"),
        ("PRESSURE", lambda rng: f"{round(rng.uniform(0,2000),1)}"),
        ("FREQUENCY", lambda rng: f"{round(rng.uniform(0.1,10000),2)}"),
        ("ENERGY", lambda rng: f"{round(rng.uniform(0,100000),1)}"),
        ("POWER", lambda rng: f"{round(rng.uniform(0,50000),1)}"),
        ("DENSITY", lambda rng: f"{round(rng.uniform(0.1,20),3)}"),
        ("CONCENTRATION", lambda rng: f"{round(rng.uniform(0,1000),2)}"),
        ("PERCENTAGE", _gen_percentage),
        ("SCORE", _gen_score),
        ("COUNT", _gen_count),
        ("DURATION", _gen_duration),
        ("AGE", _gen_age),
        ("PRICE", _gen_price),
        ("REVENUE", lambda rng: f"${rng.randint(1000,10000000):,}"),
        ("BUDGET", lambda rng: f"${rng.randint(1000,5000000):,}"),
        ("MARKET_VALUE", lambda rng: f"${rng.randint(100000,1000000000):,}"),
        ("EXCHANGE_RATE", lambda rng: f"{round(rng.uniform(0.5,150),4)}"),
        ("LATITUDE", _gen_latitude),
        ("LONGITUDE", _gen_longitude),
        ("ELEVATION", _gen_elevation),
        ("DISTANCE", _gen_distance),
        ("BANDWIDTH", lambda rng: f"{rng.randint(1,10000)}"),
        ("STORAGE", lambda rng: f"{rng.randint(1,1000000)}"),
        ("LATENCY", lambda rng: f"{rng.randint(1,5000)}"),
        ("UPTIME", lambda rng: f"{round(rng.uniform(95,100),4)}"),
        ("CPU_USAGE", lambda rng: f"{round(rng.uniform(0,100),1)}"),
        ("MEMORY_USAGE", lambda rng: f"{round(rng.uniform(0,100),1)}"),
        ("VOLTAGE", lambda rng: f"{round(rng.uniform(0,480),1)}"),
        ("CURRENT", lambda rng: f"{round(rng.uniform(0,100),2)}"),
        ("ANGLE", lambda rng: f"{round(rng.uniform(0,360),2)}"),
        ("FLOW_RATE", lambda rng: f"{round(rng.uniform(0,1000),2)}"),
        ("HUMIDITY", lambda rng: f"{round(rng.uniform(0,100),1)}"),
        ("LUMINOSITY", lambda rng: f"{round(rng.uniform(0,100000),1)}"),
        ("DECIBEL", lambda rng: f"{round(rng.uniform(0,130),1)}"),
        ("PH", lambda rng: f"{round(rng.uniform(0,14),2)}"),
        ("RESOLUTION", lambda rng: rng.choice(["1920x1080", "3840x2160", "1280x720", "2560x1440"])),
        ("RATIO", lambda rng: f"{round(rng.uniform(0,10),4)}"),
        ("POPULATION", lambda rng: f"{rng.randint(100,10000000)}"),
    ]:
        g[f"ICE.NONSENSITIVE.DESCRIPTIVE.MEASUREMENT.{code}"] = gen

    # ── Descriptive: Temporal ──
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.DATE"] = _gen_date
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.DATETIME"] = _gen_datetime
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.TIME"] = _gen_time
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.YEAR"] = _gen_year
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.MONTH"] = _gen_month
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.DAY_OF_WEEK"] = _gen_day_of_week
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.QUARTER"] = _gen_quarter
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.PERIOD"] = lambda rng: rng.choice(["Q1 2024", "FY 2025", "H2 2023", "2020-2025", "Jan-Mar"])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.START_DATE"] = _gen_date
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.END_DATE"] = _gen_date
    g["ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.CRON"] = _gen_cron

    # ── Descriptive: Numeric ──
    g["ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC.INTEGER"] = _gen_integer
    g["ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC.DECIMAL"] = _gen_decimal
    g["ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC.ORDINAL"] = lambda rng: str(rng.randint(1, 1000))
    g["ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC.PROBABILITY"] = lambda rng: f"{round(rng.uniform(0, 1), 4)}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC.BINARY"] = lambda rng: str(rng.choice([0, 1]))
    g["ICE.NONSENSITIVE.DESCRIPTIVE.NUMERIC.RANGE"] = lambda rng: f"{rng.randint(0,50)}-{rng.randint(51,100)}"

    # ── Descriptive: Embedding ──
    g["ICE.NONSENSITIVE.DESCRIPTIVE.EMBEDDING.WORD_EMBEDDING"] = lambda rng: str([round(rng.gauss(0,1),4) for _ in range(8)])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.EMBEDDING.FEATURE_VECTOR"] = lambda rng: str([round(rng.uniform(0,1),4) for _ in range(8)])
    g["ICE.NONSENSITIVE.DESCRIPTIVE.EMBEDDING.IMAGE_EMBEDDING"] = lambda rng: str([round(rng.gauss(0,0.5),4) for _ in range(8)])

    # ── Descriptive: Statistical ──
    g["ICE.NONSENSITIVE.DESCRIPTIVE.STATISTICAL.MEAN"] = lambda rng: f"{round(rng.uniform(-100,100),4)}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.STATISTICAL.MEDIAN"] = lambda rng: f"{round(rng.uniform(-100,100),4)}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.STATISTICAL.STD_DEV"] = lambda rng: f"{round(rng.uniform(0,50),4)}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.STATISTICAL.PERCENTILE"] = lambda rng: f"{round(rng.uniform(0,100),2)}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.STATISTICAL.CORRELATION"] = lambda rng: f"{round(rng.uniform(-1,1),4)}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.STATISTICAL.TREND"] = lambda rng: f"{round(rng.uniform(-50,50),2)}%"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.STATISTICAL.HISTOGRAM_BIN"] = lambda rng: f"{round(rng.uniform(0,100),2)}"
    g["ICE.NONSENSITIVE.DESCRIPTIVE.STATISTICAL.CONFIDENCE_INTERVAL"] = lambda rng: f"[{round(rng.uniform(0,40),2)}, {round(rng.uniform(60,100),2)}]"

    # ── Prescriptive ──
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.FORMAT_SPEC"] = lambda rng: rng.choice(["RFC 3339", "ISO 8601", "POSIX", "IEEE 754", "W3C Date"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.FORMULA"] = lambda rng: rng.choice(["E = mc²", "F = ma", "PV = nRT", "A = πr²", "BMI = kg/m²"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.RULE"] = lambda rng: rng.choice(["age >= 18", "amount <= 10000", "status IN ('active','pending')", "NOT NULL"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.CONFIG"] = lambda rng: f"{rng.choice(['max_retries','timeout_ms','batch_size','pool_size'])}={rng.randint(1,1000)}"
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.PERMISSION"] = lambda rng: rng.choice(["read", "write", "admin", "execute", "rw", "r-x", "rwx"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.SCHEDULE"] = lambda rng: rng.choice(["Daily at 2am", "Every 15 minutes", "Weekly on Monday", "Monthly on 1st"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.COMMAND"] = lambda rng: rng.choice(["GET /api/users", "POST /api/orders", "kubectl apply", "docker run", "npm install"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.TEMPLATE"] = lambda rng: rng.choice(["Hello {{name}}", "Order #{{id}}", "Dear {{customer}}", "{{date}} - {{event}}"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.THRESHOLD"] = lambda rng: f"{rng.choice(['max','min','warn','crit'])}={rng.randint(1,10000)}"
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.LICENSE"] = lambda rng: rng.choice(["MIT", "Apache-2.0", "GPL-3.0", "BSD-3-Clause", "Proprietary", "CC BY 4.0"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.SLA"] = lambda rng: f"{round(rng.uniform(99,100),3)}% uptime"
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.POLICY"] = lambda rng: rng.choice(["Data retained for 7 years", "PII encrypted at rest", "MFA required for admin"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.API_SPEC"] = lambda rng: rng.choice(["OpenAPI 3.0", "GraphQL", "gRPC/protobuf", "REST/JSON", "SOAP/XML"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.FILTER"] = lambda rng: f"{rng.choice(['status','type','date'])} {rng.choice(['=','!=','>','<','IN'])} '{rng.choice(['active','2024','prod'])}'"
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.MAPPING"] = lambda rng: f"{rng.choice(['source_col','field_a','input'])} -> {rng.choice(['target_col','field_b','output'])}"
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.SCHEMA_DEF"] = lambda rng: rng.choice(["CREATE TABLE t (id INT PRIMARY KEY)", '{"type": "object", "properties": {}}', "message Msg { string id = 1; }"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.CRON_SPEC"] = _gen_cron
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.REGEX"] = _gen_regex
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.RETENTION_POLICY"] = lambda rng: rng.choice(["7 days", "30 days", "1 year", "7 years", "indefinite"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.VALIDATION_RULE"] = lambda rng: rng.choice(["NOT NULL", "UNIQUE", "CHECK (val > 0)", "BETWEEN 1 AND 100", "MATCHES '^[A-Z]+'"])
    g["ICE.NONSENSITIVE.PRESCRIPTIVE.DEFAULT_VALUE"] = lambda rng: rng.choice(["0", "null", "''", "false", "CURRENT_TIMESTAMP", "1"])

    # ── Metadata ──
    g["ICE.METADATA.TIMESTAMP"] = _gen_timestamp
    g["ICE.METADATA.RECID"] = _gen_recid
    g["ICE.METADATA.STATUS"] = _gen_status
    g["ICE.METADATA.CREATED_BY"] = lambda rng: f"{rng.choice(_FIRST_NAMES).lower()}.{rng.choice(_LAST_NAMES).lower()}"
    g["ICE.METADATA.MODIFIED_BY"] = lambda rng: f"{rng.choice(_FIRST_NAMES).lower()}.{rng.choice(_LAST_NAMES).lower()}"
    g["ICE.METADATA.CREATED_AT"] = _gen_timestamp
    g["ICE.METADATA.MODIFIED_AT"] = _gen_timestamp
    g["ICE.METADATA.DELETED_AT"] = lambda rng: _gen_timestamp(rng) if rng.random() < 0.2 else ""
    g["ICE.METADATA.VERSION"] = _gen_version
    g["ICE.METADATA.SCHEMA"] = lambda rng: rng.choice(["public.users", "analytics.events", "sales.orders", "hr.employees"])
    g["ICE.METADATA.COLUMN_NAME"] = lambda rng: rng.choice(["id", "name", "email", "status", "created_at", "amount", "type"])
    g["ICE.METADATA.DATA_TYPE"] = lambda rng: rng.choice(["VARCHAR", "INTEGER", "BOOLEAN", "TIMESTAMP", "FLOAT", "TEXT", "JSON"])
    g["ICE.METADATA.ENCODING"] = lambda rng: rng.choice(["UTF-8", "ASCII", "Latin-1", "UTF-16", "ISO-8859-1"])
    g["ICE.METADATA.ROW_COUNT"] = _gen_count
    g["ICE.METADATA.CHECKSUM"] = _gen_hash
    g["ICE.METADATA.PARTITION"] = lambda rng: f"{rng.choice(['dt','region','tenant'])}={rng.choice(['2024-01','us-east','acme'])}"
    g["ICE.METADATA.TTL"] = lambda rng: f"{rng.randint(1,365)} days"
    g["ICE.METADATA.ETL_BATCH"] = lambda rng: f"batch-{_rng_date(rng).strftime('%Y%m%d')}-{rng.randint(1,99):02d}"
    g["ICE.METADATA.LINEAGE"] = lambda rng: f"{rng.choice(['raw','staging','curated'])}.{rng.choice(['users','orders','events'])} -> {rng.choice(['analytics','reports','exports'])}"
    g["ICE.METADATA.NULLABLE"] = lambda rng: rng.choice(["true", "false", "YES", "NO"])
    g["ICE.METADATA.SOURCE_SYSTEM"] = lambda rng: rng.choice(["Salesforce", "SAP", "Workday", "Snowflake", "Kafka", "PostgreSQL"])
    g["ICE.METADATA.LOAD_TIMESTAMP"] = _gen_timestamp
    g["ICE.METADATA.RECORD_TYPE"] = lambda rng: rng.choice(["insert", "update", "delete", "upsert", "snapshot"])
    g["ICE.METADATA.IS_DELETED"] = lambda rng: rng.choice(["false", "false", "false", "true"])
    g["ICE.METADATA.TENANT_ID"] = lambda rng: f"tenant-{rng.randint(1,50):03d}"


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
        "ICE.SENSITIVE.PID.IDENTITY.FULLNAME",
        "ICE.SENSITIVE.PID.IDENTITY.FIRST_NAME",
        "ICE.SENSITIVE.PID.IDENTITY.LAST_NAME",
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
        "ICE.SENSITIVE.PID.FINANCIAL.PAN",
        "ICE.SENSITIVE.PID.FINANCIAL.TXNAMT",
        "ICE.SENSITIVE.PID.FINANCIAL.CVV",
        "ICE.SENSITIVE.PID.FINANCIAL.EXPIRY",
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
        "ICE.SENSITIVE.PID.IDENTITY.MIDDLE_NAME",
        "ICE.SENSITIVE.PID.IDENTITY.MAIDEN_NAME",
        "ICE.SENSITIVE.PID.IDENTITY.NATIONALITY",
        "ICE.SENSITIVE.PID.IDENTITY.MARITAL_STATUS",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.SENSITIVE.PID.FINANCIAL.SALARY",
        "ICE.SENSITIVE.PID.FINANCIAL.TAX_ID",
        "ICE.SENSITIVE.PID.FINANCIAL.BAN",
        "ICE.SENSITIVE.PID.FINANCIAL.ROUTING_NUM",
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
        "ICE.SENSITIVE.PID.IDENTITY.PHOTO",
    ))
    cols += _take(_codes_matching(leaves,
        "ICE.SENSITIVE.PID.FINANCIAL.INSURANCE_ID",
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
        "ICE.SENSITIVE.PID.FINANCIAL.CREDIT_SCORE",
        "ICE.SENSITIVE.PID.FINANCIAL.INVESTMENT",
        "ICE.SENSITIVE.PID.FINANCIAL.CRYPTO_ADDR",
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
        "ICE.SENSITIVE.PID.IDENTITY.GOVID",
        "ICE.SENSITIVE.PID.IDENTITY.SIGNATURE",
        "ICE.SENSITIVE.PID.IDENTITY.VOICEPRINT",
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
    _register_generators()

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
    ground_truth = {}
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
            ground_truth[f"{table_name}.{col_name}"] = code
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

    # Write ground truth
    with open(GROUND_TRUTH_PATH, "w") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(tables)} tables with {total_columns} columns ({ROWS_PER_TABLE} rows each)")
    print(f"  Tables: {TABLES_DIR}/")
    print(f"  Ground truth: {GROUND_TRUTH_PATH}")
    print(f"  Leaf categories: {len(leaves)}")
    print(f"  Generator coverage: {len(leaves) - len(missing_gens)}/{len(leaves)}")
    print(f"  Opaque column names: {opaque_count}/{total_columns} ({opaque_count*100//total_columns}%)")
    print(f"\nTable breakdown:")
    for name, codes in tables:
        print(f"  {name}: {len(codes)} columns")


if __name__ == "__main__":
    main()
