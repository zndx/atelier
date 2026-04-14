#!/usr/bin/env python3
"""Expand the universal vocabulary from 16 PII leaves to 300+ BFO-grounded categories.

Every category uses our ICE.* coding scheme with definitions grounded in BFO/CCO
terms. External sources (GitTables 122 DBpedia types, meta-tagging 301 annotation
codes) informed which conceptual space to cover — the kinds of columns that exist
in real-world data — but we NEVER import their raw tags.

The mapping goes OUTWARD from our vocabulary via atelier-vocab.ttl, not inward.

Usage:
    uv run python scripts/expand_vocabulary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Output path ──────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "sample" / "ontology.json"

# ── Notation counter ─────────────────────────────────────────────

_notation_counters: dict[str, int] = {}


def _next_notation(parent_notation: str) -> str:
    """Generate the next child notation under a parent."""
    key = parent_notation
    _notation_counters[key] = _notation_counters.get(key, 0) + 1
    return f"{parent_notation}.{_notation_counters[key]}"


# ── Category builder ─────────────────────────────────────────────


def cat(
    code: str,
    label: str,
    parent_code: str | None,
    abbrev: str,
    description: str,
    common_names: str = "",
) -> dict:
    """Build a category dict matching universal_vocabulary.json schema."""
    return {
        "code": code,
        "label": label,
        "parent_code": parent_code,
        "taxonomy": "universal",
        "abbrev": abbrev,
        "description": description,
        **({"common_names": common_names} if common_names else {}),
    }


# ── Vocabulary definition ────────────────────────────────────────
#
# Structure follows the CCO ICE trichotomy:
#   ICE.NONSENSITIVE.DESIGNATIVE  ⊑  cco:DesignativeICE
#   ICE.NONSENSITIVE.DESCRIPTIVE  ⊑  cco:DescriptiveICE
#   ICE.NONSENSITIVE.PRESCRIPTIVE ⊑  cco:PrescriptiveICE
#
# Sensitive subtree expands the existing 16 PII leaves.
# Metadata subtree expands structural/provenance metadata.


def build_vocabulary() -> list[dict]:
    """Build the complete expanded vocabulary."""
    cats: list[dict] = []

    def add(code, label, parent, abbrev, desc, names=""):
        cats.append(cat(code, label, parent, abbrev, desc, names))

    # ════════════════════════════════════════════════════════════
    # ROOT
    # ════════════════════════════════════════════════════════════

    add("ICE", "Information Content Entity", None, "ICE",
        "BFO: Generically Dependent Continuant — copyable information content whose identity is the pattern, not the carrier.")

    # ════════════════════════════════════════════════════════════
    # ICE.NONSENSITIVE — Non-sensitive column types
    # ════════════════════════════════════════════════════════════

    add("ICE.NONSENSITIVE", "Non-Sensitive Data", "ICE", "C_NOS",
        "Data not requiring elevated protection. Internal operational or structural information.")

    # ── DESIGNATIVE (⊑ cco:DesignativeICE) ───────────────────

    add("ICE.NONSENSITIVE.DESIGNATIVE", "Designative Content", "ICE.NONSENSITIVE", "C_DESIG",
        "Information that designates — names, identifiers, codes, and references. Maps to cco:DesignativeICE.")

    # ── Names ──
    P = "ICE.NONSENSITIVE.DESIGNATIVE"
    add(f"{P}.NAME", "Name", P, "C_NAME",
        "Designates an entity by name — a natural language label for identification.")

    N = f"{P}.NAME"
    add(f"{N}.PERSON", "Person Name", N, "PERSNAME",
        "A person's name used in a non-PII context (e.g., author, contributor).",
        "author, contributor, reviewer, writer")
    add(f"{N}.ORG", "Organization Name", N, "ORGNAME",
        "The name of an organization, company, or institution.",
        "company, organization, institution, employer, publisher")
    add(f"{N}.PRODUCT", "Product Name", N, "PRODNAME",
        "The name of a product, software, or commercial artifact.",
        "product, brand, software, application, tool")
    add(f"{N}.SCIENTIFIC", "Scientific Name", N, "SCINAME",
        "A scientific or taxonomic name (genus, species, chemical compound).",
        "species, genus, compound, molecule, element")
    add(f"{N}.GEO", "Geographic Name", N, "GEONAME",
        "A geographic place name (country, city, region) in non-address context.",
        "country, city, state, region, continent, territory")
    add(f"{N}.EVENT", "Event Name", N, "EVTNAME",
        "The name of an event, conference, or occurrence.",
        "event, conference, meeting, festival, ceremony")
    add(f"{N}.PROJECT", "Project Name", N, "PROJNAME",
        "The name of a project, initiative, or campaign.",
        "project, initiative, campaign, program")
    add(f"{N}.MATERIAL", "Material Name", N, "MATNAME",
        "The name of a physical material or substance.",
        "material, substance, fabric, alloy, polymer")
    add(f"{N}.AWARD", "Award Name", N, "AWDNAME",
        "The name of an award, honor, or certification.",
        "award, prize, medal, certification, honor")
    add(f"{N}.BRAND", "Brand Name", N, "BRANDNAME",
        "A commercial brand or trademark name.",
        "brand, trademark, make, manufacturer")
    add(f"{N}.VENUE", "Venue Name", N, "VENUENAME",
        "The name of a venue, arena, or meeting place.",
        "venue, arena, stadium, theater, hall")
    add(f"{N}.ARTWORK", "Artwork / Creative Work", N, "ARTNAME",
        "The name of an artwork, painting, or creative work.",
        "artwork, painting, sculpture, exhibit")
    add(f"{N}.SONG", "Song / Track Title", N, "SONGNAME",
        "The title of a song, musical track, or composition.",
        "song, track, title, composition, album")
    add(f"{N}.MOVIE", "Movie / Film Title", N, "MOVIENAME",
        "The title of a movie, film, or TV show.",
        "movie, film, show, series, episode")
    add(f"{N}.BOOK", "Book Title", N, "BOOKNAME",
        "The title of a book, publication, or written work.",
        "book, publication, novel, textbook, paper")
    add(f"{N}.DISEASE", "Disease Name", N, "DISNAME",
        "The name of a disease, condition, or syndrome (non-PII context).",
        "disease, condition, syndrome, disorder, illness")
    add(f"{N}.DRUG", "Drug / Medication Name", N, "DRUGNAME",
        "The name of a drug, medication, or pharmaceutical (non-PII context).",
        "drug, medication, pharmaceutical, compound, active ingredient")
    add(f"{N}.FOOD", "Food / Ingredient Name", N, "FOODNAME",
        "The name of a food item, ingredient, or dish.",
        "food, ingredient, dish, recipe, cuisine")

    # ── Codes / Identifiers ──
    add(f"{P}.CODE", "Code / Identifier", P, "C_CODE",
        "Designates an entity by symbolic code or identifier.")

    C = f"{P}.CODE"
    add(f"{C}.ID", "Generic Identifier", C, "GENID",
        "A generic unique identifier (UUID, hash, or opaque key).",
        "id, identifier, key, uuid, guid, hash")
    add(f"{C}.ABBREV", "Abbreviation / Acronym", C, "ABBR",
        "An abbreviation or acronym used as a designator.",
        "abbreviation, acronym, code, symbol")
    add(f"{C}.POSTAL", "Postal Code", C, "POSTALCD",
        "A postal or ZIP code designating a geographic delivery area.",
        "zip, zipcode, postal code, postcode")
    add(f"{C}.ISO_COUNTRY", "ISO Country Code", C, "ISOCC",
        "An ISO 3166 country code (2-letter, 3-letter, or numeric).",
        "country code, iso alpha-2, iso alpha-3, iso 3166")
    add(f"{C}.ISO_CURRENCY", "ISO Currency Code", C, "ISOCUR",
        "An ISO 4217 currency code (USD, EUR, JPY).",
        "currency code, iso 4217, money code")
    add(f"{C}.ISO_LANGUAGE", "ISO Language Code", C, "ISOLANG",
        "An ISO 639 language code (en, fr, de).",
        "language code, iso 639, locale code")
    add(f"{C}.IATA", "IATA Code", C, "IATA",
        "An IATA airport or airline code.",
        "airport code, airline code, iata")
    add(f"{C}.NAICS", "NAICS Code", C, "NAICS",
        "A North American Industry Classification System code.",
        "naics, industry code, sic code")
    add(f"{C}.TICKER", "Stock Ticker", C, "TICKER",
        "A stock exchange ticker symbol.",
        "ticker, stock symbol, trading symbol")
    add(f"{C}.DOI", "Digital Object Identifier", C, "DOI",
        "A DOI uniquely identifying a digital publication.",
        "doi, digital object identifier")
    add(f"{C}.ISBN", "ISBN", C, "ISBN",
        "An International Standard Book Number.",
        "isbn, book number")
    add(f"{C}.ISSN", "ISSN", C, "ISSN",
        "An International Standard Serial Number for periodicals.",
        "issn, serial number, journal id")
    add(f"{C}.SKU", "SKU / Product Code", C, "SKU",
        "A Stock Keeping Unit or product catalog code.",
        "sku, product code, item number, part number, upc, ean, barcode")
    add(f"{C}.FIPS", "FIPS Code", C, "FIPS",
        "A Federal Information Processing Standard geographic code.",
        "fips, county code, state fips")
    add(f"{C}.LEI", "Legal Entity Identifier", C, "LEI",
        "A 20-character LEI per ISO 17442.",
        "lei, legal entity identifier")
    add(f"{C}.DUNS", "DUNS Number", C, "DUNS",
        "A Dun & Bradstreet business identifier.",
        "duns, duns number, d-u-n-s")
    add(f"{C}.MIME_TYPE", "MIME Type", C, "MIMETYPE",
        "A media type designation (application/json, text/csv).",
        "mime, content type, media type")
    add(f"{C}.PHONE_CODE", "Phone Country Code", C, "PHONECODE",
        "An international dialing code (+1, +44, +81).",
        "country code, dialing code, phone prefix")
    add(f"{C}.CAS", "CAS Registry Number", C, "CAS",
        "A Chemical Abstracts Service registry number.",
        "cas, cas number, chemical id")
    add(f"{C}.VIN", "Vehicle Identification Number", C, "VIN",
        "A vehicle identification number.",
        "vin, vehicle id, chassis number")
    add(f"{C}.CUSIP", "CUSIP Number", C, "CUSIP",
        "A Committee on Uniform Securities Identification Procedures number.",
        "cusip, security id, isin")
    add(f"{C}.CIK", "SEC CIK Number", C, "CIK",
        "A Central Index Key number from the SEC.",
        "cik, sec id, edgar id")
    add(f"{C}.GTIN", "Global Trade Item Number", C, "GTIN",
        "A GTIN product identifier (EAN, UPC, or ISBN).",
        "gtin, ean, upc, barcode, product id")
    add(f"{C}.HASH_ID", "Hash Identifier", C, "HASHID",
        "A cryptographic hash used as an identifier (SHA, MD5).",
        "hash, sha256, md5, sha1, digest, fingerprint")
    add(f"{C}.SEMANTIC_VERSION", "Semantic Version", C, "SEMVER",
        "A semantic version string (1.2.3, 2.0.0-rc.1).",
        "semver, version, release, v1.0, major.minor.patch")
    add(f"{C}.PLATE_NUMBER", "License Plate Number", C, "PLATENUM",
        "A vehicle registration plate number.",
        "license plate, plate number, registration")
    add(f"{C}.FLIGHT_NUMBER", "Flight Number", C, "FLIGHTNUM",
        "An airline flight number designation.",
        "flight number, flight, flight code")

    # ── Geographic ──
    add(f"{P}.GEO", "Geographic Reference", P, "C_GEO",
        "Designates a geographic site or spatial reference. Maps to BFO:Site.")

    G = f"{P}.GEO"
    add(f"{G}.COUNTRY", "Country", G, "COUNTRY",
        "A sovereign nation or territory.",
        "country, nation, state, sovereignty")
    add(f"{G}.REGION", "Region / State / Province", G, "REGION",
        "A sub-national administrative region.",
        "state, province, region, territory, prefecture, canton")
    add(f"{G}.CITY", "City / Town", G, "CITY",
        "A city, town, municipality, or urban area.",
        "city, town, municipality, village, metro")
    add(f"{G}.LOCATION", "Location Text", G, "LOCTXT",
        "General free-text location or place description.",
        "location, place, site, venue, area")
    add(f"{G}.COORDINATES", "Geographic Coordinates", G, "COORDS",
        "Latitude/longitude pair or other spatial coordinate.",
        "latitude, longitude, lat, lon, coordinates, geolocation")
    add(f"{G}.CONTINENT", "Continent", G, "CONTINENT",
        "One of the seven continents.",
        "continent")
    add(f"{G}.DISTRICT", "District / Borough", G, "DISTRICT",
        "A district, borough, ward, or neighborhood.",
        "district, borough, ward, neighborhood, county")
    add(f"{G}.TIMEZONE", "Time Zone", G, "TZ",
        "A time zone identifier (UTC, EST, America/New_York).",
        "timezone, tz, time zone, utc offset")
    add(f"{G}.ADDRESS_LINE", "Address Line", G, "ADDRLINE",
        "A non-PII address line (business address, facility address).",
        "address line, street, suite, floor, building")
    add(f"{G}.LANDMARK", "Landmark / POI", G, "LANDMARK",
        "A landmark, point of interest, or notable location.",
        "landmark, poi, monument, park, building, attraction")
    add(f"{G}.AIRPORT", "Airport", G, "AIRPORT",
        "An airport name or identifier.",
        "airport, airfield, aerodrome")
    add(f"{G}.PORT", "Port / Harbor", G, "PORT",
        "A seaport or harbor.",
        "port, harbor, dock, terminal, maritime")

    # ── References ──
    add(f"{P}.REF", "Reference", P, "C_REF",
        "References to external entities — citations, versions, sources.")

    R = f"{P}.REF"
    add(f"{R}.CITATION", "Citation", R, "CITATION",
        "A bibliographic citation or reference to a published work.",
        "citation, reference, bibliography, doi, source")
    add(f"{R}.VERSION", "Version Identifier", R, "VERID",
        "A version string or release identifier.",
        "version, release, revision, build, tag")
    add(f"{R}.SOURCE", "Source Attribution", R, "SRCATTR",
        "Attribution of data origin or provenance.",
        "source, origin, provenance, data source, feed")
    add(f"{R}.URL", "URL Reference", R, "URLREF",
        "A URL reference to an external resource (non-PII context).",
        "url, link, href, website, webpage, uri")
    add(f"{R}.FILEPATH", "File Path", R, "FPATH",
        "A filesystem path or object storage key.",
        "path, filepath, filename, directory, s3 key, blob path")

    # ── Title ──
    add(f"{P}.TITLE", "Title", P, "TITLE",
        "The title of a work, document, or creative output.",
        "title, headline, subject line, header, name")

    # ── Label ──
    add(f"{P}.LABEL", "Label / Tag", P, "LABEL",
        "A classification label, tag, or annotation applied to an entity.",
        "label, tag, annotation, category, class")

    # ── Boolean ──
    add(f"{P}.BOOLEAN", "Boolean Flag", P, "BOOLFLAG",
        "A designative boolean indicator (true/false, yes/no, 0/1).",
        "flag, boolean, indicator, is_active, enabled, has")

    # ── Email (non-PII) ──
    add(f"{P}.EMAIL_DOMAIN", "Email Domain", P, "EMAILDOM",
        "An email domain (non-PII, e.g., company domain patterns).",
        "email domain, domain, @company.com")

    # ── Hostname (non-PII) ──
    add(f"{P}.DOMAIN_NAME", "Domain Name", P, "DOMNAME",
        "An internet domain name (non-PII context).",
        "domain, hostname, fqdn, tld, subdomain")

    # ── Phone format ──
    add(f"{P}.PHONE_FORMAT", "Phone Format Pattern", P, "PHONEFMT",
        "A phone number format pattern (non-PII, e.g., +1-XXX-XXX-XXXX).",
        "phone format, country pattern, dialing format")

    # ── DESCRIPTIVE (⊑ cco:DescriptiveICE) ───────────────────

    add("ICE.NONSENSITIVE.DESCRIPTIVE", "Descriptive Content", "ICE.NONSENSITIVE", "C_DESC",
        "Information that describes — text, measurements, categories, temporal values. Maps to cco:DescriptiveICE.")

    D = "ICE.NONSENSITIVE.DESCRIPTIVE"

    # ── Text ──
    add(f"{D}.TEXT", "Text Content", D, "C_TXT",
        "Free-text descriptions, comments, and narrative content.")

    T = f"{D}.TEXT"
    add(f"{T}.DESCRIPTION", "Description", T, "DESC",
        "A general free-text description of an entity.",
        "description, desc, detail, summary, about")
    add(f"{T}.COMMENT", "Comment / Note", T, "COMMENT",
        "A comment, note, or remark attached to a record.",
        "comment, note, remark, observation, memo")
    add(f"{T}.ABSTRACT", "Abstract / Summary", T, "ABSTRACT",
        "A summary or abstract of a longer work.",
        "abstract, summary, synopsis, overview, brief")
    add(f"{T}.DEFINITION", "Definition", T, "DEFN",
        "A formal definition of a term or concept.",
        "definition, meaning, explanation, gloss")
    add(f"{T}.BODY", "Body Text", T, "BODYTEXT",
        "The main body text of a document or message.",
        "body, content, text, message, article")
    add(f"{T}.KEYWORDS", "Keywords", T, "KEYWORDS",
        "A list of keywords or key phrases.",
        "keywords, tags, key phrases, topics, terms")
    add(f"{T}.INSTRUCTION", "Instruction Text", T, "INSTRTEXT",
        "Instructional or how-to text content.",
        "instructions, directions, steps, procedure, how-to")
    add(f"{T}.REVIEW", "Review Text", T, "REVIEW",
        "A review, rating commentary, or evaluation text.",
        "review, feedback, evaluation, testimonial, critique")
    add(f"{T}.LOG_MESSAGE", "Log Message", T, "LOGMSG",
        "A structured or semi-structured log entry.",
        "log, log message, trace, event log, audit log")
    add(f"{T}.ERROR_MESSAGE", "Error Message", T, "ERRMSG",
        "An error message, exception, or diagnostic text.",
        "error, exception, stacktrace, fault, warning")
    add(f"{T}.QUERY", "Query / Search Text", T, "QUERY",
        "A search query, SQL statement, or filter expression.",
        "query, search, sql, filter, expression")
    add(f"{T}.TRANSLATION", "Translation", T, "XLATE",
        "A translated version of text content.",
        "translation, translated, localized, i18n")
    add(f"{T}.HEADLINE", "Headline / Title Text", T, "HEADLINE",
        "A headline, subject line, or title text.",
        "headline, subject, title, header, tagline")
    add(f"{T}.CAPTION", "Caption", T, "CAPTION",
        "A caption or alt-text for an image or media.",
        "caption, alt text, subtitle, legend, tooltip")
    add(f"{T}.ADDRESS_TEXT", "Address Text", T, "ADDRTXT",
        "A formatted address as free text (non-PII, business context).",
        "address, formatted address, full address, mailing address")
    add(f"{T}.BIOGRAPHY", "Biography", T, "BIO",
        "A biographical description (non-PII, public context).",
        "bio, biography, about me, profile, blurb")
    add(f"{T}.CHANGELOG", "Changelog Entry", T, "CHANGELOG",
        "A changelog or release note entry.",
        "changelog, release note, change, update, what's new")
    add(f"{T}.RECIPE", "Recipe / Instructions", T, "RECIPE",
        "A recipe, cooking instructions, or procedural text.",
        "recipe, ingredients, instructions, steps, method")

    # ── Categorical ──
    add(f"{D}.CATEGORICAL", "Categorical Value", D, "C_CAT",
        "Categorical descriptors — types, classes, ranks, enumerations.")

    CAT = f"{D}.CATEGORICAL"
    add(f"{CAT}.TYPE", "Type / Class", CAT, "TYPECLS",
        "A type or class label from a controlled enumeration.",
        "type, class, kind, category, classification")
    add(f"{CAT}.CATEGORY", "Category", CAT, "CATEG",
        "A category or grouping assignment.",
        "category, group, segment, bucket, tier")
    add(f"{CAT}.RANK", "Rank / Rating", CAT, "RANK",
        "An ordinal rank, rating, or priority level.",
        "rank, rating, priority, level, grade, tier, severity")
    add(f"{CAT}.LANGUAGE", "Language", CAT, "LANG",
        "A human language name or designation.",
        "language, lang, locale, tongue")
    add(f"{CAT}.GENDER", "Gender", CAT, "GENDER",
        "A gender designation (non-PII context, e.g., product category).",
        "gender, sex, m/f")
    add(f"{CAT}.COLOR", "Color", CAT, "COLOR",
        "A color designation or code.",
        "color, colour, hex color, rgb")
    add(f"{CAT}.SIZE", "Size / Dimension Category", CAT, "SIZECAT",
        "A categorical size designation (S, M, L, XL).",
        "size, dimension, fit, small, medium, large")
    add(f"{CAT}.STATUS", "Status / State", CAT, "CATSTAT",
        "A categorical status or lifecycle state value.",
        "status, state, phase, stage, condition, lifecycle")
    add(f"{CAT}.INDUSTRY", "Industry / Sector", CAT, "INDUSTRY",
        "An industry, sector, or vertical classification.",
        "industry, sector, vertical, domain, market")
    add(f"{CAT}.DEPARTMENT", "Department / Division", CAT, "DEPT",
        "An organizational department or division.",
        "department, division, unit, team, group, org unit")
    add(f"{CAT}.ROLE", "Role / Job Title", CAT, "JOBROLE",
        "A job title, role, or position designation.",
        "role, title, position, job, occupation, profession")
    add(f"{CAT}.FORMAT", "Data Format", CAT, "DATAFMT",
        "A data format designation (CSV, JSON, XML, Parquet).",
        "format, file format, encoding, data type, schema")
    add(f"{CAT}.PROTOCOL", "Protocol", CAT, "PROTOCOL",
        "A communication or data transfer protocol.",
        "protocol, http, https, ftp, ssh, tcp, udp")
    add(f"{CAT}.BOOLEAN_ENUM", "Boolean Enumeration", CAT, "BOOLENUM",
        "An enumerated boolean-like value (active/inactive, enabled/disabled).",
        "active, inactive, enabled, disabled, yes, no, on, off")
    add(f"{CAT}.SEVERITY", "Severity Level", CAT, "SEVERITY",
        "A severity or criticality classification.",
        "severity, criticality, urgency, priority, impact")
    add(f"{CAT}.SENTIMENT", "Sentiment", CAT, "SENTIMENT",
        "A sentiment classification (positive, negative, neutral).",
        "sentiment, polarity, mood, tone, positive, negative")
    add(f"{CAT}.LIFECYCLE", "Lifecycle Stage", CAT, "LIFECYCLE",
        "A lifecycle or maturity stage designation.",
        "lifecycle, stage, maturity, phase, draft, published, archived")
    add(f"{CAT}.CHANNEL", "Channel", CAT, "CHANNEL",
        "A communication or distribution channel.",
        "channel, medium, source, platform, touchpoint")
    add(f"{CAT}.PLATFORM", "Platform / OS", CAT, "PLATFORM",
        "An operating system, platform, or runtime environment.",
        "platform, os, operating system, runtime, environment")
    add(f"{CAT}.BROWSER", "Browser / User Agent", CAT, "BROWSER",
        "A web browser or user agent designation.",
        "browser, user agent, chrome, firefox, safari")
    add(f"{CAT}.DEVICE_TYPE", "Device Type", CAT, "DEVTYPE",
        "A device type classification (mobile, desktop, tablet).",
        "device type, mobile, desktop, tablet, iot, wearable")
    add(f"{CAT}.CONTENT_TYPE", "Content Type", CAT, "CONTTYPE",
        "A content type or media classification.",
        "content type, media, article, video, image, podcast, post")
    add(f"{CAT}.PAYMENT_METHOD", "Payment Method", CAT, "PAYMETHOD",
        "A payment method designation.",
        "payment method, credit card, debit, wire, ach, paypal, cash")
    add(f"{CAT}.SHIPPING_METHOD", "Shipping Method", CAT, "SHIPMETHOD",
        "A shipping or delivery method.",
        "shipping, delivery, express, standard, freight, overnight")
    add(f"{CAT}.EDUCATION_LEVEL", "Education Level", CAT, "EDULEVEL",
        "An educational attainment level.",
        "education, degree, diploma, bachelor, master, phd, certification")
    add(f"{CAT}.RELATIONSHIP", "Relationship Type", CAT, "RELTYPE",
        "A relationship type between entities.",
        "relationship, relation, parent, child, sibling, spouse, manager")
    add(f"{CAT}.PERMISSION_LEVEL", "Permission Level", CAT, "PERMLEVEL",
        "An access permission or authorization level.",
        "permission, access level, admin, read, write, execute, owner")
    add(f"{CAT}.RISK_LEVEL", "Risk Level", CAT, "RISKLEVEL",
        "A risk assessment classification.",
        "risk, risk level, high risk, low risk, exposure, vulnerability")
    add(f"{CAT}.APPROVAL_STATUS", "Approval Status", CAT, "APPRVSTAT",
        "An approval or review status.",
        "approved, pending, rejected, review, submitted, in review")
    add(f"{CAT}.CURRENCY_NAME", "Currency Name", CAT, "CURNAME",
        "A currency name (non-ISO-code context).",
        "currency, dollar, euro, yen, pound, rupee")
    add(f"{CAT}.UNIT", "Unit of Measure", CAT, "UNITOFMEAS",
        "A unit of measurement designation.",
        "unit, uom, kg, lb, meter, inch, gallon, liter")

    # ── Measurement ──
    add(f"{D}.MEASUREMENT", "Measurement", D, "C_MEAS",
        "Quantitative descriptions — measurements of BFO:Quality instances.")

    M = f"{D}.MEASUREMENT"
    # Physical measurements
    add(f"{M}.LENGTH", "Length", M, "MEASLEN",
        "A measurement of spatial extent along one dimension.",
        "length, distance, height, width, depth, size")
    add(f"{M}.AREA", "Area", M, "MEASAREA",
        "A measurement of two-dimensional spatial extent.",
        "area, surface area, sq ft, sq m, acreage")
    add(f"{M}.VOLUME", "Volume", M, "MEASVOL",
        "A measurement of three-dimensional spatial extent.",
        "volume, capacity, liters, gallons, cubic")
    add(f"{M}.WEIGHT", "Weight / Mass", M, "MEASWT",
        "A measurement of mass or weight.",
        "weight, mass, kg, lbs, grams, tons, ounces")
    add(f"{M}.TEMPERATURE", "Temperature", M, "MEASTEMP",
        "A measurement of temperature.",
        "temperature, temp, celsius, fahrenheit, kelvin")
    add(f"{M}.SPEED", "Speed / Velocity", M, "MEASSPD",
        "A measurement of speed or velocity.",
        "speed, velocity, mph, km/h, knots")
    add(f"{M}.PRESSURE", "Pressure", M, "MEASPRS",
        "A measurement of pressure.",
        "pressure, psi, bar, pascal, atm")
    add(f"{M}.FREQUENCY", "Frequency", M, "MEASFREQ",
        "A measurement of frequency or rate of occurrence.",
        "frequency, rate, hz, rpm, bpm, per second")
    add(f"{M}.ENERGY", "Energy", M, "MEASENRG",
        "A measurement of energy.",
        "energy, joules, calories, kwh, watt-hours")
    add(f"{M}.POWER", "Power", M, "MEASPWR",
        "A measurement of power output.",
        "power, watts, horsepower, kw")
    add(f"{M}.DENSITY", "Density", M, "MEASDENS",
        "A measurement of density (mass per volume).",
        "density, g/cm3, kg/m3, specific gravity")
    add(f"{M}.CONCENTRATION", "Concentration", M, "MEASCONC",
        "A measurement of concentration (ppm, mol/L).",
        "concentration, ppm, ppb, molarity, mg/L, percent")

    # Statistical/derived measurements
    add(f"{M}.PERCENTAGE", "Percentage", M, "MEASPCT",
        "A ratio expressed as a fraction of 100.",
        "percentage, percent, ratio, rate, proportion, share")
    add(f"{M}.SCORE", "Score / Index", M, "MEASSCORE",
        "A numeric score, index, or rating value.",
        "score, index, rating, grade, rank, points")
    add(f"{M}.COUNT", "Count", M, "MEASCNT",
        "An integer count of discrete items.",
        "count, number, quantity, total, tally, occurrences")
    add(f"{M}.DURATION", "Duration", M, "MEASDUR",
        "A measurement of elapsed time.",
        "duration, elapsed, interval, period, time span, seconds, minutes")
    add(f"{M}.AGE", "Age", M, "MEASAGE",
        "A measurement of age (years, months, days).",
        "age, years old, months old")

    # Financial measurements (non-PII — aggregates, prices, budgets)
    add(f"{M}.PRICE", "Price / Cost", M, "MEASPRICE",
        "A monetary amount representing price or cost.",
        "price, cost, fee, charge, rate, amount, value")
    add(f"{M}.REVENUE", "Revenue / Sales", M, "MEASREV",
        "An aggregate revenue or sales figure.",
        "revenue, sales, income, earnings, turnover, gross")
    add(f"{M}.BUDGET", "Budget / Allocation", M, "MEASBUDGET",
        "A budgeted or allocated monetary amount.",
        "budget, allocation, funding, spend, expenditure")
    add(f"{M}.MARKET_VALUE", "Market Value", M, "MEASMKTVAL",
        "A market valuation or capitalization figure.",
        "market value, market cap, valuation, appraisal, assessment")
    add(f"{M}.EXCHANGE_RATE", "Exchange Rate", M, "MEASXRATE",
        "A currency exchange rate.",
        "exchange rate, fx rate, conversion rate, forex")

    # Geospatial measurements
    add(f"{M}.LATITUDE", "Latitude", M, "MEASLAT",
        "A geographic latitude value in decimal degrees.",
        "latitude, lat")
    add(f"{M}.LONGITUDE", "Longitude", M, "MEASLON",
        "A geographic longitude value in decimal degrees.",
        "longitude, lon, lng")
    add(f"{M}.ELEVATION", "Elevation / Altitude", M, "MEASELEV",
        "A measurement of elevation above sea level.",
        "elevation, altitude, height, asl")
    add(f"{M}.DISTANCE", "Distance", M, "MEASDIST",
        "A measurement of distance between two points.",
        "distance, range, miles, kilometers, proximity")

    # Computing/digital measurements
    add(f"{M}.BANDWIDTH", "Bandwidth / Data Rate", M, "MEASBW",
        "A measurement of data transfer rate.",
        "bandwidth, throughput, mbps, gbps, data rate")
    add(f"{M}.STORAGE", "Storage Size", M, "MEASSTOR",
        "A measurement of data storage capacity.",
        "storage, file size, bytes, kb, mb, gb, tb")
    add(f"{M}.LATENCY", "Latency / Response Time", M, "MEASLAT_T",
        "A measurement of delay or response time.",
        "latency, response time, ping, delay, ms, milliseconds")
    add(f"{M}.UPTIME", "Uptime / Availability", M, "MEASUP",
        "A measurement of system availability.",
        "uptime, availability, sla, nine nines")
    add(f"{M}.CPU_USAGE", "CPU Usage", M, "MEASCPU",
        "A measurement of processor utilization.",
        "cpu, cpu usage, processor, utilization, load")
    add(f"{M}.MEMORY_USAGE", "Memory Usage", M, "MEASMEM",
        "A measurement of memory consumption.",
        "memory, ram, heap, memory usage, resident")
    add(f"{M}.VOLTAGE", "Voltage", M, "MEASVOLT",
        "A measurement of electrical potential difference.",
        "voltage, volts, v, potential")
    add(f"{M}.CURRENT", "Electric Current", M, "MEASCURR",
        "A measurement of electric current.",
        "current, amps, amperes, milliamps")
    add(f"{M}.ANGLE", "Angle", M, "MEASANGLE",
        "A measurement of angular displacement.",
        "angle, degrees, radians, heading, bearing, azimuth")
    add(f"{M}.FLOW_RATE", "Flow Rate", M, "MEASFLOW",
        "A measurement of fluid or data flow rate.",
        "flow rate, gallons per minute, liters per second, throughput")
    add(f"{M}.HUMIDITY", "Humidity", M, "MEASHUM",
        "A measurement of relative or absolute humidity.",
        "humidity, rh, relative humidity, dew point")
    add(f"{M}.LUMINOSITY", "Luminosity / Brightness", M, "MEASLUM",
        "A measurement of luminous intensity or brightness.",
        "luminosity, brightness, lux, lumens, candela")
    add(f"{M}.DECIBEL", "Sound Level", M, "MEASDB",
        "A measurement of sound pressure level.",
        "decibel, db, sound level, noise, volume")
    add(f"{M}.PH", "pH Level", M, "MEASPH",
        "A measurement of acidity/alkalinity (pH scale).",
        "ph, acidity, alkalinity, ph level")
    add(f"{M}.RESOLUTION", "Resolution", M, "MEASRES",
        "A measurement of image or display resolution.",
        "resolution, dpi, ppi, pixels, megapixels, 1080p, 4k")
    add(f"{M}.RATIO", "Ratio", M, "MEASRATIO",
        "A dimensionless ratio or proportion.",
        "ratio, proportion, factor, multiplier, coefficient")
    add(f"{M}.POPULATION", "Population", M, "MEASPOP",
        "A count of population or user base.",
        "population, inhabitants, users, subscribers, members")

    # ── Temporal ──
    add(f"{D}.TEMPORAL", "Temporal Value", D, "C_TEMP",
        "Temporal descriptions — dates, times, periods, durations.")

    TP = f"{D}.TEMPORAL"
    add(f"{TP}.DATE", "Date", TP, "DATEVAL",
        "A calendar date (YYYY-MM-DD or similar format).",
        "date, calendar date, day")
    add(f"{TP}.DATETIME", "Date-Time", TP, "DTVAL",
        "A combined date and time value.",
        "datetime, timestamp, date time, iso 8601")
    add(f"{TP}.TIME", "Time of Day", TP, "TIMEVAL",
        "A time-of-day value without date component.",
        "time, clock time, hour, hh:mm, time of day")
    add(f"{TP}.YEAR", "Year", TP, "YEARVAL",
        "A calendar year value.",
        "year, fiscal year, calendar year, vintage")
    add(f"{TP}.MONTH", "Month", TP, "MONTHVAL",
        "A month name or number.",
        "month, month name, month number")
    add(f"{TP}.DAY_OF_WEEK", "Day of Week", TP, "DOWVAL",
        "A day-of-week designation.",
        "day of week, weekday, day name, dow")
    add(f"{TP}.QUARTER", "Quarter", TP, "QTRVAL",
        "A fiscal or calendar quarter designation (Q1-Q4).",
        "quarter, q1, q2, q3, q4, fiscal quarter")
    add(f"{TP}.PERIOD", "Period / Interval", TP, "PERIODVAL",
        "A named time period or interval.",
        "period, interval, span, range, epoch, era")
    add(f"{TP}.START_DATE", "Start Date", TP, "STARTDT",
        "The beginning date of a temporal interval.",
        "start date, begin date, effective date, from date, since")
    add(f"{TP}.END_DATE", "End Date", TP, "ENDDT",
        "The ending date of a temporal interval.",
        "end date, expiry date, until, through, to date")
    add(f"{TP}.CRON", "Cron / Schedule", TP, "CRONVAL",
        "A cron expression or schedule specification.",
        "cron, schedule, recurrence, frequency, interval")

    # ── Numeric (non-measurement) ──
    add(f"{D}.NUMERIC", "Numeric Value", D, "C_NUM",
        "Numeric values that are descriptive but not unit-bearing measurements.")

    NU = f"{D}.NUMERIC"
    add(f"{NU}.INTEGER", "Integer", NU, "INTVAL",
        "A whole number value without units.",
        "integer, int, whole number, count")
    add(f"{NU}.DECIMAL", "Decimal / Float", NU, "DECVAL",
        "A decimal or floating-point number without units.",
        "decimal, float, double, real, number")
    add(f"{NU}.ORDINAL", "Ordinal Position", NU, "ORDVAL",
        "An ordinal position in a sequence.",
        "ordinal, position, rank, order, sequence number, index")
    add(f"{NU}.PROBABILITY", "Probability / Confidence", NU, "PROBVAL",
        "A probability, confidence score, or likelihood value [0,1].",
        "probability, confidence, likelihood, p-value, score")
    add(f"{NU}.BINARY", "Binary Value", NU, "BINVAL",
        "A binary 0/1 numeric value.",
        "binary, bit, flag, 0/1")
    add(f"{NU}.RANGE", "Numeric Range", NU, "RANGEVAL",
        "A numeric range expressed as min-max.",
        "range, min-max, band, bracket, interval")

    # ── Embedding / Vector ──
    add(f"{D}.EMBEDDING", "Embedding / Vector", D, "C_EMBED",
        "A numeric vector or embedding representation.")

    EM = f"{D}.EMBEDDING"
    add(f"{EM}.WORD_EMBEDDING", "Word Embedding", EM, "WORDEMB",
        "A word or sentence embedding vector.",
        "embedding, vector, word2vec, bert, transformer, representation")
    add(f"{EM}.FEATURE_VECTOR", "Feature Vector", EM, "FEATVEC",
        "A feature vector for machine learning input.",
        "features, feature vector, input vector, encoding")
    add(f"{EM}.IMAGE_EMBEDDING", "Image Embedding", EM, "IMGEMB",
        "An image embedding or visual feature vector.",
        "image embedding, visual features, cnn features, clip")

    # ── Statistical ──
    add(f"{D}.STATISTICAL", "Statistical Value", D, "C_STAT",
        "Statistical measures, aggregates, and derived quantities.")

    STAT = f"{D}.STATISTICAL"
    add(f"{STAT}.MEAN", "Mean / Average", STAT, "MEAN",
        "An arithmetic mean or average value.",
        "mean, average, avg, arithmetic mean")
    add(f"{STAT}.MEDIAN", "Median", STAT, "MEDIAN",
        "A median value (50th percentile).",
        "median, middle, 50th percentile")
    add(f"{STAT}.STD_DEV", "Standard Deviation", STAT, "STDDEV",
        "A standard deviation measurement.",
        "standard deviation, std, sigma, variance, spread")
    add(f"{STAT}.PERCENTILE", "Percentile", STAT, "PCTILE",
        "A percentile or quantile value.",
        "percentile, quantile, quartile, p50, p95, p99")
    add(f"{STAT}.CORRELATION", "Correlation", STAT, "CORR",
        "A correlation coefficient or association measure.",
        "correlation, r-squared, pearson, spearman, association")
    add(f"{STAT}.TREND", "Trend / Growth Rate", STAT, "TREND",
        "A trend, growth rate, or change-over-time metric.",
        "trend, growth, change, delta, yoy, mom, rate of change")
    add(f"{STAT}.HISTOGRAM_BIN", "Histogram Bin", STAT, "HISTBIN",
        "A histogram bin edge or bucket boundary.",
        "bin, bucket, histogram, distribution, bin edge")
    add(f"{STAT}.CONFIDENCE_INTERVAL", "Confidence Interval", STAT, "CI",
        "A statistical confidence interval bound.",
        "confidence interval, ci, lower bound, upper bound, margin of error")

    # ── PRESCRIPTIVE (⊑ cco:PrescriptiveICE) ─────────────────

    add("ICE.NONSENSITIVE.PRESCRIPTIVE", "Prescriptive Content", "ICE.NONSENSITIVE", "C_PRESC",
        "Information that prescribes — specifications, formulas, rules, and plans. Maps to cco:PrescriptiveICE.")

    PR = "ICE.NONSENSITIVE.PRESCRIPTIVE"
    add(f"{PR}.FORMAT_SPEC", "Format Specification", PR, "FMTSPEC",
        "A specification defining data format, schema, or structure.",
        "format spec, schema, template, pattern, layout, specification")
    add(f"{PR}.FORMULA", "Formula / Expression", PR, "FORMULA",
        "A mathematical, chemical, or logical formula.",
        "formula, equation, expression, calculation, regex")
    add(f"{PR}.RULE", "Rule / Constraint", PR, "RULE",
        "A business rule, validation constraint, or policy rule.",
        "rule, constraint, policy, validation, check, assertion")
    add(f"{PR}.CONFIG", "Configuration", PR, "CONFIG",
        "A configuration setting, parameter, or option.",
        "config, setting, parameter, option, preference, property")
    add(f"{PR}.PERMISSION", "Permission / Access Control", PR, "PERM",
        "An access control or permission specification.",
        "permission, access, role, acl, privilege, grant")
    add(f"{PR}.SCHEDULE", "Schedule / Plan", PR, "SCHED",
        "A schedule, plan, or workflow specification.",
        "schedule, plan, workflow, pipeline, process, sequence")
    add(f"{PR}.COMMAND", "Command / Action", PR, "CMD",
        "A command, action, or operation specification.",
        "command, action, operation, method, verb, endpoint")
    add(f"{PR}.TEMPLATE", "Template / Pattern", PR, "TMPL",
        "A template, pattern, or boilerplate text.",
        "template, pattern, boilerplate, placeholder, mask")
    add(f"{PR}.THRESHOLD", "Threshold / Limit", PR, "THRESH",
        "A threshold, limit, or boundary value.",
        "threshold, limit, max, min, ceiling, floor, bound, cap")
    add(f"{PR}.LICENSE", "License / Terms", PR, "LICENSE",
        "A license type, terms of service, or usage agreement.",
        "license, terms, eula, agreement, open source, copyright")
    add(f"{PR}.SLA", "Service Level Agreement", PR, "SLA",
        "A service level agreement or performance guarantee.",
        "sla, service level, uptime guarantee, response time target")
    add(f"{PR}.POLICY", "Policy Definition", PR, "POLICY",
        "A policy definition or governance rule.",
        "policy, governance, compliance, regulation, requirement")
    add(f"{PR}.API_SPEC", "API Specification", PR, "APISPEC",
        "An API specification or interface definition.",
        "api spec, openapi, swagger, graphql, endpoint, interface")
    add(f"{PR}.FILTER", "Filter / Predicate", PR, "FILTER",
        "A filter predicate, query condition, or selection criteria.",
        "filter, predicate, condition, where clause, criteria")
    add(f"{PR}.MAPPING", "Mapping / Transform", PR, "MAPPING",
        "A data mapping, transformation, or conversion rule.",
        "mapping, transform, conversion, crosswalk, lookup")
    add(f"{PR}.SCHEMA_DEF", "Schema Definition", PR, "SCHEMADEF",
        "A schema definition (DDL, JSON Schema, Avro, Protobuf).",
        "schema, ddl, json schema, avro, protobuf, model definition")
    add(f"{PR}.CRON_SPEC", "Cron Specification", PR, "CRONSPEC",
        "A cron expression or scheduling specification.",
        "cron, schedule, 0 * * * *, recurrence, job schedule")
    add(f"{PR}.REGEX", "Regular Expression", PR, "REGEX",
        "A regular expression pattern.",
        "regex, regexp, pattern, glob, wildcard, match")
    add(f"{PR}.RETENTION_POLICY", "Retention Policy", PR, "RETENTION",
        "A data retention or archival policy specification.",
        "retention, archival, purge, expiry policy, data lifecycle")
    add(f"{PR}.VALIDATION_RULE", "Validation Rule", PR, "VALRULE",
        "A data validation rule or constraint expression.",
        "validation, check constraint, data quality, assertion, invariant")
    add(f"{PR}.DEFAULT_VALUE", "Default Value", PR, "DEFVAL",
        "A prescribed default value for a field.",
        "default, fallback, initial value, seed value")

    # ════════════════════════════════════════════════════════════
    # ICE.SENSITIVE — Sensitive data (expanded from original 16)
    # ════════════════════════════════════════════════════════════

    add("ICE.SENSITIVE", "Sensitive Data", "ICE", "C_SENS",
        "Data requiring protection due to privacy, regulatory, or business sensitivity.")

    add("ICE.SENSITIVE.PID", "Personally Identifiable Data", "ICE.SENSITIVE", "C_PID",
        "Data relating to an identified or identifiable natural person.")

    # ── Contact ──
    add("ICE.SENSITIVE.PID.CONTACT", "Contact Information", "ICE.SENSITIVE.PID", "C_CI",
        "Information used to contact a person.")

    SC = "ICE.SENSITIVE.PID.CONTACT"
    add(f"{SC}.EMAIL", "Email Address", SC, "EMAIL",
        "An electronic mail address for a person.",
        "email, e-mail")
    add(f"{SC}.PHONE", "Phone Number", SC, "PHONE",
        "A telephone number for contacting a person.",
        "phone, tel, telephone")
    add(f"{SC}.ADDRESS", "Mailing Address", SC, "ADDR",
        "A physical postal address.",
        "address, street address, postal address")
    add(f"{SC}.FAX", "Fax Number", SC, "FAX",
        "A facsimile telephone number.",
        "fax, facsimile")
    add(f"{SC}.SOCIAL_MEDIA", "Social Media Handle", SC, "SOCIAL",
        "A social media account handle or profile URL.",
        "twitter, linkedin, facebook, instagram, social media, handle, username")
    add(f"{SC}.MESSENGER", "Messaging ID", SC, "MSNGR",
        "An instant messaging identifier.",
        "skype, whatsapp, telegram, messenger, chat id, im")

    # ── Identity ──
    add("ICE.SENSITIVE.PID.IDENTITY", "Identity Information", "ICE.SENSITIVE.PID", "C_ID",
        "Data that identifies a specific person.")

    SI = "ICE.SENSITIVE.PID.IDENTITY"
    add(f"{SI}.FULLNAME", "Full Name", SI, "FULLNAME",
        "A person's complete legal name.",
        "name, person name, legal name")
    add(f"{SI}.FIRST_NAME", "First Name", SI, "FNAME",
        "A person's given name.",
        "first name, given name, forename")
    add(f"{SI}.LAST_NAME", "Last Name", SI, "LNAME",
        "A person's family name.",
        "last name, surname, family name")
    add(f"{SI}.MIDDLE_NAME", "Middle Name", SI, "MNAME",
        "A person's middle name or initial.",
        "middle name, middle initial")
    add(f"{SI}.MAIDEN_NAME", "Maiden Name", SI, "MAIDEN",
        "A person's birth surname before marriage.",
        "maiden name, birth name, nee")
    add(f"{SI}.DOB", "Date of Birth", SI, "DOB",
        "The date on which a person was born.",
        "DOB, birthday, birth date")
    add(f"{SI}.GOVID", "Government Identifier", SI, "GOVID",
        "A government-issued identifier (SSN, passport, driver's license).",
        "SSN, social security, tax id, passport number")
    add(f"{SI}.GENDER_PII", "Gender (PII)", SI, "GENDERPII",
        "A person's gender identity (PII context).",
        "gender, sex, m/f, male, female")
    add(f"{SI}.NATIONALITY", "Nationality", SI, "NATL",
        "A person's citizenship or nationality.",
        "nationality, citizenship, country of origin")
    add(f"{SI}.ETHNICITY", "Ethnicity / Race", SI, "ETHNIC",
        "A person's ethnicity or racial classification.",
        "ethnicity, race, ethnic group, ancestry")
    add(f"{SI}.BIRTH_PLACE", "Birth Place", SI, "BPLACE",
        "A person's place of birth.",
        "birthplace, birth city, place of birth, born in")
    add(f"{SI}.MARITAL_STATUS", "Marital Status", SI, "MARITAL",
        "A person's marital status.",
        "marital status, married, single, divorced, widowed")
    add(f"{SI}.PHOTO", "Photograph", SI, "PHOTO",
        "A photograph or biometric image of a person.",
        "photo, image, headshot, portrait, biometric")
    add(f"{SI}.SIGNATURE", "Signature", SI, "SIG",
        "A person's handwritten or digital signature.",
        "signature, autograph, signed, digital signature")
    add(f"{SI}.VOICEPRINT", "Voiceprint", SI, "VOICEPR",
        "A biometric voiceprint or voice recording.",
        "voiceprint, voice, audio, speech sample")

    # ── Financial ──
    add("ICE.SENSITIVE.PID.FINANCIAL", "Financial Data", "ICE.SENSITIVE.PID", "C_FD",
        "Data relating to financial accounts or transactions.")

    SF = "ICE.SENSITIVE.PID.FINANCIAL"
    add(f"{SF}.PAN", "Payment Card Number", SF, "PAN",
        "A credit or debit card primary account number.",
        "PAN, credit card, card number, CC number")
    add(f"{SF}.BAN", "Bank Account Number", SF, "BAN",
        "A unique identifier for a bank account.",
        "account number, IBAN, routing number")
    add(f"{SF}.TXNAMT", "Transaction Amount", SF, "TXNAMT",
        "The monetary value of a financial transaction.",
        "amount, payment, price")
    add(f"{SF}.CVV", "Card Verification Value", SF, "CVV",
        "A card verification value or security code.",
        "cvv, cvc, security code, card verification")
    add(f"{SF}.EXPIRY", "Card Expiry Date", SF, "CARDEXP",
        "A payment card expiration date.",
        "expiry, expiration, exp date, valid thru")
    add(f"{SF}.TAX_ID", "Tax Identifier", SF, "TAXID",
        "A tax identification number (EIN, TIN, VAT number).",
        "tax id, ein, tin, vat number, tax number")
    add(f"{SF}.SALARY", "Salary / Compensation", SF, "SALARY",
        "A person's salary, wage, or compensation amount.",
        "salary, wage, compensation, pay, income")
    add(f"{SF}.CREDIT_SCORE", "Credit Score", SF, "CREDSCORE",
        "A person's credit score or credit rating.",
        "credit score, fico, credit rating, credit history")
    add(f"{SF}.INVESTMENT", "Investment Account", SF, "INVEST",
        "An investment or brokerage account identifier.",
        "investment, brokerage, portfolio, 401k, ira, pension")
    add(f"{SF}.INSURANCE_ID", "Insurance ID", SF, "INSID",
        "An insurance policy number or member ID.",
        "insurance id, policy number, member id, claim number")
    add(f"{SF}.ROUTING_NUM", "Routing Number", SF, "ROUTNUM",
        "A bank routing or transit number.",
        "routing number, aba, sort code, swift, bic")
    add(f"{SF}.CRYPTO_ADDR", "Cryptocurrency Address", SF, "CRYPTOADDR",
        "A cryptocurrency wallet address.",
        "wallet address, bitcoin, ethereum, crypto address")

    # ── Health ──
    add("ICE.SENSITIVE.PID.HEALTH", "Health Data", "ICE.SENSITIVE.PID", "C_HEALTH",
        "Protected health information (PHI) relating to a person's medical history.")

    SH = "ICE.SENSITIVE.PID.HEALTH"
    add(f"{SH}.DIAGNOSIS", "Diagnosis", SH, "DIAG",
        "A medical diagnosis or condition classification.",
        "diagnosis, condition, disease, disorder, icd code")
    add(f"{SH}.PRESCRIPTION", "Prescription", SH, "PRESCRIP",
        "A medication prescription or drug order.",
        "prescription, medication, drug, rx, dosage")
    add(f"{SH}.BLOOD_TYPE", "Blood Type", SH, "BLDTYPE",
        "A person's blood type classification.",
        "blood type, abo, rh factor")
    add(f"{SH}.ALLERGY", "Allergy", SH, "ALLERGY",
        "A documented allergy or adverse reaction.",
        "allergy, allergies, adverse reaction, sensitivity")
    add(f"{SH}.MEDICAL_RECORD", "Medical Record Number", SH, "MRN",
        "A medical record number or health ID.",
        "mrn, medical record, health id, patient id")
    add(f"{SH}.INSURANCE_CLAIM", "Insurance Claim", SH, "INSCLAIM",
        "A health insurance claim identifier.",
        "claim, insurance claim, claim number, eob")
    add(f"{SH}.LAB_RESULT", "Lab Result", SH, "LABRESULT",
        "A clinical laboratory test result.",
        "lab result, test result, panel, blood test, urinalysis")
    add(f"{SH}.VITAL_SIGN", "Vital Sign", SH, "VITAL",
        "A clinical vital sign measurement (BP, HR, temp).",
        "vital sign, blood pressure, heart rate, pulse, bmi, o2 sat")
    add(f"{SH}.PROCEDURE", "Medical Procedure", SH, "MEDPROC",
        "A medical or surgical procedure record.",
        "procedure, surgery, operation, cpt code, treatment")
    add(f"{SH}.DISABILITY", "Disability Status", SH, "DISABIL",
        "A disability classification or accommodation status.",
        "disability, handicap, ada, accommodation, impairment")

    # ── Technical ──
    add("ICE.SENSITIVE.TECHNICAL", "Technical Data", "ICE.SENSITIVE", "C_TD",
        "Data relating to technical systems and infrastructure.")

    ST = "ICE.SENSITIVE.TECHNICAL"
    add(f"{ST}.IPADDR", "IP Address", ST, "IPADDR",
        "An Internet Protocol address identifying a device on a network.",
        "IP, IPv4, IPv6, network address")
    add(f"{ST}.DEVID", "Device Identifier", ST, "DEVID",
        "A unique identifier for a computing device.",
        "UUID, MAC address, device ID, serial number")
    add(f"{ST}.URL_PII", "URL (PII context)", ST, "URLPII",
        "A URL that may expose personal activity or identity.",
        "web address, link, URI, endpoint")
    add(f"{ST}.HOSTNAME", "Hostname", ST, "HOSTNAME",
        "A DNS hostname or server name.",
        "hostname, server, fqdn, domain name")
    add(f"{ST}.API_KEY", "API Key / Token", ST, "APIKEY",
        "An API key, access token, or secret.",
        "api key, token, secret, bearer, access key")
    add(f"{ST}.PASSWORD_HASH", "Password Hash", ST, "PASSHASH",
        "A hashed password or credential digest.",
        "password, hash, bcrypt, sha256, digest, credential")
    add(f"{ST}.SESSION_TOKEN", "Session Token", ST, "SESSTOKEN",
        "A session identifier or authentication token.",
        "session, session id, jwt, cookie, auth token")
    add(f"{ST}.COOKIE", "Browser Cookie", ST, "COOKIE",
        "A browser cookie value containing tracking or session data.",
        "cookie, tracking, browser cookie")
    add(f"{ST}.SSH_KEY", "SSH Key", ST, "SSHKEY",
        "An SSH public or private key.",
        "ssh key, public key, private key, authorized key")
    add(f"{ST}.CERTIFICATE", "Digital Certificate", ST, "CERT",
        "An X.509 certificate or TLS/SSL certificate.",
        "certificate, cert, x509, ssl, tls, pem")
    add(f"{ST}.DATABASE_CONN", "Database Connection String", ST, "DBCONN",
        "A database connection string containing credentials.",
        "connection string, dsn, jdbc, odbc, database url")

    # ── Business ──
    add("ICE.SENSITIVE.BUSINESS", "Business-Sensitive Data", "ICE.SENSITIVE", "C_BIZ",
        "Data sensitive due to commercial, competitive, or contractual concerns.")

    SB = "ICE.SENSITIVE.BUSINESS"
    add(f"{SB}.TRADE_SECRET", "Trade Secret", SB, "TRADESEC",
        "Proprietary business information constituting a trade secret.",
        "trade secret, proprietary, confidential formula, secret recipe")
    add(f"{SB}.CONTRACT_VALUE", "Contract Value", SB, "CONTRVAL",
        "A contract monetary value or deal size.",
        "contract value, deal size, agreement amount, bid")
    add(f"{SB}.CUSTOMER_LIST", "Customer List", SB, "CUSTLIST",
        "A customer, client, or prospect list.",
        "customer list, client list, prospect, lead list")
    add(f"{SB}.PRICING_STRATEGY", "Pricing Strategy", SB, "PRICESTRAT",
        "Internal pricing strategy or margin data.",
        "pricing, margin, markup, discount strategy, cost structure")
    add(f"{SB}.MARKET_INTEL", "Market Intelligence", SB, "MKTINTEL",
        "Competitive or market intelligence data.",
        "market intel, competitive analysis, market research, competitor")
    add(f"{SB}.INTERNAL_MEMO", "Internal Communication", SB, "INTMEMO",
        "Internal memos, emails, or communications marked confidential.",
        "internal memo, confidential, restricted, internal only")
    add(f"{SB}.LEGAL_HOLD", "Legal Hold Document", SB, "LEGHOLD",
        "Documents under legal hold or litigation preservation.",
        "legal hold, litigation, preservation, privileged, attorney-client")
    add(f"{SB}.AUDIT_FINDING", "Audit Finding", SB, "AUDITFIND",
        "Internal or external audit findings and observations.",
        "audit finding, observation, non-conformance, gap, remediation")
    add(f"{SB}.FINANCIAL_FORECAST", "Financial Forecast", SB, "FINFORECAST",
        "Internal financial forecast or projection data.",
        "forecast, projection, estimate, guidance, outlook")
    add(f"{SB}.STRATEGIC_PLAN", "Strategic Plan", SB, "STRATPLAN",
        "Strategic planning or roadmap data.",
        "strategy, roadmap, initiative, objective, okr, kpi target")
    add(f"{SB}.EMPLOYEE_RECORD", "Employee Record", SB, "EMPREC",
        "An employee record containing sensitive HR data.",
        "employee record, personnel, hr record, staff record")
    add(f"{SB}.PERFORMANCE_REVIEW", "Performance Review", SB, "PERFREV",
        "An employee performance review or evaluation.",
        "performance review, evaluation, appraisal, feedback, rating")

    # ════════════════════════════════════════════════════════════
    # ICE.METADATA — Metadata (expanded)
    # ════════════════════════════════════════════════════════════

    add("ICE.METADATA", "Metadata", "ICE", "C_META",
        "Data about data — timestamps, identifiers, and structural information.")

    MD = "ICE.METADATA"
    add(f"{MD}.TIMESTAMP", "Timestamp", MD, "TSTAMP",
        "A date and/or time value recording when an event occurred.",
        "date, datetime, created_at, updated_at")
    add(f"{MD}.RECID", "Record Identifier", MD, "RECID",
        "A unique key identifying a database record.",
        "ID, primary key, row ID, record ID")
    add(f"{MD}.STATUS", "Status Code", MD, "STATUS",
        "A categorical value indicating the state of a record.",
        "status, state, flag, code")

    # Provenance metadata
    add(f"{MD}.CREATED_BY", "Created By", MD, "CREATEDBY",
        "The user or system that created a record.",
        "created by, author, creator, inserted by")
    add(f"{MD}.MODIFIED_BY", "Modified By", MD, "MODIFIEDBY",
        "The user or system that last modified a record.",
        "modified by, updated by, editor, changed by")
    add(f"{MD}.CREATED_AT", "Created At", MD, "CREATEDAT",
        "The timestamp when a record was created.",
        "created at, creation date, insert time, registered")
    add(f"{MD}.MODIFIED_AT", "Modified At", MD, "MODIFIEDAT",
        "The timestamp when a record was last modified.",
        "modified at, updated at, last modified, changed at")
    add(f"{MD}.DELETED_AT", "Deleted At", MD, "DELETEDAT",
        "The timestamp when a record was soft-deleted.",
        "deleted at, removed at, archived at, expired at")
    add(f"{MD}.VERSION", "Version Number", MD, "VERSIONNO",
        "A version or revision number for a record.",
        "version, revision, generation, iteration")

    # Structural metadata
    add(f"{MD}.SCHEMA", "Schema / Table Name", MD, "SCHEMANAME",
        "A database schema, table, or collection name.",
        "schema, table, collection, entity, model")
    add(f"{MD}.COLUMN_NAME", "Column / Field Name", MD, "COLNAME",
        "A column, field, or attribute name.",
        "column, field, attribute, property, key")
    add(f"{MD}.DATA_TYPE", "Data Type", MD, "DATATYPE",
        "A data type designation for a column or field.",
        "data type, type, dtype, varchar, integer, boolean")
    add(f"{MD}.ENCODING", "Encoding", MD, "ENCODING",
        "A character or data encoding specification.",
        "encoding, charset, utf-8, ascii, latin-1")
    add(f"{MD}.ROW_COUNT", "Row Count", MD, "ROWCNT",
        "The number of rows in a table or result set.",
        "row count, record count, cardinality, total rows")
    add(f"{MD}.CHECKSUM", "Checksum / Hash", MD, "CHECKSUM",
        "A data integrity checksum or hash value.",
        "checksum, hash, md5, sha256, digest, crc")
    add(f"{MD}.PARTITION", "Partition Key", MD, "PARTKEY",
        "A partitioning key for distributed data storage.",
        "partition, shard, bucket, partition key")
    add(f"{MD}.TTL", "Time to Live", MD, "TTL",
        "A time-to-live or expiration setting for cached data.",
        "ttl, expiry, cache duration, retention")
    add(f"{MD}.ETL_BATCH", "ETL Batch ID", MD, "ETLBATCH",
        "An ETL batch or pipeline execution identifier.",
        "batch id, etl id, pipeline run, job id, load id")
    add(f"{MD}.LINEAGE", "Data Lineage", MD, "LINEAGE",
        "Data lineage or provenance tracking information.",
        "lineage, provenance, origin, data flow, upstream, downstream")
    add(f"{MD}.NULLABLE", "Nullable Flag", MD, "NULLABLE",
        "Indicates whether a field allows null values.",
        "nullable, not null, required, optional, mandatory")
    add(f"{MD}.SOURCE_SYSTEM", "Source System", MD, "SRCSYS",
        "The system or application that originated the data.",
        "source system, origin, feed, integration, upstream")
    add(f"{MD}.LOAD_TIMESTAMP", "Load Timestamp", MD, "LOADTS",
        "The timestamp when data was loaded into the current system.",
        "load time, ingestion time, arrival time, landed at")
    add(f"{MD}.RECORD_TYPE", "Record Type Discriminator", MD, "RECTYPE",
        "A discriminator column indicating record type in a polymorphic table.",
        "record type, dtype, discriminator, entity type")
    add(f"{MD}.IS_DELETED", "Soft Delete Flag", MD, "ISDEL",
        "A flag indicating soft deletion of a record.",
        "is_deleted, deleted, soft delete, tombstone, removed")
    add(f"{MD}.TENANT_ID", "Tenant Identifier", MD, "TENANTID",
        "A multi-tenant partition identifier.",
        "tenant, tenant id, org id, workspace, account")

    return cats


# ── Validation ───────────────────────────────────────────────────


def validate(cats: list[dict]) -> list[str]:
    """Validate hierarchy consistency. Returns list of error messages."""
    errors = []
    codes = {c["code"] for c in cats}

    for c in cats:
        # Every parent_code must exist (except root)
        if c["parent_code"] is not None and c["parent_code"] not in codes:
            errors.append(f"Missing parent {c['parent_code']} for {c['code']}")

        # Code must start with parent_code (hierarchy consistency)
        if c["parent_code"] is not None:
            if not c["code"].startswith(c["parent_code"] + "."):
                errors.append(
                    f"Code {c['code']} doesn't start with parent {c['parent_code']}"
                )

        # Required fields
        for field in ("code", "label", "abbrev", "description"):
            if not c.get(field):
                errors.append(f"Missing {field} for {c.get('code', '???')}")

    # Check for duplicate codes
    seen = set()
    for c in cats:
        if c["code"] in seen:
            errors.append(f"Duplicate code: {c['code']}")
        seen.add(c["code"])

    # Check for duplicate abbrevs
    abbrevs: dict[str, str] = {}
    for c in cats:
        ab = c["abbrev"]
        if ab in abbrevs:
            errors.append(
                f"Duplicate abbrev '{ab}': {c['code']} vs {abbrevs[ab]}"
            )
        abbrevs[ab] = c["code"]

    return errors


# ── Main ─────────────────────────────────────────────────────────


def main():
    cats = build_vocabulary()

    # Validate
    errors = validate(cats)
    if errors:
        print("Validation errors:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    # Count
    codes = {c["code"] for c in cats}
    leaves = [c for c in cats if not any(
        c2["parent_code"] == c["code"] for c2 in cats
    )]
    internal = [c for c in cats if any(
        c2["parent_code"] == c["code"] for c2 in cats
    )]

    print(f"Total: {len(cats)} categories")
    print(f"  Leaves: {len(leaves)}")
    print(f"  Internal: {len(internal)}")

    # Breakdown by top-level subtree
    for prefix in ("ICE.NONSENSITIVE.DESIGNATIVE", "ICE.NONSENSITIVE.DESCRIPTIVE",
                    "ICE.NONSENSITIVE.PRESCRIPTIVE", "ICE.SENSITIVE", "ICE.METADATA"):
        n = sum(1 for c in leaves if c["code"].startswith(prefix))
        print(f"  {prefix}: {n} leaves")

    # Write
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(cats, f, indent=2, ensure_ascii=False)
    print(f"\nWritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
