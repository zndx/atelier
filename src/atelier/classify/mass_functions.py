"""Evidence-to-mass converters for Dempster-Shafer belief functions.

Each converter transforms a specific evidence source into a BeliefAssignment
(mass function) compatible with Dempster's rule of combination.

Six evidence sources: cosine similarity (M0), pattern detection (M0),
name matching (M0), LLM classification (M1), CatBoost (M2), SVM (M2).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

from atelier.classify.belief import BeliefAssignment, FocalElement, FrameOfDiscernment

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscountConfig:
    """DST discount factors loaded from HOCON config.

    Each value controls how much evidence mass the corresponding source
    allocates to Theta (total ignorance).  Higher = more conservative.

    Discount calibration follows Shafer's reliability operator (1976,
    §11.3): ``m'(A) = α·m(A); m'(Θ) = α·m(Θ) + (1-α)`` for source
    reliability α = 1 - discount.  When evidence sources are *non-
    distinct* — the LLM-derivative case described by Denoeux 2008 —
    Dempster's rule double-counts overlapping mass.  The principled
    response is a substantial discount on derivative sources so the
    fused belief reflects the underlying independent signal rather
    than amplified self-agreement.

    In this pipeline ``catboost_*`` and ``svm`` ride on labels that
    are deterministic transforms of the LLM sweep (see
    ``ml_train.fit_catboost_to_llm_labels`` and the frontier-SVM
    label filter at ``ml_train`` lines 118-127), so their defaults
    are calibrated above the genuinely independent ``cosine``
    discount.  See ``docs/src/architecture/dst-evidence-independence.md``
    for the full rationale.
    """

    cosine: float = 0.30
    svm: float = 0.55
    pattern_theta: float = 0.25
    name_match_exact: float = 0.70
    name_match_code: float = 0.50
    name_match_alias: float = 0.50
    name_match_overlap: float = 0.30
    catboost_base: float = 0.55
    catboost_variance_scale: float = 1.6
    catboost_max: float = 0.75
    catboost_fallback: float = 0.55
    confusable_ratio_threshold: float = 3.0

    @classmethod
    def from_cfg(cls, cfg) -> DiscountConfig:
        """Build from an AtelierConfig (reads classify_discount_* fields)."""
        return cls(
            cosine=cfg.classify_discount_cosine,
            svm=cfg.classify_discount_svm,
            pattern_theta=cfg.classify_discount_pattern_theta,
            name_match_exact=cfg.classify_discount_name_match_exact,
            name_match_code=cfg.classify_discount_name_match_code,
            name_match_alias=cfg.classify_discount_name_match_alias,
            name_match_overlap=cfg.classify_discount_name_match_overlap,
            catboost_base=cfg.classify_discount_catboost_base,
            catboost_variance_scale=cfg.classify_discount_catboost_variance_scale,
            catboost_max=cfg.classify_discount_catboost_max,
            catboost_fallback=cfg.classify_discount_catboost_fallback,
            confusable_ratio_threshold=cfg.classify_discount_confusable_ratio_threshold,
        )


def _camel_to_words(name: str) -> str:
    """Split CamelCase into lowercase words."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).lower()


def _redistribute_confusable_mass(
    masses: dict[FocalElement, float],
    frame: FrameOfDiscernment,
    ratio_threshold: float = 3.0,
) -> dict[FocalElement, float]:
    """Redistribute singleton mass to confusable pair focal elements.

    When the top-2 singleton masses both belong to a known confusable pair
    and their ratio is below *ratio_threshold*, half of the 2nd-place mass
    is moved to the pair focal element.
    """
    if not frame.confusable_map:
        return masses

    singleton_masses: list[tuple[str, FocalElement, float]] = []
    for fe, m in masses.items():
        if len(fe.codes) == 1:
            code = next(iter(fe.codes))
            singleton_masses.append((code, fe, m))

    singleton_masses.sort(key=lambda x: -x[2])
    if len(singleton_masses) < 2:
        return masses

    code1, _, m1 = singleton_masses[0]
    code2, fe2, m2 = singleton_masses[1]

    if m2 <= 1e-15:
        return masses

    ratio = m1 / m2
    if ratio >= ratio_threshold:
        return masses

    pairs1 = frame.confusable_map.get(code1, [])
    target_pair: FocalElement | None = None
    for pair_fe in pairs1:
        if code2 in pair_fe.codes:
            target_pair = pair_fe
            break

    if target_pair is None:
        return masses

    transfer = m2 / 2.0
    result = dict(masses)
    result[fe2] = m2 - transfer
    result[target_pair] = result.get(target_pair, 0.0) + transfer
    return {fe: m for fe, m in result.items() if m > 1e-15}


# ── Cosine similarity ────────────────────────────────────────────────


def cosine_to_mass(
    similarities: dict[str, float],
    frame: FrameOfDiscernment,
    discount: float = 0.3,
) -> BeliefAssignment:
    """Convert cosine similarities to a mass function.

    Applies softmax to similarities, then discounts by *discount* so
    a fraction of mass goes to Theta (total ignorance).
    """
    if not similarities:
        return frame.vacuous()

    max_sim = max(similarities.values())
    exp_sims = {
        code: math.exp(sim - max_sim)
        for code, sim in similarities.items()
        if code in frame.singletons
    }
    total_exp = sum(exp_sims.values())
    if total_exp <= 0:
        return frame.vacuous()

    masses: dict[FocalElement, float] = {}
    evidence_mass = 1.0 - discount
    for code, exp_val in exp_sims.items():
        prob = exp_val / total_exp
        mass = prob * evidence_mass
        if mass > 1e-15:
            masses[frame.singleton(code)] = mass

    masses[frame.theta] = discount
    masses = _redistribute_confusable_mass(masses, frame)
    return BeliefAssignment(masses=masses)


# ── Pattern detection ────────────────────────────────────────────────


# Pattern → annotation code mapping.  Keys are pattern names from
# features.detect_patterns(); values are ICE.* leaf codes.
#
# **Active set** below is curated to patterns whose regex + optional
# post-validator uniquely identifies a specific instance type (not a
# generic format class).  A pattern belongs here only if:
#   1. It maps to a specific leaf code, not a catch-all parent, AND
#   2. Either the regex is anchored to a distinctive literal prefix
#      (``ssh-``, ``MRN-``, ``-----BEGIN``, ``$2[aby]$``, ``IBAN``),
#      or a post-validator imposes real precision (Luhn for card
#      numbers, dotted-quad range check for IPv4).
#
# Patterns that fire on *generic-shaped* strings (any 10-digit number,
# any YYYY-MM-DD, any 17-char alnum) were quarantined on 2026-04-17
# after the Synthetic overwatch audit showed they fire wrong 8× more
# than right — they vote for parent-of-leaf codes and compete with
# LLM's specific-leaf votes.  Quarantined entries are preserved as
# ``_QUARANTINED_PATTERN_MAP`` below for provenance and are dormant;
# re-activate only after weighting them below the LLM evidence source.
DEFAULT_PATTERN_MAP: dict[str, str] = {
    # Anchored-prefix or validator-guarded patterns only.
    "email_pattern": "ICE.SENSITIVE.PID.CONTACT.EMAIL",
    "ssn_pattern": "ICE.SENSITIVE.PID.IDENTITY.GOVID.SSN",  # strict ###-##-####
    "ipv4_pattern": "ICE.SENSITIVE.TECHNICAL.IPADDR",       # dotted-quad + range validator
    "uuid_pattern": "ICE.SENSITIVE.TECHNICAL.DEVID",        # standard UUID format
    "url_pattern": "ICE.SENSITIVE.TECHNICAL.URL",           # http(s):// prefix
    "credit_card_pattern": "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.PAN",  # + Luhn
    "mac_address_pattern": "ICE.SENSITIVE.TECHNICAL.DEVID", # HH:HH:HH:HH:HH:HH
    "iban_pattern": "ICE.SENSITIVE.PID.FINANCIAL.ACCOUNT.BAN",  # 2-letter country + checksum
    "monetary_pattern": "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.TXNAMT",  # leading currency sym
    "semver_pattern": "ICE.METADATA.VERSION",
    "bcrypt_pattern": "ICE.SENSITIVE.TECHNICAL.PASSWORD_HASH",  # $2[aby]$ prefix
    "mrn_pattern": "ICE.SENSITIVE.PID.HEALTH.MRN",          # MRN- prefix
    "ssh_key_pattern": "ICE.SENSITIVE.TECHNICAL.ACCESS_KEY", # ssh-rsa/ed25519/ecdsa prefix
    "certificate_pattern": "ICE.SENSITIVE.TECHNICAL.CERTIFICATE",  # -----BEGIN prefix
}


# Quarantined as of 2026-04-17 — high false-positive rate on
# Synthetic (audit: 1,077 wrong fires vs 136 right across these 8
# patterns) and incompatible with the transparency/explainability
# thesis: patterns that fire on generic shapes vote for parent
# codes and compete with LLM-specific leaf votes rather than
# corroborating them.  To reactivate, consult the audit in
# ``build/results/a22f1f10/overwatch.md``.
_QUARANTINED_PATTERN_MAP: dict[str, str] = {
    "phone_pattern": "ICE.SENSITIVE.PID.CONTACT.PHONE",
    "date_iso_pattern": "ICE.NONSENSITIVE.DESCRIPTIVE.TEMPORAL.DATE",
    "datetime_iso_pattern": "ICE.METADATA.TIMESTAMP",
    "postal_code_pattern": "ICE.NONSENSITIVE.DESIGNATIVE.CODE.POSTAL",
    "hex_hash_pattern": "ICE.NONSENSITIVE.DESIGNATIVE.CODE.HASH_ID",
    "vin_pattern": "ICE.NONSENSITIVE.DESIGNATIVE.CODE.ID",
    "iata_pattern": "ICE.NONSENSITIVE.DESIGNATIVE.CODE.ABBREV",
    "license_plate_pattern": "ICE.NONSENSITIVE.DESIGNATIVE.CODE.ID",
    "eth_address_pattern": "ICE.SENSITIVE.TECHNICAL.DEVID",
}


def _normalize_token(value: str) -> str:
    """Lowercase + strip non-alphanumerics for alias comparison."""
    return re.sub(r"[^a-z0-9]+", "", value.lower())


# ── Ontology priors ─────────────────────────────────────────────────


_ONTOLOGY_BY_CODE_CACHE: dict[str, dict] | None = None
_ONTOLOGY_PATH_CACHE: dict[str, list[str]] = {}


def _load_universal_ontology() -> dict[str, dict]:
    """Load + index ``universal_vocabulary.json`` by code, lazily.

    Returns a flat ``{code: entry}`` map.  Cached at module level so
    every per-column ontology lookup is an O(1) dict hit after first
    call.
    """
    global _ONTOLOGY_BY_CODE_CACHE
    if _ONTOLOGY_BY_CODE_CACHE is not None:
        return _ONTOLOGY_BY_CODE_CACHE

    import json
    from pathlib import Path

    fixtures_dir = Path(__file__).parent / "fixtures"
    path = fixtures_dir / "universal_vocabulary.json"
    try:
        with open(path) as f:
            records = json.load(f)
    except Exception as exc:
        logger.warning("Failed to load universal vocabulary for ontology priors: %s", exc)
        _ONTOLOGY_BY_CODE_CACHE = {}
        return _ONTOLOGY_BY_CODE_CACHE

    indexed: dict[str, dict] = {r["code"]: r for r in records if r.get("code")}
    _ONTOLOGY_BY_CODE_CACHE = indexed
    return indexed


def _ontology_path_labels(code: str) -> list[str]:
    """Walk parent_code chain to the root, returning labels root→leaf.

    Uses the universal vocabulary's parent_code field to reconstruct
    the ontological path (e.g. ``["Sensitive Data", "Personally
    Identifiable Data", "Financial Data", "Payment Data", "Transaction
    Amount"]``).  Cached per code.
    """
    if code in _ONTOLOGY_PATH_CACHE:
        return _ONTOLOGY_PATH_CACHE[code]

    by_code = _load_universal_ontology()
    labels: list[str] = []
    current = code
    seen: set[str] = set()
    while current and current not in seen:
        seen.add(current)
        entry = by_code.get(current)
        if not entry:
            break
        labels.insert(0, entry.get("label") or current)
        current = entry.get("parent_code") or ""
    _ONTOLOGY_PATH_CACHE[code] = labels
    return labels


def lookup_pattern_ontology(
    pattern: str,
    pattern_category_map: dict[str, str] | None = None,
) -> dict | None:
    """Return canonical ICE.* metadata for a fired pattern, or None.

    Used as a publicly-grounded semantic prior fed into:

    * the column embedding text (so cosine sees ontology terms an
      embedding model recognizes from training, not just the regex
      name — see ``ColumnFeatures.to_embedding_text``);
    * the first-pass LLM user prompt (so the LLM has a translation
      anchor when the user vocabulary lacks a direct equivalent of
      the detected pattern — He et al. 2023, ontology alignment via
      LLMs);
    * SAGE/SHAP attribution (as the ``ontology_priors`` ablatable
      feature, distinct from raw embedding text).

    The returned dict carries label, description, common_names, the
    full ontological path (root→leaf), and the canonical code itself.
    Every element traces to a public source — see
    ``src/atelier/classify/fixtures/PROVENANCE.md``.

    Returns ``None`` when the pattern has no entry in the pattern map
    or when the target code is missing from the universal vocabulary
    (defensive — both should be invariants).
    """
    if pattern_category_map is None:
        pattern_category_map = DEFAULT_PATTERN_MAP

    target_code = pattern_category_map.get(pattern)
    if not target_code:
        return None

    by_code = _load_universal_ontology()
    entry = by_code.get(target_code)
    if not entry:
        return None

    return {
        "pattern": pattern,
        "code": target_code,
        "label": entry.get("label") or "",
        "description": entry.get("description") or "",
        "common_names": entry.get("common_names") or "",
        "path": _ontology_path_labels(target_code),
    }


def resolve_pattern_map(
    default_map: dict[str, str],
    category_set,
) -> dict[str, str]:
    """Resolve canonical ICE.* pattern targets against the active vocabulary.

    The default map at ``DEFAULT_PATTERN_MAP`` references canonical
    BFO/Common-Core mnemonic codes (``ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.TXNAMT``
    etc.).  When the user-selected vocabulary uses a different code
    convention — e.g. dotted-numeric ``1.1.1.2.5.1`` — every pattern
    target falls outside ``frame.singletons`` and ``pattern_to_mass``
    silently returns vacuous mass, disabling the entire pattern source
    with no operator-visible warning.

    This resolver maps each ICE.* target through three fallback layers
    against ``category_set``:

    1. **Direct hit**: target exists in ``category_set.all_by_code``.
    2. **By abbrev**: leaf mnemonic (the suffix after the last ``.``)
       matches a category's ``abbrev`` field.
    3. **By common-name alias**: leaf mnemonic appears in a category's
       ``common_names`` field after token normalization.

    On a miss we log a single ``WARNING`` and drop the entry so the
    pattern source contributes nothing for that pattern (preserving the
    pre-resolver fail-safe semantics) instead of pointing at a
    nonexistent code.

    Returns a new ``{pattern_name: resolved_code}`` map containing only
    patterns whose target was successfully resolved.

    Args:
        default_map: pattern-name → canonical-target-code mapping.
        category_set: ``HierarchicalCategorySet`` describing the active
            vocabulary; uses ``all_by_code``, ``by_abbrev``, and
            ``all_categories``.

    Notes:
        Resolver runs once per pipeline run (called from
        ``_classify_column``).  The deeper BFO/Common-Core ontology
        mapping that this shim approximates remains future work.
    """
    if not default_map:
        return {}

    all_by_code = getattr(category_set, "all_by_code", {}) or {}
    by_abbrev = getattr(category_set, "by_abbrev", {}) or {}
    all_categories = getattr(category_set, "all_categories", []) or []

    resolved: dict[str, str] = {}
    misses: list[tuple[str, str]] = []

    for pattern, target in default_map.items():
        if not target:
            continue

        if target in all_by_code:
            resolved[pattern] = target
            continue

        leaf = target.rsplit(".", 1)[-1]
        leaf_norm = _normalize_token(leaf)
        if not leaf_norm:
            misses.append((pattern, target))
            continue

        abbrev_hit = by_abbrev.get(leaf) or by_abbrev.get(leaf.upper())
        if abbrev_hit is not None:
            resolved[pattern] = abbrev_hit.code
            continue

        alias_hit = None
        for cat in all_categories:
            common = getattr(cat, "common_names", "") or ""
            if not common:
                continue
            tokens = re.split(r"[|,;/]+", common)
            for tok in tokens:
                if _normalize_token(tok) == leaf_norm:
                    alias_hit = cat
                    break
            if alias_hit is not None:
                break

        if alias_hit is not None:
            resolved[pattern] = alias_hit.code
            continue

        misses.append((pattern, target))

    if misses:
        miss_summary = ", ".join(f"{p}->{t}" for p, t in misses)
        logger.warning(
            "Pattern source: %d target(s) not in active vocabulary; "
            "patterns disabled for this run: %s",
            len(misses), miss_summary,
        )

    return resolved


def pattern_to_mass(
    pattern_signals: dict[str, float] | list[str],
    frame: FrameOfDiscernment,
    pattern_category_map: dict[str, str] | None = None,
    *,
    theta_mass: float = 0.25,
) -> BeliefAssignment:
    """Convert detected pattern signals to a mass function.

    When *pattern_signals* is a ``dict[str, float]`` (from
    :func:`detect_patterns`), the match fraction scales each pattern's
    evidence mass — a 95% match produces more mass than a 35% match.
    When it is a plain ``list[str]`` (legacy), all matched patterns get
    equal mass at full strength.

    Args:
        pattern_signals: Detected patterns — either ``{name: fraction}``
            or ``[name, ...]`` (legacy compatibility).
        theta_mass: Fraction of total mass allocated to Theta.
    """
    if pattern_category_map is None:
        pattern_category_map = DEFAULT_PATTERN_MAP

    if not pattern_signals:
        return frame.vacuous()

    # Normalize input: list → dict with fraction=1.0
    if isinstance(pattern_signals, list):
        scored: dict[str, float] = {p: 1.0 for p in pattern_signals}
    else:
        scored = pattern_signals

    # Map patterns to codes, carrying match fractions
    matched: dict[str, float] = {}  # code → fraction
    for pattern, fraction in scored.items():
        code = pattern_category_map.get(pattern)
        if code and code in frame.singletons:
            # If multiple patterns map to the same code, take the max fraction
            matched[code] = max(matched.get(code, 0.0), fraction)

    if not matched:
        return frame.vacuous()

    # Scale evidence by average match fraction across matched patterns.
    # avg_fraction=1.0 when all values match → full evidence mass.
    # avg_fraction=0.4 when barely above threshold → reduced mass.
    avg_fraction = sum(matched.values()) / len(matched)
    evidence_mass = (1.0 - theta_mass) * avg_fraction
    actual_theta = 1.0 - evidence_mass

    mass_per_code = evidence_mass / len(matched)
    masses: dict[FocalElement, float] = {}
    for code in matched:
        masses[frame.singleton(code)] = mass_per_code

    masses[frame.theta] = actual_theta
    return BeliefAssignment(masses=masses)


# ── Column name matching ─────────────────────────────────────────────


# ── Column name abbreviation expansion ────────────────────────────

_ABBREVIATION_MAP: dict[str, list[str]] = {
    "medical_record_number": ["mrn", "med_rec", "medrecnum"],
    "social_security_number": ["ssn", "soc_sec", "socsec"],
    "date_of_birth": ["dob", "birthdate", "birth_date"],
    "phone_number": ["phone", "tel", "mobile", "cell"],
    "email_address": ["email", "e_mail", "mail"],
    "credit_card": ["cc", "creditcard", "pan"],
    "payment_card": ["pan", "card_num", "cardnum"],
    "first_name": ["fname", "given_name", "givennm"],
    "last_name": ["lname", "surname", "family_name"],
    "full_name": ["name", "fullname", "display_name"],
    "street_address": ["addr", "address", "street"],
    "postal_code": ["zip", "zipcode", "zip_code", "postcode"],
    "driver_license": ["dl", "driverslicense", "license_num"],
    "passport_number": ["passport", "passportnum"],
    "bank_account": ["ban", "account_num", "acctnum"],
    "ip_address": ["ip", "ipaddr", "ip_addr"],
    "user_agent": ["ua", "useragent", "browser"],
    "latitude": ["lat"],
    "longitude": ["lon", "lng", "long"],
}


def _expand_column_name(col_name: str) -> set[str]:
    """Generate abbreviation variants for column name matching.

    Produces acronyms (medical_record_number → mrn), truncations,
    and domain-specific aliases from the static map.
    """
    import re as _re
    words = _re.findall(r"[a-z]+", col_name.lower())
    variants: set[str] = {col_name.lower().replace("_", ""), "_".join(words)}

    # First-letter acronym: medical_record_number → mrn
    if len(words) >= 2:
        variants.add("".join(w[0] for w in words))

    # Static abbreviation map
    col_lower = col_name.lower()
    for key, abbrevs in _ABBREVIATION_MAP.items():
        if key in col_lower or col_lower in key:
            variants.update(abbrevs)

    return variants


def name_match_to_mass(
    column_name: str,
    frame: FrameOfDiscernment,
    category_set,
    *,
    exact_mass: float = 0.70,
    code_mass: float = 0.50,
    alias_mass: float = 0.50,
    overlap_mass: float = 0.30,
) -> BeliefAssignment:
    """Convert column name matching into a mass function.

    Matching levels (singleton mass, remainder goes to Theta):
    - Exact label match: *exact_mass* (default 0.70)
    - Formal code (abbrev) match: *code_mass* (default 0.50)
    - Common name alias match: *alias_mass* (default 0.50)
    - Word overlap match: *overlap_mass* (default 0.30)
    - No match: vacuous (1.0 Theta)
    """
    col_words = column_name.replace("_", " ").lower().strip()
    col_compact = col_words.replace(" ", "")
    col_word_set = set(col_words.split())

    # Generate abbreviation variants for broader matching
    col_variants = _expand_column_name(column_name)

    best_code: str | None = None
    best_mass = 0.0

    for cat in category_set.categories:
        if cat.code not in frame.singletons:
            continue

        cat_words = _camel_to_words(cat.label).replace("(", "").replace(")", "").strip()
        cat_abbrev = cat.abbrev.lower().strip()

        # Exact label match ("email address" == "email address")
        if col_words == cat_words:
            if exact_mass > best_mass:
                best_code = cat.code
                best_mass = exact_mass
        # Formal code match ("pan" == "pan", "ssn" == "ssn")
        elif cat_abbrev and col_compact == cat_abbrev:
            if code_mass > best_mass:
                best_code = cat.code
                best_mass = code_mass
        # Common name alias match ("credit card" in "Credit Card|CC|DPAN")
        elif hasattr(cat, "common_names") and cat.common_names:
            aliases = [a.strip().lower() for a in cat.common_names.split("|")]
            if not aliases or not aliases[0]:
                aliases = [a.strip().lower() for a in cat.common_names.split(",")]
            if col_words in aliases or col_compact in aliases:
                if alias_mass > best_mass:
                    best_code = cat.code
                    best_mass = alias_mass
            else:
                # Word overlap from label
                cat_word_set = set(cat_words.split())
                if len(cat_word_set) > 1 and cat_word_set.issubset(col_word_set):
                    if overlap_mass > best_mass:
                        best_code = cat.code
                        best_mass = overlap_mass
        else:
            # Word overlap from label
            cat_word_set = set(cat_words.split())
            if len(cat_word_set) > 1 and cat_word_set.issubset(col_word_set):
                if overlap_mass > best_mass:
                    best_code = cat.code
                    best_mass = overlap_mass

    # Abbreviation variant matching (if no better match found)
    if best_mass < alias_mass:
        for cat in category_set.categories:
            if cat.code not in frame.singletons:
                continue
            cat_abbrev = cat.abbrev.lower().strip()
            cat_aliases = set()
            if hasattr(cat, "common_names") and cat.common_names:
                for sep in ["|", ","]:
                    cat_aliases.update(a.strip().lower() for a in cat.common_names.split(sep) if a.strip())
            cat_aliases.add(cat_abbrev)
            # Check if any column variant matches any category alias/code
            if col_variants & cat_aliases:
                if alias_mass > best_mass:
                    best_code = cat.code
                    best_mass = alias_mass
                    break

    if best_code is None:
        return frame.vacuous()

    masses: dict[FocalElement, float] = {
        frame.singleton(best_code): best_mass,
        frame.theta: 1.0 - best_mass,
    }
    return BeliefAssignment(masses=masses)


# ── LLM classification ───────────────────────────────────────────────


def _coerce_to_singleton(
    code: str,
    frame: FrameOfDiscernment,
) -> str | None:
    """Try to resolve a non-singleton code to a valid leaf code.

    Handles three common LLM failure modes:
    1. Code is already a singleton → return as-is.
    2. Code is a known internal node with a unique leaf descendant
       → return that descendant.
    3. Code is a prefix of exactly one singleton → return it.

    Returns None when the code is unresolvable or ambiguous.
    """
    if code in frame.singletons:
        return code

    # Check internal nodes (parent codes) for unique-descendant resolution
    if code in frame.internal_nodes:
        fe = frame.internal_nodes[code]
        if len(fe.codes) == 1:
            leaf = next(iter(fe.codes))
            return leaf if leaf in frame.singletons else None
        # Ambiguous parent (multiple descendants) — can't choose one
        return None

    # Prefix matching: "1.1.1.9.3" might resolve to "1.1.1.9.3.1"
    prefix = code + "."
    matches = [s for s in frame.singletons if s.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]

    return None


def llm_to_mass(
    category_code: str | None,
    confidence: float,
    alternatives: list[dict],
    frame: FrameOfDiscernment,
    discount: float = 0.10,
) -> BeliefAssignment:
    """Convert LLM classification to a mass function.

    The primary prediction receives confidence-proportional mass.
    Alternatives distribute remaining evidence mass.  Low discount
    (0.10 vs cosine's 0.30) because frontier LLM predictions are
    well-informed and typically well-calibrated.

    When the LLM returns a parent code or near-miss code, coercion
    attempts to resolve it to a valid leaf singleton.  Unresolvable
    codes produce a vacuous assignment.

    Args:
        category_code: Predicted category code (None if unclassified).
        confidence: LLM confidence 0.0-1.0 for primary prediction.
        alternatives: List of dicts with 'code' and 'confidence' keys.
        frame: The frame of discernment.
        discount: Fraction of total mass allocated to Theta.
    """
    if not category_code:
        return frame.vacuous()

    # Coerce non-singleton codes (parent codes, near-misses) to valid leaves
    resolved = _coerce_to_singleton(category_code, frame)
    if resolved is None:
        import logging
        logging.getLogger(__name__).debug(
            "LLM code %r not in frame singletons and unresolvable — vacuous",
            category_code,
        )
        return frame.vacuous()

    evidence_mass = 1.0 - discount
    masses: dict[FocalElement, float] = {}

    # Primary prediction
    primary_mass = confidence * evidence_mass
    masses[frame.singleton(resolved)] = primary_mass

    # Distribute remaining evidence to alternatives (also coerced)
    remaining = evidence_mass - primary_mass
    if remaining > 1e-15 and alternatives:
        alt_total = sum(a.get("confidence", 0.0) for a in alternatives)
        if alt_total > 0:
            for alt in alternatives:
                alt_code = alt.get("code", "")
                alt_conf = alt.get("confidence", 0.0)
                alt_resolved = _coerce_to_singleton(alt_code, frame) if alt_code else None
                if alt_resolved and alt_conf > 0:
                    alt_mass = (alt_conf / alt_total) * remaining
                    if alt_mass > 1e-15:
                        fe = frame.singleton(alt_resolved)
                        masses[fe] = masses.get(fe, 0.0) + alt_mass

    # Theta gets discount + any unallocated evidence
    assigned = sum(masses.values())
    masses[frame.theta] = discount + max(0.0, evidence_mass - assigned)
    masses = _redistribute_confusable_mass(masses, frame)
    return BeliefAssignment(masses=masses)


# ── ML classifiers (CatBoost + SVM) ─────────────────────────────────


def catboost_to_mass(
    proba: dict[str, float],
    frame: FrameOfDiscernment,
    virtual_ensembles_variance: dict[str, float] | None = None,
    *,
    base_discount: float = 0.10,
    variance_scale: float = 1.6,
    max_discount: float = 0.50,
    fallback_discount: float = 0.15,
) -> BeliefAssignment:
    """Convert CatBoost predicted probabilities to a DST mass function.

    Uses adaptive discounting driven by virtual ensemble variance:
    high variance (uncertain) → larger discount → more mass on Theta.

    Args:
        proba: {category_code: probability} from CatBoost.
        frame: Frame of discernment.
        virtual_ensembles_variance: Optional {code: variance} from
            CatBoost virtual ensembles for uncertainty quantification.
        base_discount: Base discount before variance adjustment.
        variance_scale: Multiplier for average variance.
        max_discount: Maximum adaptive discount.
        fallback_discount: Discount when no variance data available.
    """
    if not proba:
        return frame.vacuous()

    # Adaptive discount from virtual ensemble variance
    if virtual_ensembles_variance:
        avg_var = sum(virtual_ensembles_variance.values()) / len(virtual_ensembles_variance)
        discount = min(max_discount, base_discount + avg_var * variance_scale)
    else:
        discount = fallback_discount

    masses: dict[FocalElement, float] = {}
    evidence_mass = 1.0 - discount
    for code, prob in proba.items():
        if code in frame.singletons and prob > 1e-15:
            masses[frame.singleton(code)] = prob * evidence_mass

    assigned = sum(masses.values())
    masses[frame.theta] = discount + max(0.0, evidence_mass - assigned)
    masses = _redistribute_confusable_mass(masses, frame)
    return BeliefAssignment(masses=masses)


def svm_to_mass(
    proba: dict[str, float],
    frame: FrameOfDiscernment,
    discount: float = 0.20,
) -> BeliefAssignment:
    """Convert SVM calibrated probabilities to a mass function.

    Converts Platt-scaled (CalibratedClassifierCV) probability estimates
    from a TF-IDF + LinearSVC classifier into a BeliefAssignment.  The
    SVM operates on sparse lexical features (character/word n-grams),
    making it architecturally independent from the dense sentence-
    transformer embedding shared by cosine and CatBoost sources.

    The discount is lower than cosine (0.30) because calibrated SVM
    probabilities tend to be well-concentrated on the correct class
    for short-text classification tasks.

    When the frame has confusable pairs and the top-2 singletons form
    a known pair with a close mass ratio, mass is redistributed to
    the pair focal element.

    Args:
        proba: {category_code: probability} from SVM predict_proba().
        frame: The frame of discernment.
        discount: Fraction of total mass allocated to Theta.
    """
    if not proba:
        return frame.vacuous()

    masses: dict[FocalElement, float] = {}
    evidence_mass = 1.0 - discount
    for code, prob in proba.items():
        if code in frame.singletons and prob > 1e-15:
            masses[frame.singleton(code)] = prob * evidence_mass

    assigned = sum(masses.values())
    masses[frame.theta] = discount + max(0.0, evidence_mass - assigned)
    masses = _redistribute_confusable_mass(masses, frame)
    return BeliefAssignment(masses=masses)
