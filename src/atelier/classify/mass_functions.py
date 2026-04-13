"""Evidence-to-mass converters for Dempster-Shafer belief functions.

Each converter transforms a specific evidence source into a BeliefAssignment
(mass function) compatible with Dempster's rule of combination.

Ported from signals/src/sigint/mass_functions.py — 3 of 5 sources implemented
in M0 (cosine, pattern, name_match). CatBoost and SVM return vacuous (stubs).
"""

from __future__ import annotations

import math
import re

from atelier.classify.belief import BeliefAssignment, FocalElement, FrameOfDiscernment


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


# Pattern → annotation code mapping (populated by pipeline from vocabulary)
DEFAULT_PATTERN_MAP: dict[str, str] = {}


def pattern_to_mass(
    pattern_signals: list[str],
    frame: FrameOfDiscernment,
    pattern_category_map: dict[str, str] | None = None,
) -> BeliefAssignment:
    """Convert detected pattern signals to a mass function.

    Maps pattern names to category codes. When no patterns are detected,
    returns a vacuous mass function.
    """
    if pattern_category_map is None:
        pattern_category_map = DEFAULT_PATTERN_MAP

    if not pattern_signals:
        return frame.vacuous()

    matched_codes: set[str] = set()
    for pattern in pattern_signals:
        code = pattern_category_map.get(pattern)
        if code and code in frame.singletons:
            matched_codes.add(code)

    if not matched_codes:
        return frame.vacuous()

    mass_per_code = 0.9 / len(matched_codes)
    masses: dict[FocalElement, float] = {}
    for code in matched_codes:
        masses[frame.singleton(code)] = mass_per_code

    masses[frame.theta] = 0.1
    return BeliefAssignment(masses=masses)


# ── Column name matching ─────────────────────────────────────────────


def name_match_to_mass(
    column_name: str,
    frame: FrameOfDiscernment,
    category_set,
) -> BeliefAssignment:
    """Convert column name matching into a mass function.

    Matching levels:
    - Exact label match: 0.7 singleton + 0.3 Theta
    - Formal code (abbrev) match: 0.5 singleton + 0.5 Theta
    - Common name alias match: 0.5 singleton + 0.5 Theta
    - Word overlap match: 0.3 singleton + 0.7 Theta
    - No match: vacuous (1.0 Theta)
    """
    col_words = column_name.replace("_", " ").lower().strip()
    col_compact = col_words.replace(" ", "")
    col_word_set = set(col_words.split())

    best_code: str | None = None
    best_mass = 0.0

    for cat in category_set.categories:
        if cat.code not in frame.singletons:
            continue

        cat_words = _camel_to_words(cat.label).replace("(", "").replace(")", "").strip()
        cat_abbrev = cat.abbrev.lower().strip()

        # Exact label match ("email address" == "email address")
        if col_words == cat_words:
            if 0.7 > best_mass:
                best_code = cat.code
                best_mass = 0.7
        # Formal code match ("pan" == "pan", "ssn" == "ssn")
        elif cat_abbrev and col_compact == cat_abbrev:
            if 0.5 > best_mass:
                best_code = cat.code
                best_mass = 0.5
        # Common name alias match ("credit card" in "Credit Card|CC|DPAN")
        elif hasattr(cat, "common_names") and cat.common_names:
            aliases = [a.strip().lower() for a in cat.common_names.split("|")]
            if not aliases or not aliases[0]:
                aliases = [a.strip().lower() for a in cat.common_names.split(",")]
            if col_words in aliases or col_compact in aliases:
                if 0.5 > best_mass:
                    best_code = cat.code
                    best_mass = 0.5
            else:
                # Word overlap from label
                cat_word_set = set(cat_words.split())
                if len(cat_word_set) > 1 and cat_word_set.issubset(col_word_set):
                    if 0.3 > best_mass:
                        best_code = cat.code
                        best_mass = 0.3
        else:
            # Word overlap from label
            cat_word_set = set(cat_words.split())
            if len(cat_word_set) > 1 and cat_word_set.issubset(col_word_set):
                if 0.3 > best_mass:
                    best_code = cat.code
                    best_mass = 0.3

    if best_code is None:
        return frame.vacuous()

    masses: dict[FocalElement, float] = {
        frame.singleton(best_code): best_mass,
        frame.theta: 1.0 - best_mass,
    }
    return BeliefAssignment(masses=masses)


# ── LLM classification ───────────────────────────────────────────────


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

    Args:
        category_code: Predicted category code (None if unclassified).
        confidence: LLM confidence 0.0-1.0 for primary prediction.
        alternatives: List of dicts with 'code' and 'confidence' keys.
        frame: The frame of discernment.
        discount: Fraction of total mass allocated to Theta.
    """
    if not category_code or category_code not in frame.singletons:
        return frame.vacuous()

    evidence_mass = 1.0 - discount
    masses: dict[FocalElement, float] = {}

    # Primary prediction
    primary_mass = confidence * evidence_mass
    masses[frame.singleton(category_code)] = primary_mass

    # Distribute remaining evidence to alternatives
    remaining = evidence_mass - primary_mass
    if remaining > 1e-15 and alternatives:
        alt_total = sum(a.get("confidence", 0.0) for a in alternatives)
        if alt_total > 0:
            for alt in alternatives:
                alt_code = alt.get("code", "")
                alt_conf = alt.get("confidence", 0.0)
                if alt_code in frame.singletons and alt_conf > 0:
                    alt_mass = (alt_conf / alt_total) * remaining
                    if alt_mass > 1e-15:
                        fe = frame.singleton(alt_code)
                        masses[fe] = masses.get(fe, 0.0) + alt_mass

    # Theta gets discount + any unallocated evidence
    assigned = sum(masses.values())
    masses[frame.theta] = discount + max(0.0, evidence_mass - assigned)
    masses = _redistribute_confusable_mass(masses, frame)
    return BeliefAssignment(masses=masses)


# ── Stubs for future ML classifiers ─────────────────────────────────


def catboost_to_mass(
    proba: dict[str, float],
    frame: FrameOfDiscernment,
    virtual_ensembles_variance: dict[str, float] | None = None,
) -> BeliefAssignment:
    """Stub — returns vacuous assignment."""
    return frame.vacuous()


def svm_to_mass(
    proba: dict[str, float],
    frame: FrameOfDiscernment,
    discount: float = 0.20,
) -> BeliefAssignment:
    """Stub — returns vacuous assignment."""
    return frame.vacuous()
