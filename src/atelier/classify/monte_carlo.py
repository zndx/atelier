"""Monte Carlo sampling for scalable classification.

At small N (< min_corpus_size), every column gets direct LLM classification
(identical to current pipeline behavior).  At larger N, balance-first
stratified sampling selects a *balanced* subset of columns for direct LLM
classification — balanced coverage of the semantic landscape, NOT
corpus-representativeness (the directly-classified set is CatBoost's
fit-to-LLM training set, so any unseen semantic region is an unlearnable
class) — then label propagation extends coverage to the remaining corpus via
embedding similarity.

The MC layer is transparent: callers see the same BootstrapState with labels
for all columns, regardless of whether labels came from direct LLM inference
or propagation.  Propagated labels enter DST fusion as discounted evidence —
the pipeline's existing conflict detection automatically escalates uncertain
propagations back to direct LLM classification via targeted revisit.

Designed for GitTables scale: 1.7M+ tables, 15M+ columns.  The
``max_sampled_columns`` cap (default 500) bounds LLM cost regardless of corpus
size.  Stratified sampling guarantees every preliminary category stratum
contributes at least ``min_per_stratum`` directly-classified exemplars.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from atelier.classify.belief import FrameOfDiscernment
    from atelier.classify.bootstrap import BootstrapState
    from atelier.classify.sampler import ColumnSample
    from atelier.classify.taxonomy import HierarchicalCategorySet

logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────


@dataclass(frozen=True)
class MCConfig:
    """Monte Carlo sampling configuration from HOCON."""

    min_corpus_size: int = 200
    sample_fraction: float = 0.15
    min_per_stratum: int = 3
    max_sampled_columns: int = 500
    propagation_threshold: float = 0.85
    propagation_discount: float = 0.30

    @classmethod
    def from_cfg(cls, cfg) -> MCConfig:
        """Build from an AtelierConfig."""
        return cls(
            min_corpus_size=cfg.mc_min_corpus_size,
            sample_fraction=cfg.mc_sample_fraction,
            min_per_stratum=cfg.mc_min_per_stratum,
            max_sampled_columns=cfg.mc_max_sampled_columns,
            propagation_threshold=cfg.mc_propagation_threshold,
            propagation_discount=cfg.mc_propagation_discount,
        )


# ── Data Types ───────────────────────────────────────────────────


@dataclass
class PreClassification:
    """Cheap first-pass classification from M0 sources only."""

    column_name: str
    best_code: str  # Preliminary category code (or "UNRESOLVED")
    confidence: float  # Max cosine similarity to assigned category
    ambiguity: float  # sim_2nd / sim_1st ratio (higher = more ambiguous)
    pattern_codes: list[str] = field(default_factory=list)
    name_match_code: str | None = None


@dataclass
class Stratum:
    """Columns grouped by preliminary category."""

    code: str
    column_names: list[str]
    mean_confidence: float
    is_rare: bool = False

    @property
    def size(self) -> int:
        return len(self.column_names)


@dataclass
class MCPlan:
    """Sampling plan: which columns are LLM-classified directly vs. propagated.

    The plan splits the corpus into two sets: ``sampled_columns`` are
    classified by direct LLM inference; ``propagation_columns`` receive
    labels via embedding-similarity propagation from the nearest
    LLM-classified neighbor.  When the corpus is below
    ``min_corpus_size`` or ``sample_fraction >= 1.0``, ``sampled_columns``
    equals the full set and ``propagation_columns`` is empty.
    """

    strata: list[Stratum]
    sampled_columns: set[str]
    propagation_columns: set[str]
    total_columns: int
    effective_sample_fraction: float

    @property
    def is_passthrough(self) -> bool:
        """True when MC is inactive (small corpus or full-coverage mode)."""
        return len(self.sampled_columns) == self.total_columns

    def summary(self) -> dict[str, Any]:
        return {
            "strata": len(self.strata),
            "sampled_columns": len(self.sampled_columns),
            "propagation_columns": len(self.propagation_columns),
            "total_columns": self.total_columns,
            "effective_sample_fraction": round(self.effective_sample_fraction, 4),
            "is_passthrough": self.is_passthrough,
            "strata_detail": [
                {
                    "code": s.code,
                    "size": s.size,
                    "mean_confidence": round(s.mean_confidence, 4),
                    "is_rare": s.is_rare,
                }
                for s in self.strata
            ],
        }


# ── Core Functions ───────────────────────────────────────────────


def pre_classify(
    column_names: list[str],
    samples: dict[str, ColumnSample],
    category_set: HierarchicalCategorySet,
    frame: FrameOfDiscernment,
    has_embeddings: bool,
) -> dict[str, PreClassification]:
    """Cheap first-pass classification using M0 sources only.

    Runs name matching, pattern detection, and cosine similarity (no LLM,
    no CatBoost/SVM). This is the pre-classification step that feeds
    stratification.

    At 15M columns with GPU embeddings at 2,768 texts/s, this takes ~90
    minutes — acceptable as a one-time cost per run.
    """
    from atelier.classify.features import extract_features
    from atelier.classify.mass_functions import name_match_to_mass, pattern_to_mass

    results: dict[str, PreClassification] = {}

    # Batch cosine similarity for efficiency
    cosine_results: dict[str, dict[str, float]] = {}
    if has_embeddings:
        try:
            from atelier.classify.embedding import classify_cosine_batch
            features_list = []
            names_for_cosine = []
            for name in column_names:
                col = samples[name]
                feat = extract_features(
                    column_name=col.name,
                    column_type=col.column_type,
                    values=col.values,
                    siblings=col.siblings,
                    source_table=col.table_name,
                    total_count=col.total_count,
                    null_count=col.null_count,
                    distinct_count=col.distinct_count,
                )
                features_list.append(feat)
                names_for_cosine.append(name)
            batch_sims = classify_cosine_batch(features_list, category_set)
            for i, name in enumerate(names_for_cosine):
                cosine_results[name] = batch_sims[i]
        except ImportError:
            logger.warning("sentence-transformers not available for pre-classify")

    for name in column_names:
        col = samples[name]
        feat = extract_features(
            column_name=col.name,
            column_type=col.column_type,
            values=col.values,
            siblings=col.siblings,
            source_table=col.table_name,
            total_count=col.total_count,
            null_count=col.null_count,
            distinct_count=col.distinct_count,
        )

        # Name matching — extract best code
        name_mass = name_match_to_mass(col.name, frame, category_set)
        name_code = _best_singleton(name_mass)

        # Pattern detection — extract matched codes
        pattern_mass = pattern_to_mass(feat.pattern_signals, frame)
        pattern_codes = _pattern_codes(pattern_mass)

        # Cosine similarity — top-2 for confidence and ambiguity
        sims = cosine_results.get(name, {})
        if sims:
            sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
            best_code = sorted_sims[0][0]
            confidence = sorted_sims[0][1]
            ambiguity = sorted_sims[1][1] / max(confidence, 1e-10) if len(sorted_sims) > 1 else 0.0
        else:
            best_code = name_code or (pattern_codes[0] if pattern_codes else "UNRESOLVED")
            confidence = 0.5 if name_code else 0.0
            ambiguity = 1.0

        # Override best_code with strong name/pattern signals
        if name_code:
            best_code = name_code
            confidence = max(confidence, 0.7)

        results[name] = PreClassification(
            column_name=name,
            best_code=best_code,
            confidence=confidence,
            ambiguity=ambiguity,
            pattern_codes=pattern_codes,
            name_match_code=name_code,
        )

    logger.info(
        "Pre-classified %d columns: %d with cosine, %d with name match, %d with patterns",
        len(results),
        sum(1 for r in results.values() if r.confidence > 0),
        sum(1 for r in results.values() if r.name_match_code),
        sum(1 for r in results.values() if r.pattern_codes),
    )

    return results


def stratify(
    pre_results: dict[str, PreClassification],
    mc_cfg: MCConfig,
) -> list[Stratum]:
    """Partition columns into strata by preliminary category.

    Columns with low confidence or disagreeing signals go into an
    UNRESOLVED stratum (always fully classified by the LLM directly).
    """
    # Group by best_code
    groups: dict[str, list[PreClassification]] = {}
    for pre in pre_results.values():
        code = pre.best_code if pre.confidence > 0.3 else "UNRESOLVED"
        groups.setdefault(code, []).append(pre)

    strata = []
    for code, members in sorted(groups.items(), key=lambda x: -len(x[1])):
        mean_conf = sum(m.confidence for m in members) / len(members)
        is_rare = len(members) < mc_cfg.min_per_stratum * 2
        strata.append(Stratum(
            code=code,
            column_names=[m.column_name for m in members],
            mean_confidence=mean_conf,
            is_rare=is_rare,
        ))

    logger.info(
        "Stratified into %d strata: %d rare, %d normal, %d unresolved",
        len(strata),
        sum(1 for s in strata if s.is_rare and s.code != "UNRESOLVED"),
        sum(1 for s in strata if not s.is_rare and s.code != "UNRESOLVED"),
        sum(1 for s in strata if s.code == "UNRESOLVED"),
    )

    return strata


def select_sample(
    strata: list[Stratum],
    mc_cfg: MCConfig,
    total: int,
    seed: int = 42,
) -> MCPlan:
    """Select representative columns for direct LLM classification.

    Allocation strategy (balance-first):
    - UNRESOLVED + rare strata: 100% (all members)
    - Normal strata: proportional allocation over a ``min_per_stratum`` floor
    - Total budget: min(max_sampled_columns, total × sample_fraction)

    Within-stratum selection is uniform random today (see ``_importance_sample``).
    Importance weighting toward low-confidence / ambiguous columns —
    ``w = (1 - confidence) × (1 + ambiguity)`` — is the intended steering lever
    but is NOT yet wired (PreClassification confidence/ambiguity are not threaded
    into the selector); roadmap.
    """
    rng = random.Random(seed)

    # Passthrough: classify everything when below the size threshold OR
    # when the operator has explicitly requested full LLM coverage by
    # pushing sample_fraction to 1.0.  In the latter case we bypass the
    # max_sampled_columns cap as well — "100%" must literally mean 100%.
    all_names = set()
    for s in strata:
        all_names.update(s.column_names)
    if total < mc_cfg.min_corpus_size or mc_cfg.sample_fraction >= 1.0:
        return MCPlan(
            strata=strata,
            sampled_columns=all_names,
            propagation_columns=set(),
            total_columns=total,
            effective_sample_fraction=1.0,
        )

    budget = min(mc_cfg.max_sampled_columns, int(total * mc_cfg.sample_fraction))

    sampled: set[str] = set()

    # Phase 1: Mandatory selections (UNRESOLVED + rare strata)
    for stratum in strata:
        if stratum.code == "UNRESOLVED" or stratum.is_rare:
            sampled.update(stratum.column_names)

    # Phase 2: Allocate remaining budget across normal strata
    remaining_budget = max(0, budget - len(sampled))
    normal_strata = [s for s in strata if s.code != "UNRESOLVED" and not s.is_rare]
    normal_total = sum(s.size for s in normal_strata)

    if remaining_budget > 0 and normal_total > 0:
        for stratum in normal_strata:
            # Proportional allocation with min_per_stratum guarantee
            allocation = max(
                mc_cfg.min_per_stratum,
                int(remaining_budget * stratum.size / normal_total),
            )
            allocation = min(allocation, stratum.size)

            if allocation >= stratum.size:
                sampled.update(stratum.column_names)
            else:
                # Within-stratum selection: uniform today; importance
                # weighting is roadmap (see _importance_sample).
                selected = _importance_sample(
                    stratum.column_names, allocation, rng,
                )
                sampled.update(selected)

    propagation = set()
    for stratum in strata:
        for name in stratum.column_names:
            if name not in sampled:
                propagation.add(name)

    effective_fraction = len(sampled) / total if total > 0 else 1.0

    plan = MCPlan(
        strata=strata,
        sampled_columns=sampled,
        propagation_columns=propagation,
        total_columns=total,
        effective_sample_fraction=effective_fraction,
    )

    logger.info(
        "MC plan: %d sampled (%.1f%%) + %d propagation across %d strata "
        "(budget=%d, max=%d)",
        len(sampled), effective_fraction * 100,
        len(propagation), len(strata),
        budget, mc_cfg.max_sampled_columns,
    )

    return plan


def propagate_labels(
    state: BootstrapState,
    mc_plan: MCPlan,
    samples: dict[str, ColumnSample],
    mc_cfg: MCConfig,
) -> dict[str, str]:
    """Propagate labels from directly-classified to unclassified columns.

    Uses cosine similarity on column feature embeddings.  Propagation is
    *global* today: every directly-classified column is a candidate source
    (one source-embedding matrix; the nearest source above
    ``propagation_threshold`` wins).  Restricting the search to within-stratum
    (to cut O(N×K) to O(stratum_size × K_sampled) at corpus scale) is a planned
    optimization, not current behavior.

    Returns dict of {column_name: propagated_code}.
    """
    from atelier.classify.embedding import embed_texts, _get_model, get_batch_size
    from atelier.classify.features import extract_features

    propagated: dict[str, str] = {}
    threshold = mc_cfg.propagation_threshold

    # Build source embeddings (the LLM-classified sampled columns)
    sampled_classified = [
        name for name in mc_plan.sampled_columns
        if name in state.labels
    ]
    if not sampled_classified:
        logger.warning("No sampled columns classified — skipping propagation")
        return propagated

    # Embed the sampled columns; these become the propagation sources.
    sampled_texts = []
    sampled_names = []
    for name in sampled_classified:
        col = samples[name]
        feat = extract_features(
            column_name=col.name,
            column_type=col.column_type,
            values=col.values,
            siblings=col.siblings,
            source_table=col.table_name,
            total_count=col.total_count,
            null_count=col.null_count,
            distinct_count=col.distinct_count,
        )
        sampled_texts.append(feat.to_embedding_text())
        sampled_names.append(name)

    model = _get_model()
    batch_size = get_batch_size()
    sampled_embs = model.encode(
        sampled_texts, normalize_embeddings=True,
        show_progress_bar=False, batch_size=batch_size,
    )
    sampled_embs = np.array(sampled_embs)

    # Process propagation columns in chunks to manage memory
    prop_names = list(mc_plan.propagation_columns)
    chunk_size = max(1000, batch_size * 4)

    for chunk_start in range(0, len(prop_names), chunk_size):
        chunk = prop_names[chunk_start:chunk_start + chunk_size]

        chunk_texts = []
        for name in chunk:
            col = samples[name]
            feat = extract_features(
                column_name=col.name,
                column_type=col.column_type,
                values=col.values,
                siblings=col.siblings,
                source_table=col.table_name,
                total_count=col.total_count,
                null_count=col.null_count,
                distinct_count=col.distinct_count,
            )
            chunk_texts.append(feat.to_embedding_text())

        chunk_embs = model.encode(
            chunk_texts, normalize_embeddings=True,
            show_progress_bar=False, batch_size=batch_size,
        )
        chunk_embs = np.array(chunk_embs)

        # Compute similarities: (chunk_size, sampled_size)
        sims = chunk_embs @ sampled_embs.T

        for i, name in enumerate(chunk):
            best_idx = int(np.argmax(sims[i]))
            best_sim = float(sims[i, best_idx])

            if best_sim >= threshold:
                source_name = sampled_names[best_idx]
                source_code = state.labels[source_name]
                source_conf = state.confidence.get(source_name, 0.5)

                # Propagated label: same code, discounted confidence
                discounted_conf = source_conf * (1.0 - mc_cfg.propagation_discount)
                state.labels[name] = source_code
                state.confidence[name] = discounted_conf
                state.label_source[name] = "propagated"
                propagated[name] = source_code

    state.propagated_count = len(propagated)
    logger.info(
        "Propagated labels to %d/%d columns (threshold=%.2f, "
        "sampled_classified=%d)",
        len(propagated), len(mc_plan.propagation_columns),
        threshold, len(sampled_classified),
    )

    return propagated


# ── Helpers ──────────────────────────────────────────────────────


def _best_singleton(mass_assignment) -> str | None:
    """Extract the best singleton code from a BeliefAssignment."""
    best_code = None
    best_mass = 0.0
    for fe, m in mass_assignment.masses.items():
        if len(fe.codes) == 1 and m > best_mass:
            best_code = next(iter(fe.codes))
            best_mass = m
    return best_code


def _pattern_codes(mass_assignment) -> list[str]:
    """Extract singleton codes from a pattern mass assignment."""
    codes = []
    for fe, m in mass_assignment.masses.items():
        if len(fe.codes) == 1 and m > 0:
            codes.append(next(iter(fe.codes)))
    return codes


def _importance_sample(
    names: list[str],
    k: int,
    rng: random.Random,
) -> list[str]:
    """Weighted random sampling without replacement.

    Uniform weights for now — importance weighting from pre-classify
    confidence/ambiguity can be added when PreClassification data is
    threaded through.
    """
    if k >= len(names):
        return list(names)
    return rng.sample(names, k)
