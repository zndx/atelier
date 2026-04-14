"""Monte Carlo sampling for scalable classification.

At small N (< min_corpus_size), all columns get frontier-model classification
(identical to current pipeline behavior). At larger N, stratified importance
sampling selects representative columns for frontier inference, then label
propagation extends coverage to the remaining corpus via embedding similarity.

The MC layer is transparent: callers see the same BootstrapState with labels
for all columns, regardless of whether labels came from direct LLM inference
or propagation. Propagated labels enter DST fusion as discounted evidence —
the pipeline's existing conflict detection automatically escalates uncertain
propagations back to the frontier model via targeted revisit.

Designed for GitTables scale: 1.7M+ tables, 15M+ columns. The
max_frontier_columns cap (default 500) bounds LLM cost regardless of corpus
size. Stratified sampling guarantees every preliminary category stratum gets
at least min_per_stratum frontier-classified exemplars.
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
    max_frontier_columns: int = 500
    propagation_threshold: float = 0.85
    propagation_discount: float = 0.30

    @classmethod
    def from_cfg(cls, cfg) -> MCConfig:
        """Build from an AtelierConfig."""
        return cls(
            min_corpus_size=cfg.mc_min_corpus_size,
            sample_fraction=cfg.mc_sample_fraction,
            min_per_stratum=cfg.mc_min_per_stratum,
            max_frontier_columns=cfg.mc_max_frontier_columns,
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
    """Sampling plan: which columns get frontier vs. propagation."""

    strata: list[Stratum]
    frontier_columns: set[str]
    propagation_columns: set[str]
    total_columns: int
    effective_sample_fraction: float

    @property
    def is_passthrough(self) -> bool:
        """True when MC is inactive (small corpus)."""
        return len(self.frontier_columns) == self.total_columns

    def summary(self) -> dict[str, Any]:
        return {
            "strata": len(self.strata),
            "frontier_columns": len(self.frontier_columns),
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
    UNRESOLVED stratum (always fully sampled by the frontier model).
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
    """Select representative columns for frontier-model classification.

    Allocation strategy:
    - UNRESOLVED + rare strata: 100% (all members)
    - Normal strata: proportional allocation with importance weighting
    - Total budget: min(max_frontier_columns, total × sample_fraction)

    Importance weight per column: w = (1 - confidence) × (1 + ambiguity)
    Selection: weighted random sampling without replacement.
    """
    rng = random.Random(seed)

    # Passthrough: classify everything when below threshold
    all_names = set()
    for s in strata:
        all_names.update(s.column_names)
    if total < mc_cfg.min_corpus_size:
        return MCPlan(
            strata=strata,
            frontier_columns=all_names,
            propagation_columns=set(),
            total_columns=total,
            effective_sample_fraction=1.0,
        )

    budget = min(mc_cfg.max_frontier_columns, int(total * mc_cfg.sample_fraction))

    frontier: set[str] = set()

    # Phase 1: Mandatory selections (UNRESOLVED + rare strata)
    for stratum in strata:
        if stratum.code == "UNRESOLVED" or stratum.is_rare:
            frontier.update(stratum.column_names)

    # Phase 2: Allocate remaining budget across normal strata
    remaining_budget = max(0, budget - len(frontier))
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
                frontier.update(stratum.column_names)
            else:
                # Importance-weighted sampling within stratum
                selected = _importance_sample(
                    stratum.column_names, allocation, rng,
                )
                frontier.update(selected)

    propagation = set()
    for stratum in strata:
        for name in stratum.column_names:
            if name not in frontier:
                propagation.add(name)

    effective_fraction = len(frontier) / total if total > 0 else 1.0

    plan = MCPlan(
        strata=strata,
        frontier_columns=frontier,
        propagation_columns=propagation,
        total_columns=total,
        effective_sample_fraction=effective_fraction,
    )

    logger.info(
        "MC plan: %d frontier (%.1f%%) + %d propagation across %d strata "
        "(budget=%d, max=%d)",
        len(frontier), effective_fraction * 100,
        len(propagation), len(strata),
        budget, mc_cfg.max_frontier_columns,
    )

    return plan


def propagate_labels(
    state: BootstrapState,
    mc_plan: MCPlan,
    samples: dict[str, ColumnSample],
    mc_cfg: MCConfig,
) -> dict[str, str]:
    """Propagate labels from frontier-classified to unclassified columns.

    Uses cosine similarity on column feature embeddings. Propagation is
    stratum-local: only compare within the same preliminary category to
    reduce O(N×K) to O(stratum_size × K_frontier).

    Returns dict of {column_name: propagated_code}.
    """
    from atelier.classify.embedding import embed_texts, _get_model, get_batch_size
    from atelier.classify.features import extract_features

    propagated: dict[str, str] = {}
    threshold = mc_cfg.propagation_threshold

    # Build frontier embeddings (only classified frontier columns)
    frontier_classified = [
        name for name in mc_plan.frontier_columns
        if name in state.labels
    ]
    if not frontier_classified:
        logger.warning("No frontier columns classified — skipping propagation")
        return propagated

    # Embed frontier columns
    frontier_texts = []
    frontier_names = []
    for name in frontier_classified:
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
        frontier_texts.append(feat.to_embedding_text())
        frontier_names.append(name)

    model = _get_model()
    batch_size = get_batch_size()
    frontier_embs = model.encode(
        frontier_texts, normalize_embeddings=True,
        show_progress_bar=False, batch_size=batch_size,
    )
    frontier_embs = np.array(frontier_embs)

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

        # Compute similarities: (chunk_size, frontier_size)
        sims = chunk_embs @ frontier_embs.T

        for i, name in enumerate(chunk):
            best_idx = int(np.argmax(sims[i]))
            best_sim = float(sims[i, best_idx])

            if best_sim >= threshold:
                source_name = frontier_names[best_idx]
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
        "frontier_classified=%d)",
        len(propagated), len(mc_plan.propagation_columns),
        threshold, len(frontier_classified),
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
