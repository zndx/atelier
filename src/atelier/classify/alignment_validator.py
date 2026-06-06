"""Hybrid A/B validation harness for ontology alignment methods.

The first time a `(vocab_content_hash, alignment_method)` combination is
encountered, this module runs BOTH the new subsumption-prediction method
AND the legacy LLM-mediated method (one-time A/B).  The comparison is
persisted to ``build/cache/alignment_validation/`` so operators can
inspect alignment-agreement rates and classification-accuracy deltas.

Subsequent runs against the same vocabulary use only the configured
method.  Re-A/B fires on operator request (via ``scripts/validate_alignment.py``),
or automatically when the embedding model identifier or augmentation_version
changes.

The legacy LLM method is reachable ONLY via this validator — the
production classify path never calls it.

See the plan: `.claude/plans/lovely-doodling-badger.md` § P7.4.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atelier.classify.llm_backend import LLMBackend
    from atelier.classify.taxonomy import HierarchicalCategorySet

logger = logging.getLogger(__name__)

_DEFAULT_VALIDATION_DIR = Path("build/cache/alignment_validation")


@dataclass
class CoverageMetrics:
    """Per-method coverage diagnostics.

    Surfaced for both new and legacy alignment methods.  Asymmetric
    coverage is architecturally correct (ICE corpus is broader than any
    user vocabulary) — these metrics make the shape of that asymmetry
    visible to the operator.
    """

    ice_total: int
    ice_mapped: int
    ice_coverage_pct: float  # ice_mapped / ice_total

    user_codes_total: int          # all-nodes (leaves + internals) in vocab
    user_codes_covered: int        # user codes with ≥1 aligned ICE concept
    user_code_coverage_pct: float  # user_codes_covered / user_codes_total

    target_kind_distribution: dict[str, int]  # {"leaf": N, "internal": M}

    # Many-to-one histogram: {bucket_label: count_of_user_codes_in_bucket}
    # Buckets: "0" (uncovered), "1", "2", "3-5", "6-10", "11+"
    many_to_one_histogram: dict[str, int]

    # Near-tie diagnostic (only meaningful for embedding method; the
    # legacy LLM method records None).  Threshold ε is typically 0.05.
    near_tie_rate: float | None = None
    near_tie_threshold: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UncoveredUserCode:
    """One entry in the validation report's uncovered-user-codes list.

    This is the structured handoff for the deferred P7.9 Agent-SDK
    corpus-extension phase.  Each entry carries enough context for the
    Agent to reason about what generator function to author:

    - ``user_code`` + ``label`` + ``description`` identify the concept
    - ``enriched_annotation_snapshot`` provides the prototype values,
      patterns, name hints, and anti-examples the Agent uses to write
      a synth generator with verifier-passable output
    - ``parent_path`` provides hierarchical context (the new generator's
      placement in the BFO/CCO library should respect this)
    - ``closest_ice_candidates`` lists the top-3 ICE concepts whose
      cosine similarity to this user code was highest but still below
      the alignment threshold — these are starting points for the Agent
      to consider either extending or differentiating from
    """

    user_code: str
    label: str
    description: str
    parent_path: list[str]
    enriched_annotation_snapshot: dict[str, Any]
    closest_ice_candidates: list[dict[str, Any]]  # [{"ice_code": ..., "score": ...}, ...]
    kind: str  # "leaf" | "internal"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    """Results of an A/B alignment comparison.

    The report serves two consumers:
      1. Operators reviewing alignment quality (agreement rates, accuracy
         deltas).
      2. The deferred P7.9 Agent-SDK corpus-extension phase, which
         consumes ``uncovered_user_codes_new`` to author new synth
         generators for user codes the alignment couldn't reach.
    """

    vocab_hash: str
    method_new: str
    method_legacy: str
    timestamp: str
    alignment_new: dict[str, str]
    alignment_legacy: dict[str, str]
    agreement_rate: float  # fraction of ICE codes where both methods agree
    new_only_count: int    # ICE codes mapped by new but not legacy
    legacy_only_count: int  # ICE codes mapped by legacy but not new
    disagreement_details: list[dict[str, str]]  # per-ICE-code where they differ
    coverage_new: CoverageMetrics | None = None
    coverage_legacy: CoverageMetrics | None = None
    # P7.9 handoff: list of uncovered user codes with enrichment context
    # so the Agent-SDK corpus-extension phase has what it needs without
    # re-walking Qdrant.  Only the new method's uncovered list is
    # surfaced (legacy method's coverage is a baseline, not actionable).
    uncovered_user_codes_new: list[UncoveredUserCode] = field(default_factory=list)
    # Optional: if agent_mediated_path is set, classification accuracy
    accuracy_new: float | None = None
    accuracy_legacy: float | None = None
    accuracy_delta: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _vocab_content_hash(category_set: "HierarchicalCategorySet") -> str:
    """Stable hash of the vocabulary content for cache-key purposes.

    Uses all-nodes enumeration (leaves + internals) for hash identity,
    matching the alignment matcher's target set.  A vocab change that
    adds or removes internal nodes is a hash change — the validator
    will re-fire for the new shape.
    """
    from atelier.classify.subsumption_alignment import _enumerate_all_codes
    codes = sorted(_enumerate_all_codes(category_set))
    payload = json.dumps(codes, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ── Coverage metric computation ─────────────────────────────────────


def _bucket_count(n: int) -> str:
    """Histogram bucket label for many-to-one ratio."""
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if 3 <= n <= 5:
        return "3-5"
    if 6 <= n <= 10:
        return "6-10"
    return "11+"


def _compute_coverage(
    alignment: dict[str, str],
    *,
    category_set,
    ice_total: int,
    score_detail: dict[str, dict] | None = None,
    near_tie_threshold: float = 0.05,
) -> CoverageMetrics:
    """Compute per-method coverage metrics from an alignment dict.

    ``score_detail`` is the embedding-method's per-ICE diagnostic
    payload (top1/top2 scores).  Pass None for the legacy LLM method
    — near_tie_rate is then reported as None.
    """
    from atelier.classify.subsumption_alignment import _enumerate_all_codes, _classify_node_kind

    all_user_codes = _enumerate_all_codes(category_set)
    user_codes_total = len(all_user_codes)

    # User-code coverage
    covered_user_codes: set[str] = set(alignment.values())
    user_codes_covered = len(covered_user_codes & all_user_codes)

    # Target kind distribution
    kind_dist: dict[str, int] = {"leaf": 0, "internal": 0}
    for user_code in alignment.values():
        kind = _classify_node_kind(category_set, user_code)
        kind_dist[kind] = kind_dist.get(kind, 0) + 1

    # Many-to-one histogram: count user codes by how many ICE concepts
    # aligned to them
    user_to_count: dict[str, int] = {}
    for user_code in alignment.values():
        user_to_count[user_code] = user_to_count.get(user_code, 0) + 1
    # Include uncovered user codes in bucket "0"
    uncovered = all_user_codes - covered_user_codes
    for user_code in uncovered:
        user_to_count[user_code] = 0
    histogram: dict[str, int] = {
        "0": 0, "1": 0, "2": 0, "3-5": 0, "6-10": 0, "11+": 0,
    }
    for count in user_to_count.values():
        histogram[_bucket_count(count)] += 1

    # Near-tie rate (only embedding method)
    near_tie_rate: float | None = None
    if score_detail:
        tie_count = 0
        scored_count = 0
        for ice_code, detail in score_detail.items():
            top1 = detail.get("top1_score")
            top2 = detail.get("top2_score")
            if top1 is None or top2 is None:
                continue
            scored_count += 1
            if (top1 - top2) < near_tie_threshold:
                tie_count += 1
        if scored_count > 0:
            near_tie_rate = round(tie_count / scored_count, 4)

    return CoverageMetrics(
        ice_total=ice_total,
        ice_mapped=len(alignment),
        ice_coverage_pct=(
            round(100 * len(alignment) / ice_total, 2) if ice_total else 0.0
        ),
        user_codes_total=user_codes_total,
        user_codes_covered=user_codes_covered,
        user_code_coverage_pct=(
            round(100 * user_codes_covered / user_codes_total, 2)
            if user_codes_total else 0.0
        ),
        target_kind_distribution=kind_dist,
        many_to_one_histogram=histogram,
        near_tie_rate=near_tie_rate,
        near_tie_threshold=near_tie_threshold,
    )


def _build_uncovered_user_codes_list(
    alignment_new: dict[str, str],
    *,
    category_set,
    cfg=None,
    taxonomy_id: str | None = None,
    score_detail: dict[str, dict] | None = None,
) -> list[UncoveredUserCode]:
    """Build the structured P7.9 handoff list of uncovered user codes.

    For each user code with zero aligned ICE concepts (per the new
    method's alignment), produce an UncoveredUserCode entry carrying
    the enriched annotation payload + parent context + closest ICE
    candidates that fell just below the alignment threshold.

    Returns an empty list when the enriched payload can't be loaded —
    P7.9 can still proceed but with less context.
    """
    from atelier.classify.subsumption_alignment import (
        _enumerate_all_codes,
        _classify_node_kind,
        _load_enriched_concept_signatures_from_qdrant,
    )

    all_user_codes = _enumerate_all_codes(category_set)
    covered = set(alignment_new.values())
    uncovered = sorted(all_user_codes - covered)
    if not uncovered:
        return []

    # Reverse the score_detail: for each user code, find ICE concepts
    # whose top1 was that user code (even if below threshold).  Build
    # a {user_code: [(ice_code, score), ...]} index.
    user_to_close_ice: dict[str, list[tuple[str, float]]] = {}
    if score_detail:
        for ice_code, detail in score_detail.items():
            top1_user = detail.get("top1_user")
            top1_score = detail.get("top1_score")
            if not top1_user or top1_score is None:
                continue
            user_to_close_ice.setdefault(top1_user, []).append(
                (ice_code, float(top1_score))
            )
    # Sort each user's close-ICE list by score descending
    for user_code in user_to_close_ice:
        user_to_close_ice[user_code].sort(key=lambda x: -x[1])

    # Load enriched annotations to enrich the handoff entries (best
    # effort — Qdrant may not be reachable here either)
    enrichment_payloads: dict[str, dict] = {}
    try:
        from atelier.db.dao import AtelierDao
        from qdrant_client import QdrantClient

        tax_id = taxonomy_id or "default"
        if cfg is not None:
            tax_id = getattr(cfg, "classify_taxonomy_id", None) or tax_id
        dao = AtelierDao()
        row = dao.get_current_taxonomy_collection(tax_id)
        if row is not None:
            qdrant_url = row.get("qdrant_url") or "http://127.0.0.1:6333"
            collection = row.get("qdrant_collection") or ""
            if collection:
                client = QdrantClient(url=qdrant_url)
                next_offset = None
                while True:
                    points, next_offset = client.scroll(
                        collection_name=collection,
                        limit=256,
                        offset=next_offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    for p in points:
                        payload = p.payload or {}
                        code = payload.get("code") or payload.get("mnemonic")
                        if code and code in set(uncovered):
                            enrichment_payloads[code] = dict(payload)
                    if next_offset is None:
                        break
    except Exception as exc:
        logger.debug(
            "alignment_validator: enrichment payload load for uncovered codes failed: %s",
            exc,
        )

    # Build the structured entries
    by_code = getattr(category_set, "all_by_code", {})
    entries: list[UncoveredUserCode] = []
    for user_code in uncovered:
        cat = by_code.get(user_code) if hasattr(by_code, "get") else None
        label = getattr(cat, "label", "") if cat else ""
        # parent_path walk — uses category_set.ancestors if available
        parent_path: list[str] = []
        ancestors_fn = getattr(category_set, "ancestors", None)
        if callable(ancestors_fn):
            try:
                ancestors = ancestors_fn(user_code)
                # ancestors() returns parent → root order, reverse for root → parent
                parent_path = [
                    getattr(by_code.get(a), "label", a) for a in reversed(ancestors)
                ] if hasattr(by_code, "get") else list(reversed(ancestors))
            except Exception:
                parent_path = []

        payload = enrichment_payloads.get(user_code, {})

        # Closest ICE candidates: top-3 ICE concepts whose top1_user
        # was this user code (these ICE concepts WANTED to align here
        # but fell below threshold OR aligned and got booted).
        close = user_to_close_ice.get(user_code, [])[:3]
        candidates = [
            {"ice_code": ice, "top1_score": round(score, 4)}
            for ice, score in close
        ]

        entries.append(UncoveredUserCode(
            user_code=user_code,
            label=label or user_code,
            description=getattr(cat, "description", "") if cat else "",
            parent_path=parent_path,
            enriched_annotation_snapshot={
                k: v for k, v in payload.items()
                if k in (
                    "prototype_values", "value_patterns", "name_hints",
                    "anti_examples", "parent_path", "label", "description",
                )
            },
            closest_ice_candidates=candidates,
            kind=_classify_node_kind(category_set, user_code),
        ))

    return entries


def should_validate(
    cache_dir: Path,
    vocab_hash: str,
    method: str,
    embedding_model: str = "all-MiniLM-L6-v2",
    augmentation_version: str = "v1",
) -> bool:
    """Check whether this (vocab, method, model, version) has been validated.

    Returns True if no validation report exists for this combination —
    indicating the A/B comparison should run.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    pattern = f"{vocab_hash}-{method}-{embedding_model}-{augmentation_version}-*.json"
    existing = list(cache_dir.glob(pattern))
    return len(existing) == 0


def _run_legacy_llm_alignment(
    category_set: "HierarchicalCategorySet",
    llm_backend: "LLMBackend",
    system_prompt: str,
    model_name: str,
) -> dict[str, str]:
    """Run the legacy LLM-mediated alignment for comparison purposes.

    This is a self-contained reimplementation of the excised LLM alignment
    logic — NOT the production path.  It exists solely for A/B comparison.
    """
    import random
    from atelier.classify.sampler import ColumnSample
    from atelier.classify.synth_generators import GENERATORS
    from atelier.classify.subsumption_alignment import _ice_to_humanized_name

    _SAMPLES_PER_ICE_LEAF = 5
    rng = random.Random(0xA11167)

    ice_leaves = sorted(GENERATORS.keys())
    samples: list[ColumnSample] = []
    ice_for_samples: list[str] = []

    for ice in ice_leaves:
        gen = GENERATORS.get(ice)
        if gen is None:
            continue
        try:
            values = [str(gen(rng)) for _ in range(_SAMPLES_PER_ICE_LEAF)]
        except Exception:
            continue
        samples.append(
            ColumnSample(
                name=_ice_to_humanized_name(ice),
                column_type="STRING",
                values=values,
                table_name="_synth_alignment_validation",
            ),
        )
        ice_for_samples.append(ice)

    if not samples:
        return {}

    humanized_to_ice = {s.name: ice for s, ice in zip(samples, ice_for_samples)}

    try:
        response = llm_backend.classify_batch(
            samples=samples,
            system_prompt=system_prompt,
            table_name="_synth_alignment_validation",
        )
    except Exception as exc:
        logger.warning("alignment_validator: legacy LLM call failed: %s", exc)
        return {}

    # Parse response
    alignment: dict[str, str] = {}
    classifications = getattr(response, "classifications", None) or []
    for cls in classifications:
        humanized = getattr(cls, "column_name", None)
        user_code = getattr(cls, "category_code", None)
        if not humanized or not user_code:
            continue
        ice = humanized_to_ice.get(humanized)
        if ice is None:
            continue
        alignment[ice] = str(user_code)

    return alignment


def run_ab_validation(
    category_set: "HierarchicalCategorySet",
    cfg=None,
    *,
    llm_backend: "LLMBackend | None" = None,
    system_prompt: str | None = None,
    model_name: str | None = None,
    taxonomy_id: str | None = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    score_threshold: float = 0.35,
    cache_dir: Path = _DEFAULT_VALIDATION_DIR,
    force: bool = False,
) -> ValidationReport | None:
    """Run A/B validation comparing subsumption vs legacy LLM alignment.

    If the (vocab_hash, method) combo has already been validated and
    ``force`` is False, returns None (skip).

    Requires ``llm_backend`` to be provided for the legacy method;
    if not available, only the new method is run and the report
    records the legacy side as empty.

    Returns:
        ValidationReport with agreement rates and per-code details,
        or None if validation was already done and force=False.
    """
    from atelier.classify.subsumption_alignment import (
        _METHOD_TAG,
        build_alignment_via_subsumption,
    )

    vocab_hash = _vocab_content_hash(category_set)

    if not force and not should_validate(
        cache_dir, vocab_hash, _METHOD_TAG, embedding_model,
    ):
        logger.debug(
            "alignment_validator: validation already exists for vocab=%s method=%s",
            vocab_hash, _METHOD_TAG,
        )
        return None

    # Run the new method
    alignment_new = build_alignment_via_subsumption(
        category_set,
        cfg=cfg,
        taxonomy_id=taxonomy_id,
        embedding_model=embedding_model,
        score_threshold=score_threshold,
    )

    # Run the legacy method (if LLM backend available)
    alignment_legacy: dict[str, str] = {}
    if llm_backend is not None and system_prompt is not None:
        alignment_legacy = _run_legacy_llm_alignment(
            category_set, llm_backend, system_prompt,
            model_name or "unknown",
        )
    else:
        logger.info(
            "alignment_validator: no llm_backend provided — "
            "legacy comparison will be empty"
        )

    # Compute agreement metrics
    all_ice_codes = set(alignment_new.keys()) | set(alignment_legacy.keys())
    agreements = 0
    disagreements: list[dict[str, str]] = []
    new_only = 0
    legacy_only = 0

    for ice in sorted(all_ice_codes):
        new_val = alignment_new.get(ice)
        legacy_val = alignment_legacy.get(ice)

        if new_val == legacy_val and new_val is not None:
            agreements += 1
        elif new_val is not None and legacy_val is None:
            new_only += 1
        elif legacy_val is not None and new_val is None:
            legacy_only += 1
        elif new_val != legacy_val:
            disagreements.append({
                "ice_code": ice,
                "new": new_val or "",
                "legacy": legacy_val or "",
            })

    total_comparable = agreements + len(disagreements)
    agreement_rate = (
        agreements / total_comparable if total_comparable > 0 else 0.0
    )

    # Coverage metrics for both methods.  The new method's score_detail
    # is loaded from the cache the matcher just wrote.  Legacy has no
    # score_detail — its near_tie_rate is reported as None.
    new_score_detail: dict[str, dict] = {}
    try:
        from atelier.classify.subsumption_alignment import (
            _DEFAULT_CACHE_DIR,
            _alignment_cache_key,
            _build_ice_concept_signatures,
        )
        all_user_codes = sorted(
            __import__(
                "atelier.classify.subsumption_alignment", fromlist=["_enumerate_all_codes"]
            )._enumerate_all_codes(category_set)
        )
        ice_keys = sorted(_build_ice_concept_signatures().keys())
        cache_key = _alignment_cache_key(
            ice_keys, all_user_codes, embedding_model, _METHOD_TAG,
        )
        cache_path = _DEFAULT_CACHE_DIR / f"{cache_key}.json"
        if cache_path.exists():
            blob = json.loads(cache_path.read_text())
            new_score_detail = blob.get("score_detail", {}) or {}
    except Exception as exc:
        logger.debug(
            "alignment_validator: could not load score_detail for coverage metrics: %s",
            exc,
        )

    ice_total_for_coverage = (
        len(new_score_detail) if new_score_detail
        else max(len(alignment_new), 1)  # best effort denominator
    )
    coverage_new = _compute_coverage(
        alignment_new,
        category_set=category_set,
        ice_total=ice_total_for_coverage,
        score_detail=new_score_detail,
    )
    coverage_legacy = _compute_coverage(
        alignment_legacy,
        category_set=category_set,
        ice_total=ice_total_for_coverage,
        score_detail=None,  # legacy has no top1/top2 — near_tie_rate stays None
    )

    # P7.9 handoff: structured uncovered-user-codes list with enrichment
    # snapshot + closest ICE candidates per uncovered code
    uncovered_entries = _build_uncovered_user_codes_list(
        alignment_new,
        category_set=category_set,
        cfg=cfg,
        taxonomy_id=taxonomy_id,
        score_detail=new_score_detail,
    )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report = ValidationReport(
        vocab_hash=vocab_hash,
        method_new=_METHOD_TAG,
        method_legacy="llm_classify_batch_v1",
        timestamp=ts,
        alignment_new=alignment_new,
        alignment_legacy=alignment_legacy,
        agreement_rate=agreement_rate,
        new_only_count=new_only,
        legacy_only_count=legacy_only,
        disagreement_details=disagreements,
        coverage_new=coverage_new,
        coverage_legacy=coverage_legacy,
        uncovered_user_codes_new=uncovered_entries,
        metadata={
            "embedding_model": embedding_model,
            "score_threshold": score_threshold,
            "model_name": model_name,
            "ice_total_new": len(alignment_new),
            "ice_total_legacy": len(alignment_legacy),
            "uncovered_count_new": len(uncovered_entries),
        },
    )

    # Persist the report
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_filename = (
        f"{vocab_hash}-{_METHOD_TAG}-{embedding_model}-v1-{ts.replace(':', '')}.json"
    )
    report_path = cache_dir / report_filename
    try:
        report_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
        logger.info(
            "alignment_validator: A/B report saved — agreement=%.1f%% "
            "(%d agree, %d disagree, %d new-only, %d legacy-only) → %s",
            agreement_rate * 100, agreements, len(disagreements),
            new_only, legacy_only, report_path,
        )
    except Exception as exc:
        logger.warning("alignment_validator: report write failed: %s", exc)

    return report
