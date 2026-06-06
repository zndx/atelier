"""Subsumption-prediction alignment for the per-vocabulary SVM.

Replaces the LLM-mediated alignment in ``ontology_alignment.py`` with a
sentence-transformer cosine subsumption matcher.  Functional equivalent
of BertSubs (https://krr-oxford.github.io/DeepOnto/bertsubs/) — the MVP
uses cosine similarity over MiniLM embeddings as a lightweight
subsumption proxy between the ICE.* concept ontology and the operator's
in-situ user vocabulary.

Input:
  - ICE.* concept signatures: humanized name + 5 generator output samples
    (from ``synth_generators.GENERATORS``)
  - User-vocab concept signatures: enriched annotation payloads (label,
    description, prototype_values, name_hints) from the Qdrant collection
    registered in ``taxonomy_registry``

Output:
  - ``dict[str, str]`` mapping ICE leaf code → user taxonomy code
    (same shape as the predecessor LLM-mediated alignment)

DST independence impact:
  The alignment no longer shares LLM weights or inference-time provenance
  with the runtime classify-time LLM sweep.  The weak non-distinctness
  shifts from "shared alignment LLM" to "shared enrichment-LLM upstream"
  — the same structural dependency the late-interaction cosine source
  already carries (enrichment was generated offline by an LLM, but the
  runtime embedding + scoring is structurally independent).  This allows
  a lower discount than the LLM-mediated predecessor (0.22 vs 0.30).

See also:
  - ``docs/src/architecture/dst-evidence-independence.md``
  - ``docs/src/architecture/late-interaction-cosine.md``
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from atelier.classify.taxonomy import HierarchicalCategorySet

logger = logging.getLogger(__name__)


_DEFAULT_CACHE_DIR = Path("build/cache/alignment")
_SAMPLES_PER_ICE_LEAF = 5
_DEFAULT_SCORE_THRESHOLD = 0.35
_METHOD_TAG = "subsumption_st_v1"


# ── Cache key ───────────────────────────────────────────────────────


def _alignment_cache_key(
    ice_leaves: list[str],
    user_codes: list[str],
    embedding_model: str,
    method: str,
) -> str:
    """Stable sha256 over (ICE leaves, user codes, embedding model, method)."""
    payload = json.dumps(
        {
            "ice_leaves": sorted(ice_leaves),
            "user_codes": sorted(user_codes),
            "embedding_model": embedding_model,
            "method": method,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ── ICE-side concept signatures ─────────────────────────────────────


def _ice_to_humanized_name(ice_code: str) -> str:
    """Convert ICE.SENSITIVE.PID.IDENTITY.GOVID.SSN -> 'ssn (govid identity pid)'.

    Replicates the logic from ``ontology_alignment._ice_to_humanized_name``
    to maintain alignment-text consistency with the existing synth corpus.
    """
    parts = ice_code.split(".")
    tail = parts[2:] if len(parts) > 2 else parts
    if not tail:
        return ice_code
    leaf = tail[-1].lower()
    context = " ".join(p.lower() for p in tail[:-1])
    return f"{leaf} ({context})" if context else leaf


def _build_ice_concept_signatures() -> dict[str, str]:
    """Build concept-text signatures for each ICE leaf that has a generator.

    Signature format: "{humanized_name} | {sample1}, {sample2}, ..."
    The generated samples give the embedding model distributional context
    — the same column-value-shaped information the LLM alignment got.
    """
    import random
    from atelier.classify.synth_generators import GENERATORS

    rng = random.Random(0xA11167)  # same seed as legacy alignment
    signatures: dict[str, str] = {}

    for ice_code in sorted(GENERATORS.keys()):
        gen = GENERATORS[ice_code]
        try:
            values = [str(gen(rng)) for _ in range(_SAMPLES_PER_ICE_LEAF)]
        except Exception as exc:
            logger.debug(
                "subsumption_alignment: generator %s raised — skipping: %s",
                ice_code, exc,
            )
            continue
        humanized = _ice_to_humanized_name(ice_code)
        sig = f"{humanized} | {', '.join(values)}"
        signatures[ice_code] = sig

    return signatures


# ── User-vocab concept signatures (from Qdrant enrichment) ──────────


def _load_enriched_concept_signatures_from_index(
    annotation_index,
) -> dict[str, str]:
    """Extract concept-text signatures from an in-memory annotation index.

    For each annotation, builds a rich text representation from the
    payload fields.  The text is designed for embedding comparison
    against ICE concept signatures.

    Signature format:
      "{label} | {description} | values: {v1}, {v2}, ... | names: {n1}, {n2}, ..."
    """
    signatures: dict[str, str] = {}

    for view in annotation_index.views:
        code = view.code
        if not code:
            continue

        label = view.label or code
        signatures[code] = label

    return signatures


def _load_enriched_concept_signatures_from_qdrant(
    cfg,
    taxonomy_id: str | None = None,
) -> dict[str, str] | None:
    """Load enriched annotation payloads directly from Qdrant collection.

    Returns {user_code: concept_text} or None on failure.
    Mirrors the resolution pattern from maxsim_bridge.py.
    """
    try:
        from atelier.db.dao import AtelierDao
    except Exception as exc:
        logger.debug("subsumption_alignment: AtelierDao unavailable: %s", exc)
        return None

    tax_id = taxonomy_id or _resolve_taxonomy_id(cfg)
    try:
        dao = AtelierDao()
        row = dao.get_current_taxonomy_collection(tax_id)
    except Exception as exc:
        logger.debug(
            "subsumption_alignment: taxonomy lookup failed for %s: %s",
            tax_id, exc,
        )
        return None

    if row is None:
        logger.debug(
            "subsumption_alignment: no current collection for taxonomy_id=%s",
            tax_id,
        )
        return None

    qdrant_url = row.get("qdrant_url") or ""
    collection = row.get("qdrant_collection") or ""
    if not collection:
        return None

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        logger.warning(
            "subsumption_alignment: qdrant-client not installed — "
            "cannot load enriched annotations"
        )
        return None

    try:
        client = QdrantClient(url=qdrant_url or "http://127.0.0.1:6333")
    except Exception as exc:
        logger.warning("subsumption_alignment: Qdrant connect failed: %s", exc)
        return None

    # Scroll through collection extracting text payloads
    signatures: dict[str, str] = {}
    next_offset = None

    while True:
        try:
            points, next_offset = client.scroll(
                collection_name=collection,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,  # text only — we'll embed ourselves
            )
        except Exception as exc:
            logger.warning(
                "subsumption_alignment: Qdrant scroll failed: %s", exc,
            )
            return None if not signatures else signatures

        for p in points:
            payload = p.payload or {}
            code = payload.get("code") or payload.get("mnemonic")
            if not code:
                continue

            label = payload.get("label", "")
            description = payload.get("description", "")
            prototype_values = payload.get("prototype_values", [])
            name_hints = payload.get("name_hints", [])

            # Build concept text matching the ICE signature density
            parts = [label or code]
            if description:
                parts.append(description)
            if prototype_values:
                vals = prototype_values[:5] if isinstance(prototype_values, list) else []
                if vals:
                    parts.append(f"values: {', '.join(str(v) for v in vals)}")
            if name_hints:
                hints = name_hints[:5] if isinstance(name_hints, list) else []
                if hints:
                    parts.append(f"names: {', '.join(str(h) for h in hints)}")

            signatures[code] = " | ".join(parts)

        if next_offset is None:
            break

    return signatures if signatures else None


def _resolve_taxonomy_id(cfg) -> str:
    """Derive taxonomy_id from config — mirrors maxsim_bridge."""
    if cfg is None:
        return "default"
    explicit = getattr(cfg, "classify_taxonomy_id", None)
    return explicit or "default"


def _enumerate_all_codes(category_set) -> frozenset[str]:
    """Enumerate every code in the user taxonomy — leaves AND internal nodes.

    ``HierarchicalCategorySet`` exposes the full tree via
    ``all_by_code``.  Plain ``CategorySet`` only carries leaves under
    ``.categories``, so we fall back to that when the hierarchical
    accessor is absent.

    Per the dynamic-annotations + P3.8 hierarchical-integrity discipline,
    every node is a first-class tagging target.  Alignment-target
    enumeration must reflect that: an ICE leaf may align to a user
    internal node when the user's vocabulary has the family but not the
    specific leaf.  Restricting to leaves is the regression this helper
    fixes.
    """
    # HierarchicalCategorySet path — gives leaves + internals together
    all_by_code = getattr(category_set, "all_by_code", None)
    if all_by_code:
        return frozenset(all_by_code.keys())
    # Plain CategorySet fallback — leaves only is correct when there's
    # no hierarchy to walk in the first place
    categories = getattr(category_set, "categories", [])
    return frozenset(c.code for c in categories if getattr(c, "code", None))


def _classify_node_kind(
    category_set, code: str,
) -> str:
    """Return 'leaf' | 'internal' for an arbitrary user code.

    Used for the alignment-entry's ``target_kind`` annotation in the
    cache, so downstream auditing can see the leaf/internal alignment
    ratio without a re-walk.
    """
    leaf_codes = getattr(category_set, "leaf_codes", None)
    if leaf_codes is not None and code in leaf_codes:
        return "leaf"
    if leaf_codes is not None:
        return "internal"
    # Plain CategorySet — every category is effectively a leaf
    return "leaf"


# ── Cosine alignment computation ────────────────────────────────────


def _cosine_argmax_alignment(
    ice_signatures: dict[str, str],
    user_signatures: dict[str, str],
    score_threshold: float,
) -> tuple[dict[str, str], dict[str, dict]]:
    """Compute cosine-similarity alignment between ICE and user concepts.

    For each ICE concept, finds the user concept with highest cosine
    similarity.  If the max similarity is below ``score_threshold``,
    the ICE concept is left unmapped (the SVM won't contribute mass
    for that concept — same as the legacy vacuous fallback).

    This is the core subsumption-prediction step: "ICE.X is subsumed by
    user-Y" iff the embedding spaces overlap above threshold.

    Returns
    -------
    alignment : dict[str, str]
        Mapping ICE code -> user code (only for ICE concepts with a
        best match at or above ``score_threshold``).
    score_detail : dict[str, dict]
        Per ICE code, the alignment-confidence diagnostic record:
        ``{"top1_user": str, "top1_score": float, "top2_user": str|None,
        "top2_score": float|None}``.  Preserved for near-tie analysis
        and the validation harness without requiring a re-run.  Records
        exist for ICE concepts that DID NOT meet ``score_threshold``
        too — those carry ``top1_user`` (the would-have-been-pick) for
        diagnostic context.
    """
    from atelier.classify.embedding import embed_texts

    ice_codes = sorted(ice_signatures.keys())
    user_codes = sorted(user_signatures.keys())

    if not ice_codes or not user_codes:
        return {}, {}

    ice_texts = [ice_signatures[c] for c in ice_codes]
    user_texts = [user_signatures[c] for c in user_codes]

    # Embed both sets — MiniLM returns normalized vectors (unit L2)
    logger.info(
        "subsumption_alignment: embedding %d ICE + %d user concepts",
        len(ice_texts), len(user_texts),
    )
    ice_emb = np.array(embed_texts(ice_texts), dtype=np.float32)
    user_emb = np.array(embed_texts(user_texts), dtype=np.float32)

    # Cosine similarity matrix [N_ICE × N_user]
    # Since vectors are L2-normalized, cosine = dot product
    sim_matrix = ice_emb @ user_emb.T

    alignment: dict[str, str] = {}
    score_detail: dict[str, dict] = {}
    for i, ice_code in enumerate(ice_codes):
        row = sim_matrix[i]
        # Top-1 and top-2 in one pass — argsort is overkill for what we
        # need, but readability wins here.
        order = np.argsort(-row)
        best_j = int(order[0])
        best_score = float(row[best_j])
        second_j = int(order[1]) if len(order) > 1 else None
        second_score = float(row[second_j]) if second_j is not None else None

        score_detail[ice_code] = {
            "top1_user": user_codes[best_j],
            "top1_score": round(best_score, 4),
            "top2_user": user_codes[second_j] if second_j is not None else None,
            "top2_score": round(second_score, 4) if second_score is not None else None,
        }

        if best_score >= score_threshold:
            alignment[ice_code] = user_codes[best_j]
        else:
            logger.debug(
                "subsumption_alignment: %s best match %.3f < threshold %.3f — unmapped",
                ice_code, best_score, score_threshold,
            )

    return alignment, score_detail


# ── Public API ───────────────────────────────────────────────────────


def build_alignment_via_subsumption(
    category_set: "HierarchicalCategorySet",
    cfg=None,
    *,
    taxonomy_id: str | None = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> dict[str, str]:
    """Build the ICE.* -> user-taxonomy alignment via subsumption prediction.

    This is the replacement for the LLM-mediated alignment.  Uses
    sentence-transformer embeddings to compute cosine similarity between
    ICE concept signatures (humanized name + generator samples) and user
    vocabulary concept signatures (enriched annotations from Qdrant).

    Returns:
        dict mapping ICE.* leaf codes to user taxonomy codes.
        Empty dict on failure (SVM contributes vacuous mass — logged loudly).

    Raises no exceptions — all failures are caught, logged, and result
    in the empty-dict fallback with WARN-level messaging.

    Hierarchical integrity (load-bearing): the alignment target set is
    EVERY user code in the hierarchy — leaves AND internal nodes — per
    the dynamic-annotations principle that every node is a first-class
    tagging target.  An ICE leaf may legitimately align to a user
    internal node when the user's vocabulary covers a concept family
    without a leaf-specific equivalent.  Restricting to ``leaf_codes``
    would silently reject the parent-family fallback that is the
    architecturally-correct behavior.
    """
    user_codes = sorted(_enumerate_all_codes(category_set))
    if not user_codes:
        logger.warning(
            "subsumption_alignment: category_set has no codes — "
            "alignment cannot proceed"
        )
        return {}

    # Build ICE-side signatures
    ice_signatures = _build_ice_concept_signatures()
    if not ice_signatures:
        logger.warning(
            "subsumption_alignment: no ICE generators available — "
            "alignment cannot proceed"
        )
        return {}
    ice_leaves = sorted(ice_signatures.keys())

    # Check cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _alignment_cache_key(ice_leaves, user_codes, embedding_model, _METHOD_TAG)
    cache_path = cache_dir / f"{key}.json"

    if cache_path.exists():
        try:
            blob = json.loads(cache_path.read_text())
            if blob.get("method") == _METHOD_TAG:
                mapping = blob.get("alignment", {})
                if isinstance(mapping, dict):
                    logger.info(
                        "subsumption_alignment: loaded %d entries from cache (%s)",
                        len(mapping), cache_path,
                    )
                    return {str(k): str(v) for k, v in mapping.items() if v}
        except Exception as exc:
            logger.warning(
                "subsumption_alignment: cache read failed at %s: %s — rebuilding",
                cache_path, exc,
            )

    # Load user-side concept signatures from Qdrant enrichment.  The
    # Qdrant collection is expected to carry signatures for EVERY user
    # code — leaves and internals.  Missing-node coverage is a deployment
    # issue, not a normal operating mode (see no-silent-DST-degradation
    # discipline) and is surfaced loudly below.
    user_signatures = _load_enriched_concept_signatures_from_qdrant(
        cfg, taxonomy_id=taxonomy_id,
    )
    if user_signatures is None:
        logger.warning(
            "subsumption_alignment: could not load enriched annotations — "
            "alignment requires an enriched annotations collection registered "
            "in taxonomy_registry; run scripts/enrich_annotations.py first. "
            "SVM evidence will be vacuous on this run."
        )
        return {}

    # Filter to codes in the active category_set (all nodes — leaves +
    # internals).  Codes in Qdrant that don't appear in the category set
    # are silently dropped (stale enrichments from a prior vocab version,
    # or codes the operator removed).
    user_code_set = set(user_codes)
    user_signatures = {
        code: sig for code, sig in user_signatures.items()
        if code in user_code_set
    }
    if not user_signatures:
        logger.warning(
            "subsumption_alignment: enriched annotations found but none match "
            "category_set codes — check taxonomy_id alignment"
        )
        return {}

    # Loud-fail if the enrichment is incomplete vs the active vocabulary.
    # Per the no-silent-DST-degradation discipline, alignment running
    # against a partial enrichment is a deployment-degraded state.  We
    # proceed (the matcher's argmax remains valid over whatever subset
    # is available), but the gap is surfaced for operator triage.
    missing_codes = sorted(user_code_set - set(user_signatures.keys()))
    if missing_codes:
        sample = ", ".join(missing_codes[:10])
        ellipsis = "..." if len(missing_codes) > 10 else ""
        logger.warning(
            "subsumption_alignment: %d/%d user codes have no enriched "
            "signature (alignment proceeds against the covered subset; "
            "missing codes will not be reachable as alignment targets) — "
            "sample missing: %s%s",
            len(missing_codes), len(user_code_set), sample, ellipsis,
        )

    logger.info(
        "subsumption_alignment: computing alignment for %d ICE × %d user concepts "
        "(%d leaves + %d internals in user signatures)",
        len(ice_signatures), len(user_signatures),
        sum(1 for c in user_signatures if _classify_node_kind(category_set, c) == "leaf"),
        sum(1 for c in user_signatures if _classify_node_kind(category_set, c) == "internal"),
    )

    # Compute the alignment + per-ICE score detail
    alignment, score_detail = _cosine_argmax_alignment(
        ice_signatures, user_signatures, score_threshold,
    )

    # Build the reverse index (user_code -> [ice_codes]) for many-to-one
    # auditing.  Many-to-one is signal, not noise — the SVM benefits
    # from multiple ICE concepts contributing training rows to a single
    # user code.  Surfaces user codes with strong vs weak coverage.
    reverse_index: dict[str, list[str]] = {}
    for ice_code, user_code in alignment.items():
        reverse_index.setdefault(user_code, []).append(ice_code)
    for user_code in reverse_index:
        reverse_index[user_code].sort()

    # Per-alignment target_kind annotations (leaf vs internal)
    target_kinds = {
        ice_code: _classify_node_kind(category_set, user_code)
        for ice_code, user_code in alignment.items()
    }

    # Persist to cache
    try:
        cache_path.write_text(json.dumps({
            "method": _METHOD_TAG,
            "alignment": alignment,
            "target_kinds": target_kinds,
            "score_detail": score_detail,
            "user_code_to_ice_concepts": reverse_index,
            "ice_leaf_count": len(ice_leaves),
            "user_code_count": len(user_codes),
            "user_enriched_count": len(user_signatures),
            "user_codes_missing_enrichment": missing_codes,
            "embedding_model": embedding_model,
            "score_threshold": score_threshold,
            "mapped": len(alignment),
            "user_codes_covered": sorted(reverse_index.keys()),
            "user_codes_uncovered": sorted(
                set(user_signatures.keys()) - set(reverse_index.keys())
            ),
        }, indent=2) + "\n")
        logger.info(
            "subsumption_alignment: built %d/%d entries via %s (threshold=%.2f), "
            "covering %d/%d user codes — cached at %s",
            len(alignment), len(ice_leaves), embedding_model,
            score_threshold, len(reverse_index), len(user_codes), cache_path,
        )
    except Exception as exc:
        logger.warning("subsumption_alignment: cache write failed: %s", exc)

    return alignment
