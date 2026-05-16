# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Enrichment loop: source taxonomy → enriched Qdrant collection.

Orchestrates the per-annotation work: generate, verify, retry-on-fail,
embed, write to Qdrant.  Embedding computation is delegated to a
caller-provided callable so the loop stays independent of the specific
embedding model.

See ``docs/src/architecture/late-interaction-cosine.md`` for the
end-to-end design; this module is the runtime composition of the
deterministic verifiers, the pluggable LLM generator, and the Qdrant
writer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

from atelier.enrichment.llm_generator import (
    EnrichmentGenerator,
    GenerationResult,
)
from atelier.enrichment.qdrant_writer import (
    AnnotationVectors,
    EnrichedAnnotationPoint,
    build_point,
    collection_name_for,
    ensure_collection,
    point_cache_key,
    point_exists,
    source_row_hash,
    taxonomy_version_hash,
    upsert_point,
)
from atelier.enrichment.verifiers import VerifierReport, run_verifier_suite

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


# ── Loop config + result types ────────────────────────────────────


@dataclass
class EnrichmentLoopConfig:
    """Tunables for the enrichment loop."""

    taxonomy_id: str
    augmentation_version: str
    embedding_model: str
    embedding_dim: int
    qdrant_url: str | None = None  # informational; the client owns the connection
    max_attempts_per_row: int = 3
    skip_on_cache_hit: bool = True
    recreate_collection: bool = False  # set True for clean rebuilds
    dry_run: bool = False  # if True, skip Qdrant writes


@dataclass
class RowOutcome:
    """Per-row result of an enrichment attempt."""

    code: str
    status: str  # 'enriched' | 'cache_hit' | 'verifier_failed' | 'generator_failed'
    attempts: int = 0
    verifier_checks_passed: int = 0
    verifier_checks_total: int = 0
    point_id: str | None = None
    failure_reason: str = ""


@dataclass
class LoopResult:
    """Aggregate result over a full enrichment run."""

    collection_name: str
    taxonomy_version_hash: str
    started_at: str
    finished_at: str = ""
    rows: list[RowOutcome] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.rows:
            out[r.status] = out.get(r.status, 0) + 1
        return out


# ── Embedding callable signature ──────────────────────────────────


# A callable that turns text(s) into embedding vectors.  Single string
# → single embedding (list[float]); list[str] → list[list[float]].
# Implementations are sentence-transformers wrappers, ONNX models, etc.
EmbedFn = Callable[[str | list[str]], list[float] | list[list[float]]]


# ── Main loop ─────────────────────────────────────────────────────


def run_enrichment(
    *,
    source_rows: list[dict],
    generator: EnrichmentGenerator,
    embed: EmbedFn,
    config: EnrichmentLoopConfig,
    qdrant_client: "QdrantClient | None",
    valid_tag_codes: set[str] | None = None,
    expected_parent_path_for: Callable[[dict], list[str] | None] | None = None,
) -> LoopResult:
    """Run the enrichment loop over ``source_rows``.

    Parameters
    ----------
    source_rows
        Each row is a dict with at least ``code``, ``label``, ``mnemonic``,
        ``description``, ``parent_code``.  Additional fields are passed
        through to the payload unchanged.
    generator
        An :class:`EnrichmentGenerator` implementation.  The loop calls
        ``generator.generate(...)`` per row with retry-on-verifier-fail
        semantics.
    embed
        Callable that produces embeddings for one or more strings.
    config
        :class:`EnrichmentLoopConfig` controlling collection naming,
        retry budgets, dry-run mode.
    qdrant_client
        Qdrant client to write to.  May be None in dry-run mode.
    valid_tag_codes
        Set of valid taxonomy codes/mnemonics for anti-example target
        verification.  When None, that verifier degrades to a pass.
    expected_parent_path_for
        Callable that returns the deterministic expected parent path
        for a row, used by the parent-path consistency verifier.
        When None, the parent-path check degrades to a pass.

    Returns
    -------
    LoopResult
        Aggregate per-row outcomes + run metadata.
    """
    if not config.dry_run and qdrant_client is None:
        raise ValueError("qdrant_client is required when dry_run=False")

    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    version_hash = taxonomy_version_hash(source_rows)
    collection = collection_name_for(config.taxonomy_id, config.augmentation_version)

    if not config.dry_run:
        ensure_collection(
            qdrant_client,
            collection=collection,
            embedding_dim=config.embedding_dim,
            recreate=config.recreate_collection,
        )

    result = LoopResult(
        collection_name=collection,
        taxonomy_version_hash=version_hash,
        started_at=started_at,
    )

    taxonomy_context = {
        "all_codes": list(valid_tag_codes) if valid_tag_codes else [],
        "row_count": len(source_rows),
    }

    for row in source_rows:
        code = row.get("code") or row.get("mnemonic") or "<unknown>"

        expected_parent = (
            expected_parent_path_for(row) if expected_parent_path_for else None
        )
        row_context = {**taxonomy_context, "expected_parent_path": expected_parent}

        # Cache-hit short circuit
        if config.skip_on_cache_hit and not config.dry_run:
            sr_hash = source_row_hash(row)
            pid = point_cache_key(
                taxonomy_id=config.taxonomy_id,
                taxonomy_version_hash_value=version_hash,
                augmentation_version=config.augmentation_version,
                embedding_model=config.embedding_model,
                source_row_hash_value=sr_hash,
            )
            if point_exists(qdrant_client, collection=collection, point_id=pid):
                result.rows.append(
                    RowOutcome(
                        code=code, status="cache_hit", attempts=0, point_id=pid
                    )
                )
                continue

        outcome = _enrich_one_row(
            row=row,
            generator=generator,
            embed=embed,
            config=config,
            row_context=row_context,
            valid_tag_codes=valid_tag_codes or set(),
            expected_parent_path=expected_parent,
            taxonomy_version_hash_value=version_hash,
            qdrant_client=qdrant_client,
            collection=collection,
        )
        result.rows.append(outcome)

    result.finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info(
        "Enrichment loop done: collection=%s outcomes=%s",
        collection,
        result.counts,
    )
    return result


def _enrich_one_row(
    *,
    row: dict,
    generator: EnrichmentGenerator,
    embed: EmbedFn,
    config: EnrichmentLoopConfig,
    row_context: dict,
    valid_tag_codes: set[str],
    expected_parent_path: list[str] | None,
    taxonomy_version_hash_value: str,
    qdrant_client: "QdrantClient | None",
    collection: str,
) -> RowOutcome:
    """Generate, verify, and write one row.  Retry on verifier failure."""
    code = row.get("code") or row.get("mnemonic") or "<unknown>"
    last_attempt: GenerationResult | None = None
    last_report: VerifierReport | None = None

    for attempt in range(1, config.max_attempts_per_row + 1):
        try:
            attempt_result = generator.generate(
                row,
                taxonomy_context=row_context,
                prior_attempt=last_attempt,
                verifier_feedback=last_report.to_dict() if last_report else None,
            )
        except NotImplementedError as exc:
            return RowOutcome(
                code=code, status="generator_failed",
                attempts=attempt, failure_reason=f"generator not implemented: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 — surface any generator error
            logger.warning("Generator failed for %s on attempt %d: %s", code, attempt, exc)
            return RowOutcome(
                code=code, status="generator_failed",
                attempts=attempt, failure_reason=str(exc),
            )

        last_attempt = attempt_result
        report = run_verifier_suite(
            attempt_result.enrichment,
            valid_tag_codes=valid_tag_codes,
            expected_parent_path=expected_parent_path,
        )
        last_report = report
        if report.passed:
            break
        logger.info(
            "Verifier failed for %s on attempt %d: failed=%s",
            code, attempt, [c.name for c in report.failed()],
        )

    assert last_attempt is not None and last_report is not None
    if not last_report.passed:
        return RowOutcome(
            code=code, status="verifier_failed",
            attempts=config.max_attempts_per_row,
            verifier_checks_passed=last_report.checks_passed,
            verifier_checks_total=last_report.checks_total,
            failure_reason=", ".join(c.name for c in last_report.failed()),
        )

    # Verified — now embed each view and build the point.
    vectors = _embed_views(last_attempt.enrichment, row=row, embed=embed)
    point = build_point(
        source_row=row,
        enrichment=last_attempt.enrichment,
        vectors=vectors,
        verifier_results=last_report.to_dict(),
        taxonomy_id=config.taxonomy_id,
        taxonomy_version_hash_value=taxonomy_version_hash_value,
        taxonomy_version_label=time.strftime("%Y-%m-%d"),
        augmentation_version=config.augmentation_version,
        embedding_model=config.embedding_model,
        embedding_dim=config.embedding_dim,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        generated_by=generator.name,
    )

    if not config.dry_run:
        upsert_point(qdrant_client, collection=collection, point=point)

    return RowOutcome(
        code=code, status="enriched", attempts=last_attempt.attempts,
        verifier_checks_passed=last_report.checks_passed,
        verifier_checks_total=last_report.checks_total,
        point_id=point.point_id,
    )


def _embed_views(enrichment: dict, *, row: dict, embed: EmbedFn) -> AnnotationVectors:
    """Compute embeddings for each named-vector slot.

    Multi-vector slots become lists-of-embeddings; single-vector slots
    become flat lists.  Empty multi-vector slots are tolerated (the
    Qdrant write accepts an empty list under a named-vector slot).
    """
    label = row.get("label", "") or ""
    mnemonic = row.get("mnemonic", "") or ""
    description = row.get("description", "") or ""
    parent_path = enrichment.get("parent_path") or []

    label_view_text = f"{label} — {mnemonic} — {description}".strip(" —")
    description_view_text = description or label
    parent_path_text = " > ".join(parent_path) if parent_path else label

    label_vec = embed(label_view_text)
    description_vec = embed(description_view_text)
    parent_path_vec = embed(parent_path_text)

    def embed_many(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out = embed(texts)
        # Defensive: handle both "embed returns list-of-vectors" and
        # "embed returns a single vector when given a single string"
        # under a degenerate single-element list.
        if texts and isinstance(out, list) and out and not isinstance(out[0], list):
            return [out]  # single vector wrapped
        return out  # type: ignore[return-value]

    prototype_texts = [str(v) for v in (enrichment.get("prototype_values") or []) if v is not None]
    pattern_texts = [
        (p.get("expr") if isinstance(p, dict) else str(p))
        for p in (enrichment.get("value_patterns") or [])
        if p is not None
    ]
    name_hint_texts = [str(h) for h in (enrichment.get("name_hints") or []) if h]
    anti_texts = [
        (a.get("value") if isinstance(a, dict) else str(a))
        for a in (enrichment.get("anti_examples") or [])
        if a is not None
    ]

    return AnnotationVectors(
        label_view=label_vec,  # type: ignore[arg-type]
        description_view=description_vec,  # type: ignore[arg-type]
        parent_path_view=parent_path_vec,  # type: ignore[arg-type]
        prototype_values=embed_many(prototype_texts),
        value_patterns=embed_many(pattern_texts),
        name_hints=embed_many(name_hint_texts),
        anti_examples=embed_many(anti_texts),
    )
