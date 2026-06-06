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

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
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
    compose_annotation_text,
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
    # Per-row JSONL checkpoint path.  When None, auto-derives to
    # ``build/enrichment/{collection}-{started_at}.jsonl`` so a long-
    # running enrichment becomes inspectable via the shared filesystem
    # (one JSON event per row, append-only, tail-able mid-run).  Pass
    # an empty string to disable checkpointing explicitly.  Default is
    # the auto path — never silently off, consistent with the no-
    # silent-degradation discipline: an unobservable long-running
    # production process IS a degraded operating mode.
    checkpoint_path: str | None = None


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

    # Resolve the checkpoint path.  None → auto-derive a NFS-visible
    # path under build/enrichment/; "" → explicit disable; anything
    # else is used verbatim.  The directory is created eagerly so the
    # first event lands cleanly.
    checkpoint_file: Path | None = _resolve_checkpoint_path(
        config.checkpoint_path, collection=collection, started_at=started_at,
    )
    if checkpoint_file is not None:
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        _emit_checkpoint(checkpoint_file, {
            "event": "loop_start",
            "ts": started_at,
            "collection": collection,
            "taxonomy_id": config.taxonomy_id,
            "augmentation_version": config.augmentation_version,
            "embedding_model": config.embedding_model,
            "embedding_dim": config.embedding_dim,
            "taxonomy_version_hash": version_hash,
            "source_row_count": len(source_rows),
            "max_attempts_per_row": config.max_attempts_per_row,
            "skip_on_cache_hit": config.skip_on_cache_hit,
            "dry_run": config.dry_run,
        })
        logger.info("Enrichment checkpoint: %s", checkpoint_file)

    result = LoopResult(
        collection_name=collection,
        taxonomy_version_hash=version_hash,
        started_at=started_at,
    )

    # Pre-index the taxonomy tree for efficient per-row context building.
    by_code = {r.get("code"): r for r in source_rows if r.get("code")}
    children_of: dict[str | None, list[dict]] = {}
    for r in source_rows:
        pc = r.get("parent_code")
        children_of.setdefault(pc, []).append(r)

    # A code is a leaf iff it has no children in the source taxonomy.
    codes_with_children = {
        pc for pc in children_of if pc is not None and pc in by_code
    }

    for row in source_rows:
        code = row.get("code") or row.get("mnemonic") or "<unknown>"

        expected_parent = (
            expected_parent_path_for(row) if expected_parent_path_for else None
        )

        # Build rich per-row taxonomy context for the prompt.
        parent_code = row.get("parent_code")
        siblings = [
            {"code": s.get("code"), "label": s.get("label")}
            for s in children_of.get(parent_code, [])
            if s.get("code") != row.get("code")
        ]
        children = [
            {"code": c.get("code"), "label": c.get("label")}
            for c in children_of.get(row.get("code"), [])
        ]
        is_leaf = row.get("code") not in codes_with_children

        # Cross-level confusables: uncle/aunt tags (siblings of ancestors)
        # that may carry visually similar values.
        cross_level: list[dict] = []
        cur_parent = parent_code
        for _ in range(8):  # Walk up max 8 levels
            if not cur_parent or cur_parent not in by_code:
                break
            grandparent_code = by_code[cur_parent].get("parent_code")
            for s in children_of.get(grandparent_code, []):
                if s.get("code") != cur_parent:
                    cross_level.append(
                        {"code": s.get("code"), "label": s.get("label")}
                    )
            cur_parent = grandparent_code
        # Limit to 20 most relevant
        cross_level = cross_level[:20]

        row_context = {
            "valid_tag_codes": list(valid_tag_codes) if valid_tag_codes else [],
            "siblings": siblings,
            "children": children,
            "cross_level_confusables": cross_level,
            "is_leaf": is_leaf,
            "row_count": len(source_rows),
            "expected_parent_path": expected_parent,
        }

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
                outcome = RowOutcome(
                    code=code, status="cache_hit", attempts=0, point_id=pid
                )
                result.rows.append(outcome)
                if checkpoint_file is not None:
                    _emit_checkpoint(checkpoint_file, _outcome_event(outcome))
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
        if checkpoint_file is not None:
            _emit_checkpoint(checkpoint_file, _outcome_event(outcome))

    result.finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if checkpoint_file is not None:
        _emit_checkpoint(checkpoint_file, {
            "event": "loop_end",
            "ts": result.finished_at,
            "collection": collection,
            "counts": result.counts,
            "rows_total": len(result.rows),
        })
    logger.info(
        "Enrichment loop done: collection=%s outcomes=%s",
        collection,
        result.counts,
    )
    return result


# ── Checkpoint helpers ────────────────────────────────────────────


def _resolve_checkpoint_path(
    configured: str | None,
    *,
    collection: str,
    started_at: str,
) -> Path | None:
    """Resolve the configured checkpoint setting to an actual Path.

    Semantics:
      - ``None``  → auto-derive ``build/enrichment/{coll}-{ts}.jsonl``
      - ``""``    → explicit disable (returns None)
      - else      → use verbatim
    """
    if configured == "":
        return None
    if configured is None:
        # NFS-visible default.  Timestamp tagged so concurrent or
        # restarted runs land in distinct files; collection name
        # makes the file's target identifiable at a glance.
        safe_ts = started_at.replace(":", "")
        return Path("build/enrichment") / f"{collection}-{safe_ts}.jsonl"
    return Path(configured)


def _emit_checkpoint(path: Path, event: dict) -> None:
    """Append one JSON event to the checkpoint file.

    Append-only + line-buffered so ``tail -f`` reads each event as it
    lands.  Failures here are logged at WARNING and swallowed — a
    flaky NFS write must never crash the enrichment loop.
    """
    try:
        line = json.dumps(event, default=str)
        with path.open("a", buffering=1) as fh:
            fh.write(line + "\n")
    except Exception as exc:  # noqa: BLE001 — observability is best-effort
        logger.warning("Checkpoint write failed (continuing): %s", exc)


def _outcome_event(outcome: RowOutcome) -> dict:
    """Project a RowOutcome to a JSON-safe checkpoint event."""
    return {
        "event": "row",
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code": outcome.code,
        "status": outcome.status,
        "attempts": outcome.attempts,
        "verifier_pass": (
            f"{outcome.verifier_checks_passed}/{outcome.verifier_checks_total}"
            if outcome.verifier_checks_total else None
        ),
        "point_id": outcome.point_id,
        "failure_reason": outcome.failure_reason or None,
    }


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
    vectors = _embed_annotation(last_attempt.enrichment, row=row, embed=embed)
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


def _embed_annotation(enrichment: dict, *, row: dict, embed: EmbedFn) -> AnnotationVectors:
    """Encode the annotation's composed text through ColBERT.

    Composes a single text passage from the annotation row + enrichment
    features via ``compose_annotation_text``, then encodes it to produce
    per-token ColBERT vectors for Qdrant's native MaxSim.

    The ``embed`` callable should be ``colbert_encoder.get_encoder().encode_single``
    (returns an ndarray of shape ``(num_tokens, dim)``).
    """
    payload = {**row, **enrichment}
    text = compose_annotation_text(payload)
    if not text:
        text = row.get("label") or row.get("code") or "unknown"

    token_vectors = embed(text)

    if hasattr(token_vectors, "tolist"):
        token_vectors = token_vectors.tolist()

    return AnnotationVectors(colbert=token_vectors)
