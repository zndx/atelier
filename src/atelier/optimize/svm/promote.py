"""Promotion utility: serialize an NHSVM head adapter + register it.

Connects the Phase D training output to the runtime registry.  The
typical flow:

  1. Phase D trains a ``FactorizedNHSVMHead`` (e.g., the best-performing
     fold from k-fold reference-primary)
  2. Caller wraps it in an ``NHSVMHeadAdapter`` with encoder identity
     + training metadata
  3. ``promote_head(adapter, ...)`` writes artifacts to
     ``build/cache/nhsvm/{head_sig}/`` and registers a row in
     ``nhsvm_head_registry`` with ``status='building'``
  4. Operator (or gate-passing automation) calls
     ``atelier.registry.nhsvm_head.promote_to_current(head_id)``
     to atomically demote the prior 'current' and promote the new one

Promotion is intentionally TWO-STEP — registration + explicit current-
flip — so the promotion-quality gate (variance threshold, Gate A
header) can intervene between them.

Per ``feedback_synth_corpus_is_load_bearing``: this function writes
NEW artifacts under ``build/cache/nhsvm/`` keyed by content-addressable
head_sig; it never modifies or deletes existing artifacts.  Cache
directories with old head_sigs remain in place until the operator
explicitly cleans them.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from atelier.classify.factorized_nhsvm import NHSVMHeadAdapter
from atelier.db.dao import AtelierDao
from atelier.registry.nhsvm_head import (
    compute_head_sig,
    register_building,
)

log = logging.getLogger("atelier.optimize.svm.promote")

DEFAULT_CACHE_ROOT = Path("build/cache/nhsvm")


def promote_head(
    adapter: NHSVMHeadAdapter,
    *,
    taxonomy_id: str,
    vocab_sig: str,
    training_mode: str,
    reference_hash: str | None = None,
    corpus_hash: str | None = None,
    fold_seed: int | None = None,
    augment_floor: int | None = None,
    metrics: dict[str, Any] | None = None,
    summary: str | None = None,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    dao: AtelierDao | None = None,
) -> dict:
    """Save the adapter and register a `building` row in nhsvm_head_registry.

    The row is registered with ``status='building'``; promotion to
    ``'current'`` is a separate explicit call (gate may intervene).

    Returns the registry row dict (including the assigned ``id`` and
    ``head_sig``).

    Idempotency: re-promoting an adapter with identical content
    (same head_sig) overwrites the on-disk artifacts byte-equivalently
    but does NOT re-insert the registry row — the head_sig is UNIQUE,
    so the second registration would fail.  Callers that expect to
    rebuild an existing head should look it up first via
    ``atelier.registry.nhsvm_head.get_by_sig``.
    """
    # Content-addressable signature — same key used in registry lookup
    head_sig = compute_head_sig(
        vocab_sig=vocab_sig,
        reference_hash=reference_hash,
        corpus_hash=corpus_hash,
        encoder=adapter.encoder_id,
        embedding_dim=adapter.embed_dim,
        training_mode=training_mode,
        fold_seed=fold_seed,
        augment_floor=augment_floor,
    )

    # Artifact location
    artifact_path = cache_root / head_sig
    log.info(
        "promote_head: saving adapter (taxonomy_id=%s, head_sig=%s, mode=%s) "
        "to %s",
        taxonomy_id, head_sig, training_mode, artifact_path,
    )
    adapter.save(artifact_path)

    # Registry row
    head_id = str(uuid.uuid4())
    row = register_building(
        dao=dao,
        head_id=head_id,
        taxonomy_id=taxonomy_id,
        head_sig=head_sig,
        vocab_sig=vocab_sig,
        encoder=adapter.encoder_id,
        embedding_dim=adapter.embed_dim,
        training_mode=training_mode,
        artifact_path=str(artifact_path),
        reference_hash=reference_hash,
        corpus_hash=corpus_hash,
        fold_seed=fold_seed,
        augment_floor=augment_floor,
        metrics=metrics,
        summary=summary,
    )
    log.info(
        "promote_head: registered building row id=%s head_sig=%s "
        "(promote to 'current' separately via "
        "atelier.registry.nhsvm_head.promote_to_current)",
        head_id, head_sig,
    )
    return row
