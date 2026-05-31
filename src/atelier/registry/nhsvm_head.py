"""DAO for the ``nhsvm_head_registry`` table.

Administrative pointers to trained factorized NHSVM head artifacts.
Mirrors the ``taxonomy_registry`` pattern in ``atelier.db.dao``:

  - ``register_building(...)`` — insert a new row in ``status='building'``
  - ``get_by_id(head_id)`` / ``get_current(taxonomy_id, encoder)`` — read
  - ``list_all(...)`` — list, newest first
  - ``promote_to_current(head_id)`` — atomic demote-prior + promote-new
  - ``set_status(head_id, status)`` — non-current transitions

Status invariants:
  - At most one ``status='current'`` row per (taxonomy_id, encoder) —
    enforced by partial unique index ``idx_nhsvm_head_registry_one_current``
  - Promotion is transactional: prior current → stale, new row → current

See ``db/migrations/20260527000000_nhsvm_head_registry.sql`` for the
schema this DAO maps to, and ``.claude/plans/lovely-doodling-badger.md``
Step 4 for the design rationale.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from atelier.db.dao import AtelierDao


# ──────────────────────────────────────────────────────────────────────
# head_sig: content-addressable identifier
# ──────────────────────────────────────────────────────────────────────


def compute_head_sig(
    *,
    vocab_sig: str,
    reference_hash: str | None,
    corpus_hash: str | None,
    encoder: str,
    embedding_dim: int,
    training_mode: str,
    fold_seed: int | None,
    augment_floor: int | None,
) -> str:
    """Compute the canonical content-addressable head_sig.

    Two heads with identical (vocab_sig, reference_hash, corpus_hash,
    encoder, embedding_dim, training_mode, fold_seed, augment_floor)
    should produce byte-identical models (modulo nondeterminism in the
    training loop, which is bounded under fixed seed).

    The 16-character hex prefix is enough for ~uniqueness across the
    realistic set of registered heads while remaining short enough to
    appear in artifact paths and logs.
    """
    parts = [
        vocab_sig,
        reference_hash or "none",
        corpus_hash or "none",
        encoder,
        str(embedding_dim),
        training_mode,
        str(fold_seed) if fold_seed is not None else "none",
        str(augment_floor) if augment_floor is not None else "none",
    ]
    blob = "|".join(parts)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────
# DAO operations — wrap AtelierDao's session machinery
# ──────────────────────────────────────────────────────────────────────


def register_building(
    *,
    dao: AtelierDao | None = None,
    head_id: str,
    taxonomy_id: str,
    head_sig: str,
    vocab_sig: str,
    encoder: str,
    embedding_dim: int,
    training_mode: str,
    artifact_path: str,
    reference_hash: str | None = None,
    corpus_hash: str | None = None,
    fold_seed: int | None = None,
    augment_floor: int | None = None,
    metrics: dict[str, Any] | None = None,
    summary: str | None = None,
) -> dict:
    """Insert a new head registry row in ``status='building'``.

    Returns the created row as a dict.  Callers promote to ``'current'``
    via :func:`promote_to_current` after gate checks pass.
    """
    dao = dao or AtelierDao()
    from atelier.db.model import NhsvmHeadRegistry
    with dao.get_session() as session:
        row = NhsvmHeadRegistry(
            id=head_id,
            taxonomy_id=taxonomy_id,
            head_sig=head_sig,
            vocab_sig=vocab_sig,
            reference_hash=reference_hash,
            corpus_hash=corpus_hash,
            training_mode=training_mode,
            fold_seed=fold_seed,
            encoder=encoder,
            embedding_dim=embedding_dim,
            augment_floor=augment_floor,
            artifact_path=artifact_path,
            status="building",
            metrics=metrics,
            summary=summary,
        )
        session.add(row)
        session.flush()
        return _row_to_dict(row)


def get_by_id(head_id: str, *, dao: AtelierDao | None = None) -> dict | None:
    """Return a head row by primary key, or None."""
    dao = dao or AtelierDao()
    from atelier.db.model import NhsvmHeadRegistry
    with dao.get_session() as session:
        r = session.query(NhsvmHeadRegistry).filter_by(id=head_id).first()
        return _row_to_dict(r) if r is not None else None


def get_by_sig(head_sig: str, *, dao: AtelierDao | None = None) -> dict | None:
    """Return a head row by content-addressable head_sig, or None."""
    dao = dao or AtelierDao()
    from atelier.db.model import NhsvmHeadRegistry
    with dao.get_session() as session:
        r = session.query(NhsvmHeadRegistry).filter_by(head_sig=head_sig).first()
        return _row_to_dict(r) if r is not None else None


def get_current(
    taxonomy_id: str,
    encoder: str,
    *,
    dao: AtelierDao | None = None,
) -> dict | None:
    """Return the ``status='current'`` row for (taxonomy_id, encoder), or None.

    Backed by the partial unique index — at most one match by
    construction.
    """
    dao = dao or AtelierDao()
    from atelier.db.model import NhsvmHeadRegistry
    with dao.get_session() as session:
        r = (
            session.query(NhsvmHeadRegistry)
            .filter_by(taxonomy_id=taxonomy_id, encoder=encoder, status="current")
            .first()
        )
        return _row_to_dict(r) if r is not None else None


def list_all(
    taxonomy_id: str | None = None,
    encoder: str | None = None,
    include_archived: bool = False,
    *,
    dao: AtelierDao | None = None,
) -> list[dict]:
    """List head registry rows, newest build first.

    Filters by ``taxonomy_id`` and/or ``encoder`` when provided;
    excludes ``archived`` unless explicitly requested.
    """
    dao = dao or AtelierDao()
    from atelier.db.model import NhsvmHeadRegistry
    with dao.get_session() as session:
        q = session.query(NhsvmHeadRegistry)
        if taxonomy_id is not None:
            q = q.filter_by(taxonomy_id=taxonomy_id)
        if encoder is not None:
            q = q.filter_by(encoder=encoder)
        if not include_archived:
            q = q.filter(NhsvmHeadRegistry.status != "archived")
        rows = q.order_by(NhsvmHeadRegistry.built_at.desc()).all()
        return [_row_to_dict(r) for r in rows]


def promote_to_current(head_id: str, *, dao: AtelierDao | None = None) -> bool:
    """Promote one head to ``status='current'`` for its (taxonomy_id, encoder).

    Demote-then-promote runs in a single transaction to honor the
    partial unique index ``idx_nhsvm_head_registry_one_current``.  Any
    prior ``current`` row for the same (taxonomy_id, encoder)
    transitions to ``stale``.

    Returns True when the target row was found and promoted.
    """
    dao = dao or AtelierDao()
    from atelier.db.model import NhsvmHeadRegistry
    with dao.get_session() as session:
        target = (
            session.query(NhsvmHeadRegistry).filter_by(id=head_id).first()
        )
        if target is None:
            return False
        # Demote any current row(s) for this (taxonomy_id, encoder) to 'stale'.
        (
            session.query(NhsvmHeadRegistry)
            .filter_by(
                taxonomy_id=target.taxonomy_id,
                encoder=target.encoder,
                status="current",
            )
            .update({"status": "stale"})
        )
        session.flush()
        target.status = "current"
        return True


def set_status(
    head_id: str, status: str, *, dao: AtelierDao | None = None,
) -> bool:
    """Set a head's status to an arbitrary non-current value.

    Use :func:`promote_to_current` to promote to ``current`` (it
    handles the demotion invariant).  This is for ``stale`` / ``archived``
    transitions where the partial-unique index isn't at risk.
    """
    if status == "current":
        raise ValueError(
            "use promote_to_current() to set status='current' "
            "(it honors the one-current-per-(taxonomy_id, encoder) invariant)"
        )
    dao = dao or AtelierDao()
    from atelier.db.model import NhsvmHeadRegistry
    with dao.get_session() as session:
        count = (
            session.query(NhsvmHeadRegistry)
            .filter_by(id=head_id)
            .update({"status": status})
        )
        return count > 0


def update_metrics(
    head_id: str,
    metrics: dict[str, Any],
    *,
    dao: AtelierDao | None = None,
) -> bool:
    """Replace the metrics blob for a head row.

    Promotion gates (e.g., fold variance threshold) read this column;
    callers update it after Gate A measurement before calling
    :func:`promote_to_current`.
    """
    dao = dao or AtelierDao()
    from atelier.db.model import NhsvmHeadRegistry
    with dao.get_session() as session:
        count = (
            session.query(NhsvmHeadRegistry)
            .filter_by(id=head_id)
            .update({"metrics": metrics})
        )
        return count > 0


# ──────────────────────────────────────────────────────────────────────
# Row → dict helper
# ──────────────────────────────────────────────────────────────────────


def _row_to_dict(r) -> dict:
    # SQLAlchemy JSON column returns the deserialized dict directly;
    # tolerate legacy string-encoded values for forward-compat.
    metrics = r.metrics
    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except (ValueError, TypeError):
            metrics = {"_raw": metrics}
    return {
        "id": r.id,
        "taxonomy_id": r.taxonomy_id,
        "head_sig": r.head_sig,
        "vocab_sig": r.vocab_sig,
        "reference_hash": r.reference_hash,
        "corpus_hash": r.corpus_hash,
        "training_mode": r.training_mode,
        "fold_seed": r.fold_seed,
        "encoder": r.encoder,
        "embedding_dim": r.embedding_dim,
        "augment_floor": r.augment_floor,
        "artifact_path": r.artifact_path,
        "status": r.status,
        "metrics": metrics,
        "summary": r.summary,
        "built_at": str(r.built_at or ""),
        "updated_at": str(r.updated_at or ""),
    }
