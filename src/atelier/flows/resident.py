"""Resident classify helpers used by ClassificationFlow steps.

Steps re-load config/vocabulary from source_id so artifacts stay pickleable
for @kubernetes. Probe is cheap; ensure_preconditioned runs only when stale.
"""

from __future__ import annotations

from typing import Any

from atelier.flows.classify_phases import probe_requires_precondition


def load_category_set(cfg: Any, source_id: str):
    """Vocabulary for this source. SDG sample uses filesystem annotations."""
    if source_id in ("sdg-corpora", "ootb-sample", ""):
        from atelier.classify.sampler import load_sample_vocabulary

        return load_sample_vocabulary(hierarchical=True)
    # Filesystem / DAO sources: taxonomy from cfg vocab URI when present.
    vocab_uri = getattr(cfg, "classify_vocab_uri", None) or None
    if vocab_uri:
        from atelier.classify.taxonomy import load_annotations_from_filesystem

        return load_annotations_from_filesystem(vocab_uri, hierarchical=True)
    from atelier.classify.sampler import load_sample_vocabulary

    return load_sample_vocabulary(hierarchical=True)


def probe_status(cfg: Any, source_id: str, category_set=None):
    from atelier.optimize.precondition import probe

    cats = category_set if category_set is not None else load_category_set(cfg, source_id)
    tax = source_id or getattr(cfg, "classify_taxonomy_id", None) or "default"
    return probe(cfg, cats, taxonomy_id=tax)


def run_precondition_if_needed(cfg: Any, source_id: str, category_set=None, *, heartbeat=None) -> bool:
    """Return True if expensive precondition ran."""
    from atelier.optimize.precondition import ensure_preconditioned

    cats = category_set if category_set is not None else load_category_set(cfg, source_id)
    status = probe_status(cfg, source_id, cats)
    if not probe_requires_precondition(status):
        return False
    tax = source_id or getattr(cfg, "classify_taxonomy_id", None) or "default"
    ensure_preconditioned(cfg, cats, taxonomy_id=tax, heartbeat=heartbeat)
    return True
