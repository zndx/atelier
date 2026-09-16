"""Resident classify helpers used by ClassificationFlow steps.

Steps re-load config/vocabulary from source_id so artifacts stay pickleable
for @kubernetes. Probe is cheap; ensure_preconditioned runs only when stale.
"""

from __future__ import annotations

from typing import Any

from atelier.flows.classify_phases import probe_requires_precondition


def load_category_set(cfg: Any, source_id: str):
    """Vocabulary for this source. Sample ids use the sample annotations.csv."""
    from atelier.sdg.sample import is_sample_source_id, sample_dir_from_source_id

    if is_sample_source_id(source_id):
        sample_dir = sample_dir_from_source_id(source_id)
        if sample_dir is None:
            raise RuntimeError(
                f"sdg-corpora sample {source_id!r} missing under build/sdg_sample/"
            )
        from atelier.classify.taxonomy import load_annotations_from_filesystem

        return load_annotations_from_filesystem(
            sample_dir / "annotations.csv", hierarchical=True,
        )
    if source_id in ("ootb-sample", ""):
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


def load_classification_rows(result: dict) -> list[dict]:
    """Rows from pipeline result_path/classifications.json (pickle-friendly)."""
    import json
    from pathlib import Path

    raw = result.get("result_path") or ""
    if not raw:
        return []
    path = Path(raw)
    if path.is_dir():
        path = path / "classifications.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def run_dst_pipeline(cfg: Any, source_id: str, *, skip_precondition: bool) -> dict:
    """DST classify via Engine/Complete. Precondition already handled by the flow."""
    import dataclasses

    from atelier.classify import get_fsm
    from atelier.classify.complete_backend import CompleteLLMBackend
    from atelier.classify.pipeline import run_classification_pipeline
    from atelier.db.dao import AtelierDao

    cfg = dataclasses.replace(
        cfg,
        classify_precondition_enabled=not skip_precondition,
        overwatch_nautilus_enabled=False,
    )
    fsm = get_fsm(AtelierDao())
    return run_classification_pipeline(
        cfg,
        fsm,
        source_id=source_id or None,
        llm_backend=CompleteLLMBackend.from_cfg(cfg),
    )
