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


def run_precondition_if_needed(
    cfg: Any, source_id: str, category_set=None, *, heartbeat=None,
    stages: tuple[str, ...] | None = None,
) -> bool:
    """Return True if expensive precondition ran."""
    from atelier.optimize.precondition import ensure_preconditioned

    cats = category_set if category_set is not None else load_category_set(cfg, source_id)
    status = probe_status(cfg, source_id, cats)
    if not probe_requires_precondition(status) and not stages:
        return False
    tax = source_id or getattr(cfg, "classify_taxonomy_id", None) or "default"
    ensure_preconditioned(
        cfg, cats, taxonomy_id=tax, heartbeat=heartbeat, stages=stages,
    )
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


def load_sweep_samples(cfg: Any, source_id: str):
    """Entities the decisions shards walk. Blind: no reference key."""
    from atelier.sdg.sample import is_sample_source_id, sample_dir_from_source_id

    sample_size = int(getattr(cfg, "classify_sample_size", 50) or 50)
    if is_sample_source_id(source_id or ""):
        sample_dir = sample_dir_from_source_id(source_id)
        if sample_dir is None:
            raise RuntimeError(
                f"sdg-corpora sample {source_id!r} is not on disk under "
                "build/sdg_sample/. Run `just sdg-sample`."
            )
        from atelier.classify.filesystem_source import load_filesystem_source

        return load_filesystem_source(
            str(sample_dir / "tables"),
            sample_size=sample_size,
            database=source_id,
        )
    if source_id == "ootb-sample":
        from atelier.classify.sampler import load_sample_source

        return load_sample_source()
    release = str(getattr(cfg, "aegir_release_dir", "") or "").strip()
    if release:
        from atelier.classify.aegir_release import load_aegir_release_samples

        return load_aegir_release_samples(
            release, sample_size=sample_size, with_reference=False,
        )
    raise RuntimeError(
        "decisions shard has no classification sample and no Aegir release. "
        f"source_id={source_id!r}."
    )


def run_decisions_shard(
    cfg: Any,
    source_id: str,
    shard: int,
    *,
    width: int | None = None,
    taxonomy_id: str | None = None,
) -> list[dict]:
    """Walk one replica's columns. ``parallel`` stays 1 inside the pod."""
    from atelier.classify.decisions_backend import (
        REPLICA_COUNT,
        DecisionsBackend,
        select_shard_columns,
    )
    from atelier.flows.lattice import GURU_NOCAP

    width = REPLICA_COUNT if width is None else int(width)
    samples = load_sweep_samples(cfg, source_id)
    chosen = select_shard_columns(samples, int(shard), width)
    tax = (taxonomy_id or source_id or "").strip()
    cats = load_category_set(cfg, tax)
    backend = DecisionsBackend.from_cfg(cfg)
    if not backend.health_check():
        raise RuntimeError(
            f"{GURU_NOCAP} decisions is not healthy. "
            "Begin a Gaius workload that requires the decisions capability. "
            "The sweep does not fall back to instruct, and Decide does not start the replicas."
        )
    backend.bind_taxonomy(cats)
    groups: list[tuple[Any, list]] = []
    current = None
    bucket: list = []
    for table, column in chosen:
        if table is not current:
            bucket = []
            groups.append((table, bucket))
            current = table
        bucket.append(column)
    labels: list[dict] = []
    for table, columns in groups:
        if not columns:
            continue
        response = backend.classify_batch(
            columns,
            "Choose the genus for this entity.",
            table_name=getattr(table, "name", None) or None,
        )
        table_name = getattr(table, "name", "") or ""
        for item in response.classifications:
            bare = item.column_name
            qualified = f"{table_name}.{bare}" if table_name else bare
            labels.append({
                "qualified_name": qualified,
                "column_name": bare,
                "predicted_code": item.category_code or "",
                "confidence": float(item.confidence or 0.0),
                "alternatives": list(item.alternatives or []),
            })
    return labels


def run_dst_pipeline(
    cfg: Any, source_id: str, *, skip_precondition: bool,
    taxonomy_id: str | None = None,
    include_revisit: bool = True,
    llm_backend=None,
) -> dict:
    """DST classify the holdout ``source_id``.

    ``taxonomy_id`` is the artifact key (NHSVM current + Qdrant current).
    When a reference sample trained the head, pass that id so classify
    does not look up ``default`` or the holdout id.

    The sweep asks capability ``decisions``. ``include_revisit`` false
    stops before the thinking revisit so that pass can run as its own step.
    ``llm_backend`` replays shard labels so fusion does not call Decide again.
    """
    import dataclasses

    from atelier.classify import get_fsm
    from atelier.classify.pipeline import run_classification_pipeline
    from atelier.db.dao import AtelierDao

    tax = (taxonomy_id or source_id or "").strip() or None
    cats = None
    if tax and source_id and tax != source_id:
        cats = load_category_set(cfg, tax)
    cfg = dataclasses.replace(
        cfg,
        classify_precondition_enabled=not skip_precondition,
        overwatch_nautilus_enabled=False,
        classify_taxonomy_id=tax or getattr(cfg, "classify_taxonomy_id", None),
    )
    fsm = get_fsm(AtelierDao())
    return run_classification_pipeline(
        cfg,
        fsm,
        source_id=source_id or None,
        category_set=cats,
        taxonomy_id=tax,
        include_revisit=include_revisit,
        llm_backend=llm_backend,
    )


def run_dst_revisit(
    cfg: Any,
    result_path: str = "",
    *,
    rows: list[dict] | None = None,
) -> dict:
    """Thinking revisit over decisions labels, from disk or from memory."""
    from atelier.classify.complete_backend import CompleteLLMBackend
    from atelier.classify.sampler import ColumnSample

    if rows is None:
        rows = load_classification_rows({"result_path": result_path})
    grouped: dict[str, list[tuple[str, str, str]]] = {}
    for row in rows:
        qualified = str(row.get("qualified_name") or row.get("column_name") or "")
        code = str(row.get("predicted_code") or "")
        if not qualified:
            continue
        if "." in qualified:
            table, bare = qualified.split(".", 1)
        else:
            table, bare = "", qualified
        grouped.setdefault(table, []).append((bare, code, qualified))
    if not grouped:
        where = result_path or "memory"
        raise RuntimeError(
            "decisions sweep wrote no classifications to revisit "
            f"at {where!r}"
        )
    backend = CompleteLLMBackend.from_cfg(cfg)
    revisited: list[dict] = []
    for table, items in grouped.items():
        samples = [
            ColumnSample(name=bare, values=[], table_name=table)
            for bare, _code, _qualified in items
        ]
        context = {
            bare: {"llm_code": code, "source": "decisions"}
            for bare, code, _qualified in items
        }
        response = backend.classify_batch(
            samples,
            "Revisit the decisions sweep labels.",
            revisit_context=context,
            table_name=table or None,
        )
        by_bare = {bare: qualified for bare, _code, qualified in items}
        for item in response.classifications:
            revisited.append({
                "column_name": item.column_name,
                "qualified_name": by_bare.get(item.column_name, item.column_name),
                "predicted_code": item.category_code or "",
                "confidence": item.confidence,
            })
    return {
        "state": "OK",
        "capability": "thinking",
        "classifications": revisited,
    }
