"""In-situ pre-conditioning — finalize run artifacts before classification.

A cold environment fails classification twice by design: the SVM
selector (``svm.source = "registered"``) raises when no NHSVM head is
current for ``(taxonomy_id, encoder)``, and the maxsim channel raises
``MaxSimUnavailable`` when the taxonomy registry has no current
collection.  This module converts those two typed failures into an
automated repair pass, executed as the ``PRECONDITIONING`` FSM phase:

1. **Probe** — is each artifact *final*?  Finality is presence AND
   signature match: the collection's ``augmentation_version`` encodes
   the vocabulary signature, and the head registry row stores
   ``vocab_sig`` directly.  A vocabulary change (or an sdg-corpora pin
   bump that changes the sample vocabulary) automatically invalidates.
2. **Enrich → collection** — LLM-enrich the run vocabulary via the
   classify backend (``ClassifyBackedEnrichmentGenerator``), ColBERT-
   encode, upsert to Qdrant, register + promote the collection.
3. **Corpus → head** — synthesize a training corpus from the
   enrichment payloads (real ``prototype_values`` first), adapt to
   training rows, ModernBERT-encode, fit the factorized NHSVM, and
   promote the head to ``current``.

Every stage is idempotent by artifact inspection and **fails loudly**:
any enrichment row failure, any training-label mismatch, any registry
inconsistency raises ``PreconditionError`` with the failing detail —
the run lands in FSM ``ERROR`` rather than degrading silently.

The test fixture lane (``optimize/svm/fixture.py``) is deliberately
NOT used here — fixtures never surface in the application.  This
module generalizes the same train→promote pattern from live enrichment
data instead.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

ENCODER_ID = "answerdotai/ModernBERT-base"
EMBED_DIM = 768
COLBERT_MODEL = "colbert-ir/colbertv2.0"
TRAINING_MODE = "precondition"


class PreconditionError(RuntimeError):
    """A pre-conditioning stage failed; the run must not proceed."""


# ── Probe ────────────────────────────────────────────────────────

@dataclass
class PreconditionStatus:
    taxonomy_id: str
    vocab_sig: str
    augmentation_version: str
    collection_final: bool
    head_final: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def final(self) -> bool:
        return self.collection_final and self.head_final

    def to_dict(self) -> dict[str, Any]:
        return {
            "taxonomy_id": self.taxonomy_id,
            "vocab_sig": self.vocab_sig,
            "augmentation_version": self.augmentation_version,
            "collection_final": self.collection_final,
            "head_final": self.head_final,
            "final": self.final,
            "reasons": self.reasons,
        }


def _vocab_sig_for(category_set) -> str:
    from atelier.classify.artifact_set import compute_vocab_signature
    codes = sorted(c.code for c in category_set.all_categories)
    return compute_vocab_signature(codes)


def _augmentation_version(vocab_sig: str) -> str:
    # The version string carries the vocabulary signature so
    # signature-matching is structural: a changed vocabulary yields a
    # new (taxonomy_id, augmentation_version) registry row rather than
    # silently reusing a stale collection.
    return f"vs{vocab_sig[:10]}"


def probe(cfg, category_set, *, taxonomy_id: str) -> PreconditionStatus:
    """Assess artifact finality for this run's taxonomy.  Read-only."""
    from atelier.db.dao import AtelierDao
    from atelier.registry.nhsvm_head import get_current

    vocab_sig = _vocab_sig_for(category_set)
    aug_version = _augmentation_version(vocab_sig)
    status = PreconditionStatus(
        taxonomy_id=taxonomy_id, vocab_sig=vocab_sig,
        augmentation_version=aug_version,
        collection_final=False, head_final=False,
    )

    dao = AtelierDao()
    current = dao.get_current_taxonomy_collection(taxonomy_id)
    if current is None:
        status.reasons.append("no current semantic collection")
    elif current.get("augmentation_version") != aug_version:
        status.reasons.append(
            f"semantic collection is signature-stale "
            f"({current.get('augmentation_version')!r} != {aug_version!r})"
        )
    else:
        status.collection_final = True

    head = get_current(taxonomy_id, ENCODER_ID)
    if head is None:
        status.reasons.append("no current NHSVM head")
    elif head.get("vocab_sig") != vocab_sig:
        status.reasons.append(
            f"NHSVM head is signature-stale "
            f"({str(head.get('vocab_sig'))[:12]}… != {vocab_sig[:12]}…)"
        )
    elif not Path(head.get("artifact_path", "")).exists():
        status.reasons.append(
            f"NHSVM head artifacts missing on disk "
            f"({head.get('artifact_path')})"
        )
    else:
        status.head_final = True
    return status


# ── Stage: enrich → semantic collection ──────────────────────────

def _category_rows(category_set) -> list[dict]:
    """Adapt a HierarchicalCategorySet to enrichment source rows.

    ``mnemonic`` mirrors ``abbrev`` and roots get ``parent_code=""``
    — both required for hash stability with the enrichment loop's
    ``source_row_hash`` (which reads only ``mnemonic`` and
    JSON-serializes ``parent_code``).
    """
    rows: list[dict] = []
    for c in category_set.all_categories:
        rows.append({
            "code": c.code,
            "label": c.label or c.code,
            "mnemonic": getattr(c, "abbrev", "") or "",
            "description": getattr(c, "description", "") or "",
            "parent_code": getattr(c, "parent_code", None) or "",
        })
    return rows


def _parent_path_closure(category_set) -> Callable[[dict], list[str] | None]:
    by_code = {c.code: c for c in category_set.all_categories}

    def _expected(row: dict) -> list[str] | None:
        labels: list[str] = []
        seen: set[str] = set()
        current = (row.get("parent_code") or "").strip()
        while current and current not in seen and current in by_code:
            seen.add(current)
            cat = by_code[current]
            labels.append(cat.label or cat.code)
            current = getattr(cat, "parent_code", None) or ""
        labels.reverse()
        return labels or None

    return _expected


def build_semantic_collection(
    cfg, category_set, *, taxonomy_id: str, vocab_sig: str,
    heartbeat: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Enrich the vocabulary and promote the resulting collection.

    Returns a summary dict.  Raises ``PreconditionError`` on any row
    failure or registry inconsistency — a partially-enriched
    vocabulary is not a final artifact.
    """
    from qdrant_client import QdrantClient

    from atelier.classify.colbert_encoder import get_encoder, warmup
    from atelier.db.dao import AtelierDao
    from atelier.enrichment.llm_generator import (
        ClassifyBackedEnrichmentGenerator,
    )
    from atelier.enrichment.loop import EnrichmentLoopConfig, run_enrichment
    from atelier.enrichment.qdrant_writer import collection_name_for

    aug_version = _augmentation_version(vocab_sig)
    rows = _category_rows(category_set)

    # Fail-fast constructions: generator raises EnrichmentModelError
    # on backend misconfiguration before any loop work; warmup pays
    # the ColBERT load before the first row rather than mid-loop.
    generator = ClassifyBackedEnrichmentGenerator(cfg)
    warmup()
    encoder = get_encoder()

    qdrant_url = f"http://{cfg.qdrant_host}:{cfg.qdrant_http_port}"
    client = QdrantClient(url=qdrant_url)

    checkpoint_dir = _REPO_ROOT / cfg.artifact_root / "enrichment"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    total = len(rows)
    seen_codes: set[str] = set()
    inner_generate = generator.generate

    def _counted_generate(source_row, **kwargs):
        # Progress = distinct terms reached, not LLM calls — retries
        # on one term must not walk the counter past the total.
        seen_codes.add(source_row.get("code", ""))
        if heartbeat:
            heartbeat({
                "sub_phase": "enrich",
                "current": len(seen_codes), "total": total,
                "unit": "terms", "code": source_row.get("code", ""),
            })
        return inner_generate(source_row, **kwargs)

    generator.generate = _counted_generate  # type: ignore[method-assign]

    loop_cfg = EnrichmentLoopConfig(
        taxonomy_id=taxonomy_id,
        augmentation_version=aug_version,
        embedding_model=COLBERT_MODEL,
        embedding_dim=encoder.dim,
        max_attempts_per_row=int(
            getattr(cfg, "enrichment_max_attempts", 3) or 3),
        checkpoint_path=str(
            checkpoint_dir / f"{taxonomy_id}_{aug_version}.jsonl"),
    )
    result = run_enrichment(
        source_rows=rows,
        generator=generator,
        embed=encoder.encode_single,
        config=loop_cfg,
        qdrant_client=client,
        valid_tag_codes={r["code"] for r in rows},
        expected_parent_path_for=_parent_path_closure(category_set),
    )

    failed = [r for r in result.rows
              if r.status in ("generator_failed", "verifier_failed")]
    if failed:
        detail = "; ".join(
            f"{r.code}[{r.status}: {r.failure_reason or 'verifier'}]"
            for r in failed[:5])
        raise PreconditionError(
            f"Enrichment failed for {len(failed)}/{total} term(s) — "
            f"first: {detail}.  The semantic collection is incomplete; "
            f"refusing to promote it."
        )

    dao = AtelierDao()
    coll_name = collection_name_for(taxonomy_id, aug_version)
    coll_id = f"{taxonomy_id}-{aug_version}"
    try:
        dao.register_taxonomy_collection(
            id=coll_id,
            taxonomy_id=taxonomy_id,
            source_table=f"precondition/{taxonomy_id}",
            qdrant_collection=coll_name,
            augmentation_version=aug_version,
            embedding_model=COLBERT_MODEL,
            embedding_dim=encoder.dim,
            qdrant_url=qdrant_url,
            summary=(
                f"Pre-conditioned enrichment: {total} terms, "
                f"vocab_sig {vocab_sig[:12]}"
            ),
        )
    except Exception:
        # UNIQUE(taxonomy_id, augmentation_version) — a prior partial
        # run registered the row; the loop's point-level cache made
        # this re-run cheap and the promote below converges status.
        existing = [
            c for c in dao.list_taxonomy_collections(taxonomy_id)
            if c.get("augmentation_version") == aug_version
        ]
        if not existing:
            raise
        coll_id = existing[0]["id"]
        logger.info("Reusing registered collection row %s", coll_id)

    if not dao.set_current_taxonomy_collection(coll_id):
        raise PreconditionError(
            f"Failed to promote collection {coll_id!r} to current — "
            f"registry row vanished mid-stage."
        )

    counts = result.counts
    logger.info(
        "Semantic collection final: %s (%s) — %s",
        coll_name, aug_version, counts,
    )
    return {"collection": coll_name, "collection_id": coll_id,
            "counts": counts}


# ── Stage: corpus → NHSVM head ───────────────────────────────────

def _corpus_rows_from_synth(
    corpus_dir: Path, table_metadata: list[dict],
) -> list[Any]:
    """Adapt generated synth CSVs to NHSVM training Rows.

    Iterates per-CSV (headers become ``siblings_full``) rather than
    using the flat loader, so sibling context reflects actual table
    membership.
    """
    import csv as csv_mod

    from atelier.classify.ml_train import _infer_column_type
    from atelier.optimize.svm.reflect import Row

    rows: list[Any] = []
    for meta in table_metadata:
        csv_path = corpus_dir / f"{meta['table_name']}.csv"
        if not csv_path.is_file():
            raise PreconditionError(
                f"Corpus table {csv_path} missing — generation reported "
                f"it but the file is absent."
            )
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            headers = reader.fieldnames or []
            data = list(reader)
        for col, code in (meta.get("reference_labels") or {}).items():
            values = [r.get(col, "") for r in data if r.get(col)]
            rows.append(Row(
                table=meta["table_name"],
                column=col,
                column_type=_infer_column_type(values[:10]) or "object",
                sample_values=values[:8],
                siblings_full=[h for h in headers if h != col],
                mnemonic="",
                code=code,
            ))
    return rows


def build_nhsvm_head(
    cfg, category_set, *, taxonomy_id: str, vocab_sig: str,
    heartbeat: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Train + promote an NHSVM head from enrichment-payload corpus.

    Returns the registry row dict.  Raises ``PreconditionError`` on
    empty corpus, label mismatch, or registry inconsistency.
    """
    import numpy as np

    from atelier.classify.enrichment_loader import load_enrichment_payloads
    from atelier.classify.factorized_nhsvm import (
        NHSVMHeadAdapter,
        fit_factorized_nhsvm,
    )
    from atelier.classify.synth import generate_user_taxonomy_corpus
    from atelier.optimize.svm.encoder import encode_modernbert
    from atelier.optimize.svm.promote import promote_head
    from atelier.optimize.svm.reflect import build_texts_and_labels
    from atelier.registry.nhsvm_head import (
        compute_head_sig,
        get_by_sig,
        promote_to_current,
    )

    if heartbeat:
        heartbeat({"sub_phase": "corpus", "current": 0, "total": 1})

    payloads = load_enrichment_payloads(cfg=cfg, taxonomy_id=taxonomy_id)
    corpus_dir = (_REPO_ROOT / cfg.artifact_root / "svm_training"
                  / "precondition" / f"{taxonomy_id}_{vocab_sig[:10]}")
    corpus_dir.mkdir(parents=True, exist_ok=True)

    table_metadata, coverage = generate_user_taxonomy_corpus(
        category_set, payloads, corpus_dir, seed=42,
    )
    rows = _corpus_rows_from_synth(corpus_dir, table_metadata)
    if not rows:
        missing = [c for c, src in coverage.items() if src == "missing"]
        raise PreconditionError(
            f"Corpus generation produced zero training rows for "
            f"taxonomy {taxonomy_id!r} ({len(missing)} of "
            f"{len(coverage)} codes had no generator).  Enrichment "
            f"payloads lack prototype_values — inspect the collection."
        )

    corpus_file = corpus_dir / "training_rows.jsonl"
    corpus_file.write_text("\n".join(
        json.dumps(r.__dict__, sort_keys=True) for r in rows))
    corpus_hash = hashlib.sha1(corpus_file.read_bytes()).hexdigest()[:12]

    head_sig = compute_head_sig(
        vocab_sig=vocab_sig, reference_hash=None, corpus_hash=corpus_hash,
        encoder=ENCODER_ID, embedding_dim=EMBED_DIM,
        training_mode=TRAINING_MODE, fold_seed=None, augment_floor=None,
    )
    existing = get_by_sig(head_sig)
    if existing is not None and Path(
            existing.get("artifact_path", "")).exists():
        # Byte-identical inputs — the head already exists; converge the
        # pointer instead of retraining (promote_head's UNIQUE head_sig
        # would reject a re-insert anyway).
        if not promote_to_current(existing["id"]):
            raise PreconditionError(
                f"Existing head {existing['id']} could not be promoted "
                f"to current."
            )
        logger.info("Reusing existing NHSVM head %s (sig %s)",
                    existing["id"], head_sig)
        return existing

    if heartbeat:
        heartbeat({"sub_phase": "train_head",
                   "current": 0, "total": len(rows), "unit": "rows"})

    texts, labels = build_texts_and_labels(rows)
    X = np.asarray(encode_modernbert(texts), dtype=np.float32)
    head, train_result = fit_factorized_nhsvm(
        X, labels, category_set, embed_dim=EMBED_DIM, verbose=False,
    )
    adapter = NHSVMHeadAdapter(
        head, encoder_id=ENCODER_ID, embed_dim=EMBED_DIM,
        training_metadata={
            "source": TRAINING_MODE,
            "training_mode": TRAINING_MODE,
            "taxonomy_id": taxonomy_id,
            "n_rows": len(rows),
        },
    )
    row = promote_head(
        adapter,
        taxonomy_id=taxonomy_id,
        vocab_sig=vocab_sig,
        training_mode=TRAINING_MODE,
        corpus_hash=corpus_hash,
        metrics={
            "final_train_acc": train_result.final_train_acc,
            "n_rows": len(rows),
            "n_nodes": len(category_set.all_categories),
        },
        summary=(
            f"Pre-conditioned head: {len(rows)} rows, "
            f"train_acc {train_result.final_train_acc:.3f}"
        ),
        cache_root=_REPO_ROOT / cfg.artifact_root / "cache" / "nhsvm",
    )
    if not promote_to_current(row["id"]):
        raise PreconditionError(
            f"Freshly registered head {row['id']} could not be promoted "
            f"to current."
        )
    logger.info(
        "NHSVM head final: %s (train_acc %.3f over %d rows)",
        row["id"], train_result.final_train_acc, len(rows),
    )
    return row


# ── Orchestrator ─────────────────────────────────────────────────

def ensure_preconditioned(
    cfg, category_set, *, taxonomy_id: str,
    heartbeat: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Probe, then execute only the missing stages.  Fail-loud.

    Returns a summary dict with the probe result and per-stage
    outcomes; raises ``PreconditionError`` (or the stage's own typed
    error) when an artifact cannot be finalized.
    """
    status = probe(cfg, category_set, taxonomy_id=taxonomy_id)
    summary: dict[str, Any] = {"probe": status.to_dict(), "stages": {}}
    if status.final:
        logger.info(
            "Pre-conditioning skipped: artifacts final for %r "
            "(vocab_sig %s)", taxonomy_id, status.vocab_sig[:12],
        )
        summary["skipped"] = True
        return summary

    logger.info(
        "Pre-conditioning %r: %s", taxonomy_id, "; ".join(status.reasons),
    )
    summary["skipped"] = False

    if not status.collection_final:
        summary["stages"]["semantic_collection"] = build_semantic_collection(
            cfg, category_set, taxonomy_id=taxonomy_id,
            vocab_sig=status.vocab_sig, heartbeat=heartbeat,
        )

    if not status.head_final:
        summary["stages"]["nhsvm_head"] = {
            k: v for k, v in build_nhsvm_head(
                cfg, category_set, taxonomy_id=taxonomy_id,
                vocab_sig=status.vocab_sig, heartbeat=heartbeat,
            ).items() if k in ("id", "head_sig", "status", "metrics")
        }

    post = probe(cfg, category_set, taxonomy_id=taxonomy_id)
    if not post.final:
        raise PreconditionError(
            f"Pre-conditioning completed its stages but the post-probe "
            f"still reports non-final artifacts: {'; '.join(post.reasons)}"
        )
    summary["post_probe"] = post.to_dict()
    return summary
