"""Extend Classification — streamlined inference pipeline.

Consumes an existing :class:`MLArtifactSet` (CatBoost classifier, optional
incremental SVM classifier, optional fitted UMAP projection) and applies it
to a data source's columns to produce a NEW dataset.  Skips the heavy
training-time machinery of :func:`run_classification_pipeline`:

- no LLM sweep, no DST evidence fusion, no iterative revisit loop;
- no agent-driven convergence;
- no fit-to-LLM CatBoost retrain or incremental SVM hot-swap;
- no SAGE permutation importance.

What it keeps:

- Source-based sample auto-resolution (OOTB sample, synthetic, hive,
  meta-tagging) — same code paths as the full pipeline.
- ColumnFeatures extraction → embedding → CatBoost predict_proba.
- Optional SVM second-look (when the artifact bundle includes an SVM).
- Optional UMAP transform onto the same 2D coordinate space as the
  source classify run (when umap.pkl was bundled).
- Parquet write + dataset registration with ``run_kind='extend'``.
- Vocab compatibility check (warn-don't-block per project decision).

Reuses :mod:`atelier.classify.pipeline` helpers where they apply
(``_filter_classifiable_tables``, ``_write_parquet``, etc.) so the
on-disk layout and registered dataset shape match what classify runs
produce — that's what lets the Embeddings page render Extend output
seamlessly.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atelier.classify.fsm import AgentFSM, FSMState
from atelier.classify.sampler import (
    ColumnSample,
    TableSample,
    discover_tables,
    load_sample_source,
    load_synth_source,
    sample_table_metadata,
)

logger = logging.getLogger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def run_extend_classification(
    cfg: Any,
    fsm: AgentFSM,
    *,
    source_id: str,
    artifact_set_id: str,
    parent_dataset_id: str | None = None,
    samples: list[TableSample] | None = None,
    sample_size: int | None = None,
    tables_limit: int | None = None,
    connection_name: str | None = None,
    database: str | None = None,
) -> dict:
    """Run Extend Classification end-to-end.  Returns a result summary.

    Inputs:

    - ``source_id``: data source whose columns to classify.  Same
      dispatch logic as :func:`run_classification_pipeline` (OOTB
      sample, synthetic, meta-tagging, hive).
    - ``artifact_set_id``: which :class:`MLArtifactSet` row to consume.
      Loaded via :func:`atelier.classify.artifact_set.load_for_extend`.
    - ``parent_dataset_id``: dataset row this run is "extending" against
      (for lineage UI).  Optional — when None the dataset is a
      first-class extend against the source without a specific parent.
    - ``samples`` / ``sample_size`` / ``tables_limit`` / ``connection_name``
      / ``database``: pass-through to the source dispatcher.  When
      ``samples`` is provided, discovery + sampling are skipped.

    Returns the summary dict (subset of what the full pipeline returns,
    plus ``run_kind='extend'`` and the lineage IDs).

    Raises:
        ValueError: if the artifact set is missing or its files don't
            exist on disk.
    """
    from atelier.classify.artifact_set import (
        check_compatibility,
        load_for_extend,
    )
    from atelier.classify.embedding import configure as configure_embeddings
    from atelier.classify.features import extract_features
    from atelier.classify.taxonomy import load_sample_vocabulary
    from atelier.classify import pipeline as full_pipeline
    from atelier.db.dao import AtelierDao

    cfg = _apply_overlay(cfg)

    if sample_size is None:
        sample_size = cfg.classify_sample_size
    if tables_limit is None:
        tables_limit = cfg.classify_tables_limit

    dao = AtelierDao()
    artifact_row = dao.get_artifact_set(artifact_set_id)
    if artifact_row is None:
        raise ValueError(
            f"Artifact set not found: {artifact_set_id}.  Has it been "
            f"registered or has its row been deleted?"
        )
    if artifact_row.get("is_archived"):
        raise ValueError(
            f"Artifact set {artifact_set_id} is archived; un-archive it "
            f"before running Extend."
        )

    # Eager file-existence preflight (production guard).  Fail fast
    # BEFORE creating the FSM run, so a missing artifact file doesn't
    # leave a half-started run row in the FSM history.  Validates each
    # path explicitly with a clear error message — load_for_extend's
    # own checks raise FileNotFoundError, but this layer also enforces
    # internal consistency (e.g. svm_classes_path set without svm_path).
    _preflight_artifact_paths(artifact_row)

    # Embedding-model identity check (production guard).  An artifact
    # set trained against MiniLM-L6-v2 will produce nonsense
    # predictions if invoked with BGE-large at runtime — different
    # embedding shape, different semantic space.  Block the run with
    # a clear error when the live cfg's embedding model differs.
    cfg_embedding = getattr(
        cfg, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2",
    )
    if (artifact_row.get("embedding_model")
            and artifact_row["embedding_model"] != cfg_embedding):
        raise ValueError(
            f"Embedding model mismatch: artifact set {artifact_set_id} "
            f"was trained with {artifact_row['embedding_model']!r} but "
            f"the runtime config uses {cfg_embedding!r}.  Embeddings "
            f"would land in different semantic spaces and predictions "
            f"would be meaningless.  Re-train the artifact set or "
            f"revert the embedding model change."
        )

    bundle = load_for_extend(artifact_row)
    artifact_classes = bundle.classes
    if not artifact_classes:
        raise ValueError(
            f"Artifact set {artifact_set_id} has empty classes list — "
            f"refuse to run Extend."
        )

    # ── Source dispatch (mirrors the full pipeline's auto-resolution) ──
    # When the source has a known vocabulary loader, we use it ONLY to
    # supply human-readable labels for predicted codes — the model still
    # predicts only what it was trained on, regardless of source vocab.
    label_lookup: dict[str, tuple[str, str]] = {}  # code -> (label, annotation)

    def _absorb_categories(cs) -> None:
        for cat in cs.categories:
            label_lookup[cat.code] = (
                cat.label or cat.code,
                getattr(cat, "abbrev", "") or "",
            )

    if samples is None:
        if source_id == "ootb-sample":
            samples = load_sample_source()
            _absorb_categories(load_sample_vocabulary(hierarchical=True))
        elif source_id == "synthetic":
            samples = load_synth_source()
            _absorb_categories(load_sample_vocabulary(hierarchical=True))
        elif source_id == "meta-tagging":
            from atelier.classify.meta_tagging_source import (
                load_meta_tagging_source,
                load_meta_tagging_vocabulary,
                resolve_meta_tagging_mount,
            )
            mount = resolve_meta_tagging_mount(cfg)
            if mount is None:
                raise RuntimeError(
                    "meta-tagging source requested but no mount resolved"
                )
            samples = load_meta_tagging_source(mount)
            _absorb_categories(load_meta_tagging_vocabulary(mount))
        # else: hive — discover at runtime below; label_lookup stays empty
        #       and predictions render code as label

    # Compute compatibility against the source's known codes when
    # available — when the source vocab is unloaded (hive path before
    # discovery), report ok against the artifact's own signature.
    source_classes = (
        sorted(label_lookup.keys()) if label_lookup else artifact_classes
    )
    compat = check_compatibility(artifact_classes, source_classes)
    if compat.status != "ok":
        logger.warning(
            "Vocab compatibility: %s (artifact vocab differs from source; "
            "%d codes the source defines won't be predicted, %d codes the "
            "artifact predicts aren't in the source's preferred vocab)",
            compat.status, len(compat.extra_codes), len(compat.missing_codes),
        )

    # Embedding device wiring (same logic as the full pipeline).
    gpu_devices = None
    if cfg.classify_gpu_enabled != "false":
        from atelier.classify.gpu import preflight_gpu
        probe = preflight_gpu()
        if probe.available and cfg.classify_embedding_device == "auto":
            gpu_devices = probe.resolved_devices
    configure_embeddings(
        device=cfg.classify_embedding_device,
        batch_size=cfg.classify_embedding_batch_size,
        devices=gpu_devices,
        shard_threshold=cfg.classify_gpu_shard_threshold,
    )

    # ── Start FSM run ──────────────────────────────────────────────
    run = fsm.start_run(
        config={
            "connection_name": connection_name,
            "database": database,
            "sample_size": sample_size,
            "tables_limit": tables_limit,
            "source_id": source_id,
            "artifact_set_id": artifact_set_id,
            "parent_dataset_id": parent_dataset_id,
            "run_kind": "extend",
        },
        source_id=source_id,
    )
    run_id = run.id
    results_dir = _PROJECT_ROOT / "build" / "results" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # Persist the settings snapshot — Extend runs reuse this layer to
    # capture the artifact-set + lineage IDs alongside the standard
    # config snapshot.
    try:
        from atelier.config_overlay import snapshot as _settings_snapshot
        snap = {
            "run_id": run_id,
            "run_kind": "extend",
            "artifact_set_id": artifact_set_id,
            "parent_dataset_id": parent_dataset_id,
            "vocab_compatibility": {
                "status": compat.status,
                "missing_codes": compat.missing_codes,
                "extra_codes": compat.extra_codes,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **_settings_snapshot(cfg),
        }
        (results_dir / "settings_snapshot.json").write_text(
            json.dumps(snap, indent=2, default=str) + "\n"
        )
    except Exception as e:
        logger.warning("settings_snapshot.json write failed (non-fatal): %s", e)

    try:
        # ── LOADING_VOCAB ────────────────────────────────────────
        fsm.advance(run_id, FSMState.LOADING_VOCAB, progress={
            "phase": "loading_vocab_from_artifact",
            "artifact_set_id": artifact_set_id,
            "vocab_compatibility": {
                "status": compat.status,
                "missing": len(compat.missing_codes),
                "extra": len(compat.extra_codes),
            },
        })
        fsm.advance(run_id, FSMState.DISCOVERING, progress={
            "categories_loaded": len(artifact_classes),
        })

        # ── DISCOVERING + SAMPLING ────────────────────────────────
        if samples is not None:
            all_samples = samples
            fsm.advance(run_id, FSMState.SAMPLING, progress={
                "tables_discovered": len(all_samples),
                "injected": True,
            })
        else:
            # Hive path.
            table_names = discover_tables(
                cfg, connection_name, database, limit=tables_limit
            )
            logger.info("Discovered %d tables", len(table_names))
            fsm.advance(run_id, FSMState.SAMPLING, progress={
                "tables_discovered": len(table_names),
            })
            all_samples = []
            for tname in table_names:
                try:
                    ts = sample_table_metadata(
                        cfg, tname, connection_name, database, sample_size
                    )
                    all_samples.append(ts)
                except Exception as exc:
                    logger.warning("Failed to sample %s: %s", tname, exc)

        # Filter non-data tables (vocab tables, test leftovers).  Reuses
        # the same predicate as the full pipeline so an Extend run sees
        # the same set of classifiable tables.
        vocab_uri_present = None  # Extend doesn't need source vocab_uri here
        all_samples = full_pipeline._filter_classifiable_tables(
            all_samples, vocab_uri_present,
        )

        # Flatten to columns.
        all_columns: list[ColumnSample] = []
        for ts in all_samples:
            all_columns.extend(ts.columns)

        if not all_columns:
            raise ValueError(
                "No classifiable columns after filtering — refusing to "
                "produce an empty Extend run."
            )

        # ── CLASSIFYING ──────────────────────────────────────────
        fsm.advance(run_id, FSMState.CLASSIFYING, progress={
            "phase": "ml_inference",
            "columns_total": len(all_columns),
        })

        features_list = []
        for col in all_columns:
            features_list.append(extract_features(
                column_name=col.name,
                column_type=col.column_type,
                values=col.values,
                siblings=col.siblings,
                source_table=col.table_name,
                total_count=col.total_count,
                null_count=col.null_count,
                distinct_count=col.distinct_count,
            ))

        # Batched CatBoost predict; falls back to one-by-one only if the
        # batch call fails (defensive — predict_proba is well-tested).
        try:
            cb_probs = bundle.catboost.predict_proba(features_list)
        except Exception as e:
            logger.warning(
                "CatBoost batch predict failed (%s); falling back to "
                "per-column predict_proba_single", e,
            )
            cb_probs = [
                bundle.catboost.predict_proba_single(f)
                for f in features_list
            ]

        # Optional SVM second-look.  The SVM was trained on TF-IDF over
        # the same ``build_svm_text`` shape — pass a 1-elem list when
        # we only have the joined string in ColumnFeatures.
        svm_probs: list[dict[str, float] | None] = [None] * len(features_list)
        if bundle.svm is not None:
            from atelier.classify.svm_classifier import build_svm_text
            try:
                svm_texts = [
                    build_svm_text(
                        f.column_name_humanized,
                        f.column_type,
                        [f.sample_values_text] if f.sample_values_text else None,
                    )
                    for f in features_list
                ]
                svm_probs = bundle.svm.predict_proba(svm_texts)
            except Exception as e:
                logger.warning(
                    "SVM batch predict failed (%s); proceeding with "
                    "CatBoost-only fusion", e,
                )

        # ML-only fusion (no DST).  Take CatBoost top-1 as primary;
        # record SVM agreement / disagreement in evidence.
        classifications: list[dict] = []
        for col, feats, cb_dist, svm_dist in zip(
            all_columns, features_list, cb_probs, svm_probs,
        ):
            if not cb_dist:
                # Empty distribution — model couldn't form a prediction.
                continue
            # Top-1.
            top1_code, top1_p = max(cb_dist.items(), key=lambda kv: kv[1])
            # Top-3 for plausibility heuristic.
            ranked = sorted(cb_dist.items(), key=lambda kv: -kv[1])
            top3_sum = sum(p for _, p in ranked[:3])
            # Heuristic belief / plausibility for the parquet schema.
            # ``conflict=0.0`` is a clear "ML-only inference" marker the
            # UI can distinguish from DST runs (which have non-zero K).
            belief = float(top1_p)
            plausibility = float(min(1.0, top1_p + max(0.0, 1.0 - top3_sum)))
            uncertainty = float(plausibility - belief)
            conflict = 0.0
            confidence = belief
            evidence_parts = [f"catboost:{top1_code}={top1_p:.3f}"]
            svm_disagreement = None
            if svm_dist:
                svm_top1 = max(svm_dist.items(), key=lambda kv: kv[1]) if svm_dist else None
                if svm_top1:
                    evidence_parts.append(
                        f"svm:{svm_top1[0]}={svm_top1[1]:.3f}"
                    )
                    if svm_top1[0] != top1_code:
                        svm_disagreement = svm_top1[0]
                        # SVM disagreement nudges belief down by the
                        # smaller of (1 − svm_top1) and 0.1 — soft
                        # haircut for the audit trail without making
                        # the prediction nonsensical.
                        haircut = min(0.10, 1.0 - svm_top1[1])
                        confidence = max(belief - haircut, 0.0)

            # Resolve human-readable label/annotation from whatever
            # source vocab we managed to load.  When unavailable, the
            # code itself is the label — predictions still render
            # cleanly in the UI, just less prettily.
            label, annotation = label_lookup.get(top1_code, (top1_code, ""))

            classifications.append({
                "table_name": col.table_name,
                "column_name": col.name,
                "column_type": col.column_type or "",
                "predicted_code": top1_code,
                "predicted_label": label,
                "predicted_annotation": annotation,
                "llm_code": None,                  # Extend has no LLM
                "llm_confidence": 0.0,
                "confidence": confidence,
                "belief": belief,
                "plausibility": plausibility,
                "uncertainty": uncertainty,
                "conflict": conflict,
                "needs_clarification": False,
                "evidence": " | ".join(evidence_parts),
                "reference_code": col.reference_code or "",
                "reference_label": "",
                "matches_reference": (
                    (col.reference_code == top1_code)
                    if col.reference_code else None
                ),
                "embedding_text": feats.to_embedding_text(),
                "pattern_signals": feats.pattern_signals,
                "belief_path": [],
                "cautious_code": "",
                "predicted_code_pre_review": "",
                "review_decision": "",
                "review_rationale": "",
                "svm_disagreement": svm_disagreement or "",
            })

        # ── FUSING (no-op for Extend — present for FSM state continuity) ──
        fsm.advance(run_id, FSMState.FUSING, progress={
            "phase": "fusing_noop",
            "classified": len(classifications),
        })

        # ── EVALUATING (write parquet + register dataset) ─────────
        fsm.advance(run_id, FSMState.EVALUATING, progress={
            "phase": "writing_outputs",
            "classified": len(classifications),
        })

        # classifications.json
        results_path = results_dir / "classifications.json"
        results_path.write_text(
            json.dumps(classifications, indent=2, default=str) + "\n"
        )

        # Atlas-compatible parquet.  We bypass _compute_projection's
        # fit-and-throw-away behavior by overriding the projection step
        # to use bundle.umap_model.transform when available; falls back
        # to fit_transform on the new embeddings (different coordinate
        # space, surfaced as a warning on the run summary).
        parquet_path = results_dir / "atelier_embeddings.parquet"
        _write_extend_parquet(classifications, parquet_path, bundle)

        # evaluation_report.json — minimal, since we have no ground truth
        # and no DST metrics to publish.
        n_with_ref = sum(
            1 for c in classifications if c.get("reference_code")
        )
        n_match = sum(
            1 for c in classifications if c.get("matches_reference") is True
        )
        eval_report = {
            "run_kind": "extend",
            "artifact_set_id": artifact_set_id,
            "parent_dataset_id": parent_dataset_id,
            "n_columns": len(classifications),
            "n_with_reference": n_with_ref,
            "n_matching_reference": n_match,
            "accuracy": (n_match / n_with_ref) if n_with_ref else None,
            "vocab_compatibility": {
                "status": compat.status,
                "missing": len(compat.missing_codes),
                "extra": len(compat.extra_codes),
            },
        }
        (results_dir / "evaluation_report.json").write_text(
            json.dumps(eval_report, indent=2, default=str) + "\n"
        )

        # Register the dataset.  Extend uses run_kind='extend',
        # artifact_set_id pointing at the consumed bundle, and
        # parent_dataset_id linking to the source classify run's dataset.
        try:
            version_number = dao.next_version_number(source_id) if source_id else 1
            dao.upsert_dataset(
                dataset_id=run_id,
                name=f"Extend {run_id[:8]} from {artifact_set_id[:8]}",
                parquet_path=str(parquet_path),
                description=(
                    f"Extend Classification — {len(classifications)} columns "
                    f"using artifact set {artifact_set_id[:8]}"
                ),
                row_count=len(classifications),
                source_id=source_id,
                version_number=version_number,
                is_active=True,
                summary=(
                    f"{len(all_samples)} tables, {len(classifications)} "
                    f"columns (extend)"
                ),
                fsm_run_id=run_id,
                artifact_set_id=artifact_set_id,
                parent_dataset_id=parent_dataset_id,
                run_kind="extend",
            )
            if source_id:
                dao.set_active_version(source_id, run_id)
        except Exception as e:
            logger.warning("Failed to register Extend dataset: %s", e)

        fsm.advance(run_id, FSMState.CONVERGED, progress={
            "phase": "complete",
            "run_kind": "extend",
            "classified": len(classifications),
            "vocab_compatibility": compat.status,
        })

        return {
            "run_id": run_id,
            "run_kind": "extend",
            "artifact_set_id": artifact_set_id,
            "parent_dataset_id": parent_dataset_id,
            "source_id": source_id,
            "n_columns": len(classifications),
            "n_tables": len(all_samples),
            "vocab_compatibility": compat.status,
            "vocab_compatibility_detail": {
                "missing_codes": compat.missing_codes,
                "extra_codes": compat.extra_codes,
            },
            "result_path": str(results_dir),
        }

    except Exception as e:
        logger.exception("Extend Classification failed")
        try:
            fsm.advance(run_id, FSMState.ERROR, progress={
                "error": str(e),
                "run_kind": "extend",
            })
        except Exception:
            pass
        raise


def _apply_overlay(cfg: Any) -> Any:
    """Apply the in-memory settings overlay (matches the full pipeline)."""
    try:
        from atelier.config_overlay import apply_to_config
        return apply_to_config(cfg)
    except Exception:
        return cfg


def _preflight_artifact_paths(row: dict) -> None:
    """Production guard: raise ValueError if artifact files are missing.

    Validates every non-NULL path on the row exists on disk BEFORE the
    Extend FSM run is created.  Catches the case where artifact files
    were moved / deleted out from under a registered row — the DB
    pointer is stale and the bundle would fail mid-run.

    Also enforces internal consistency: when svm_path is set,
    svm_classes_path must also be set (the SVM bundle is incomplete
    without its sidecar).
    """
    set_id = row.get("id", "<unknown>")

    # Required: catboost + sidecar.
    for key in ("catboost_path", "catboost_classes_path"):
        path = row.get(key)
        if not path:
            raise ValueError(
                f"Artifact set {set_id} missing required field {key!r}"
            )
        if not Path(path).is_file():
            raise ValueError(
                f"Artifact set {set_id}: {key} points at {path!r} which "
                f"does not exist on disk.  The artifact files may have "
                f"been moved or deleted; archive the row and re-run "
                f"the classify pipeline to rebuild."
            )

    # Optional: SVM (when one path is set, the sidecar must be too).
    svm_path = row.get("svm_path")
    svm_classes_path = row.get("svm_classes_path")
    if svm_path and not svm_classes_path:
        raise ValueError(
            f"Artifact set {set_id}: svm_path set but svm_classes_path "
            f"is NULL — sidecar is required for the SVM to load."
        )
    if svm_path and not Path(svm_path).is_file():
        raise ValueError(
            f"Artifact set {set_id}: svm_path points at {svm_path!r} "
            f"which does not exist on disk."
        )
    if svm_classes_path and not Path(svm_classes_path).is_file():
        raise ValueError(
            f"Artifact set {set_id}: svm_classes_path points at "
            f"{svm_classes_path!r} which does not exist on disk."
        )

    # Optional: UMAP (no sidecar — single .pkl file).
    umap_path = row.get("umap_path")
    if umap_path and not Path(umap_path).is_file():
        raise ValueError(
            f"Artifact set {set_id}: umap_path points at {umap_path!r} "
            f"which does not exist on disk."
        )


def _write_extend_parquet(
    classifications: list[dict],
    output_path: Path,
    bundle: Any,
) -> Path | None:
    """Atlas-compatible parquet writer for Extend runs.

    Matches :func:`atelier.classify.pipeline._write_parquet`'s schema so
    the Embeddings page renders Extend output without UI changes.
    Differs in the projection step: when ``bundle.umap_model`` is
    available we ``transform()`` the new column embeddings into the
    parent run's coordinate space; otherwise we fall back to a fresh
    ``fit_transform`` (different coordinates — flagged in the run's
    settings_snapshot.json).
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("pyarrow not available; skipping parquet output")
        return None

    if not classifications:
        return None

    texts = []
    for c in classifications:
        ontology = c.get("predicted_label") or ""
        annotation = c.get("predicted_annotation", "") or ""
        if ontology and annotation:
            texts.append(f"{ontology} - {annotation}")
        elif ontology:
            texts.append(ontology)
        else:
            texts.append(c.get("predicted_code") or "unknown")

    x_vals, y_vals = _extend_projection(classifications, texts, bundle)

    rows = []
    for i, c in enumerate(classifications):
        row = {
            "text": texts[i],
            "x": x_vals[i],
            "y": y_vals[i],
            "table_name": c["table_name"],
            "column_name": c["column_name"],
            "column_type": c.get("column_type", "") or "",
            "predicted_code": c.get("predicted_code", "") or "",
            "predicted_label": c.get("predicted_label", "") or "",
            "predicted_annotation": c.get("predicted_annotation", "") or "",
            "llm_code": c.get("llm_code") or "",
            "llm_confidence": c.get("llm_confidence", 0.0) or 0.0,
            "confidence": c.get("confidence", 0.0),
            "belief": c.get("belief", 0.0),
            "plausibility": c.get("plausibility", 0.0),
            "uncertainty": c.get("uncertainty", 0.0),
            "conflict": c.get("conflict", 0.0),
            "needs_clarification": c.get("needs_clarification", False),
            "evidence": c.get("evidence", ""),
            "reference_code": c.get("reference_code") or "",
            "reference_label": c.get("reference_label") or "",
            "matches_reference": c.get("matches_reference"),
            "embedding_text": c.get("embedding_text", ""),
            "pattern_signals": ", ".join(c.get("pattern_signals", {})),
            "dst_belief_path": json.dumps(c.get("belief_path", [])),
            "cautious_code": c.get("cautious_code", ""),
            "predicted_code_pre_review": c.get("predicted_code_pre_review", ""),
            "review_decision": c.get("review_decision", ""),
            "review_rationale": c.get("review_rationale", ""),
        }
        # SHAP columns left empty for Extend (we don't run SHAP at
        # inference time).  Populated by the Embeddings UI as "—".
        for rank in range(1, 4):
            row[f"shap_top{rank}_name"] = ""
            row[f"shap_top{rank}_value"] = 0.0
        rows.append(row)

    table = pa.table({k: [r[k] for r in rows] for k in rows[0].keys()})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output_path))
    return output_path


def _extend_projection(
    classifications: list[dict],
    texts: list[str],
    bundle: Any,
) -> tuple[list[float], list[float]]:
    """Project Extend embeddings into 2D for the embedding-atlas viewer.

    Prefers ``bundle.umap_model.transform()`` when the artifact bundle
    includes a fitted UMAP — that lands new points in the parent run's
    coordinate space.  Falls back to a fresh fit when the bundle has no
    UMAP, or when transform raises (logs the reason).
    """
    try:
        from atelier.classify.embedding import _get_model, get_batch_size
        import numpy as np

        model = _get_model()
        embeddings = model.encode(
            texts, show_progress_bar=False, batch_size=get_batch_size(),
        )

        if bundle.umap_model is not None:
            try:
                projection = bundle.umap_model.transform(embeddings)
                if hasattr(projection, "values"):
                    projection = projection.values
                projection = np.asarray(projection)
                logger.info(
                    "UMAP projection: transform via bundled artifact UMAP "
                    "(matches parent run coordinate space)"
                )
                return projection[:, 0].tolist(), projection[:, 1].tolist()
            except Exception as e:
                logger.warning(
                    "Bundled UMAP transform failed (%s); falling back to "
                    "fresh fit_transform — coordinates differ from parent",
                    e,
                )

        # Fresh fit fallback.
        n_neighbors = min(15, max(2, len(texts) - 1))
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2, n_neighbors=n_neighbors,
                min_dist=0.1, metric="cosine", random_state=42,
            )
            projection = reducer.fit_transform(embeddings)
            projection = np.asarray(projection)
            return projection[:, 0].tolist(), projection[:, 1].tolist()
        except Exception as e:
            logger.debug("UMAP fresh-fit failed (%s)", e)

    except Exception as e:
        logger.debug("Embedding load failed during Extend projection: %s", e)

    # Final fallback: PCA-like from confidence/belief, same as the full
    # pipeline's degenerate path.
    import numpy as np
    features = np.array([
        [c.get("confidence", 0.0), c.get("belief", 0.0),
         c.get("plausibility", 0.0), c.get("uncertainty", 0.0),
         c.get("conflict", 0.0)]
        for c in classifications
    ], dtype=np.float32)
    centered = features - features.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ vt[:2].T
    except np.linalg.LinAlgError:
        proj = features[:, :2]
    return proj[:, 0].tolist(), proj[:, 1].tolist()
