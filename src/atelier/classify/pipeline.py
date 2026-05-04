# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""End-to-end classification pipeline with bootstrap convergence.

Drives the AgentFSM through:
  LOADING_VOCAB → DISCOVERING → SAMPLING
    → LLM_SWEEP → VALIDATING → (revisit loop until converged)
    → CLASSIFYING → FUSING → EVALUATING → CONVERGED

The LLM is a required evidence source.  The backend is selected via
``ANTHROPIC_SUBAGENT_MODEL`` (backend type inferred from model format —
Bedrock ARN → ``BedrockStructuredBackend``, plain Anthropic ID →
``AnthropicStructuredBackend``).  An explicit classify LLM
(``ATELIER_LLM_API_KEY``) overrides the subagent model.

For dev/test, inject ``samples=`` and ``llm_backend=`` explicitly.

Writes results to build/results/{run_id}/ as JSON and parquet.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atelier.classify.belief import (
    FrameOfDiscernment,
    HierarchicalClassification,
    combine_multiple,
)
from atelier.classify.evaluation import evaluate_classifications
from atelier.classify.features import extract_features
from atelier.classify.fsm import AgentFSM, FSMState
from atelier.classify.mass_functions import (
    DEFAULT_PATTERN_MAP,
    DiscountConfig,
    catboost_to_mass,
    cosine_to_mass,
    llm_to_mass,
    name_match_to_mass,
    pattern_to_mass,
    resolve_pattern_map,
    svm_to_mass,
)
from atelier.classify.sampler import (
    ColumnSample,
    TableSample,
    discover_tables,
    load_sample_source,
    load_synth_source,
    sample_table_metadata,
)
from atelier.classify.taxonomy import (
    HierarchicalCategorySet,
    load_annotations_from_hive,
    load_annotations_from_json,
    load_sample_vocabulary,
    load_universal_vocabulary,
    save_annotations_json,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# Table-name tokens that signal "this isn't real data — don't classify
# it".  The annotations table is the vocabulary backing the run; ``ice_t1``
# is a historical test leftover.  Extend this list rather than inlining
# new cases at the call site.
_NON_DATA_TABLE_NAMES: frozenset[str] = frozenset({
    "annotations",
    "ice_t1",
})


def _convergence_progress(
    state,
    column_names: list[str],
    disagreements,
    boot_cfg,
) -> dict:
    """Build the iteration-boundary FSM progress payload.

    Surfaces the thesis-aligned signals the live Status page renders
    (``mean_gap``, ``mean_bel``, ``clarity``, ``gap_contraction_rate``,
    revisit-queue count + fraction, LLM-fit count + fraction *f*) plus
    the reference thresholds the UI needs to color-code them.

    The composite ``residual_norm`` and its associated
    ``contraction_rate`` (which is a ratio over the heuristic
    composite, not over the actual stopping criterion ``mean_gap``)
    are deliberately NOT surfaced here — see
    ``.claude/plans/how-about-extend-the-golden-sedgewick.md`` for the
    design rationale.  They remain in ``IterationMetrics`` and
    ``column_trajectories.json`` for post-run analysis.

    The latest ``IterationMetrics`` snapshot is read from
    ``state.iteration_metrics[-1]`` when present; before the first
    iteration completes (e.g. agent-convergence entry) the snapshot
    keys are omitted and the UI falls back to the pre-iteration view.
    """
    n_cols = max(1, len(column_names))
    n_disagree = len(disagreements) if disagreements is not None else 0
    payload: dict = {
        # Reference thresholds for UI color-coding.
        "gap_threshold": boot_cfg.gap_threshold,
        "bel_floor": boot_cfg.bel_floor,
        "clarity_target": boot_cfg.clarity_target,
        # Thesis core: the LLM-labeled fraction *f* in the operator's
        # thesis, rendered explicitly.
        "llm_fit_labels": len(state.labels),
        "llm_fit_fraction": round(len(state.labels) / n_cols, 4),
        # Revisit queue: count is the LLM budget for the next iteration;
        # fraction is the thesis-relevant scale-invariant view.
        "disagreements_count": n_disagree,
        "disagreements_frac": round(n_disagree / n_cols, 4),
    }
    if state.iteration_metrics:
        m = state.iteration_metrics[-1]
        payload.update({
            "mean_gap": m.mean_gap,
            "mean_bel": m.mean_bel,
            "clarity": round(1.0 - m.frac_unclear, 4),
            "gap_contraction_rate": m.gap_contraction_rate,
            "indep_tier_disagreement_frac": m.indep_tier_disagreement_frac,
        })
    return payload


def _filter_classifiable_tables(
    samples: list[TableSample],
    vocab_uri: str | None,
) -> list[TableSample]:
    """Drop tables that shouldn't be treated as data (vocab + test leftovers).

    Matches by **table name only** (case-insensitive), not schema or
    qualifier, so `default.annotations`, `meta.annotations`, and plain
    `annotations` all get filtered.  Additionally strips any table whose
    name matches the trailing component of ``vocab_uri`` — when the
    vocab URI is e.g. ``"default.annotations"`` we want that specific
    table gone from classification even if it doesn't match the
    hard-coded list above.
    """
    skip = set(_NON_DATA_TABLE_NAMES)
    if vocab_uri:
        # vocab_uri shapes: "db.annotations" (hive), "annotations.csv"
        # (filesystem file path tail), "meta.vocab", etc.  Strip scheme,
        # directory, extension, and db-qualifier to get the bare name.
        bare = vocab_uri.rsplit("/", 1)[-1]
        bare = bare.rsplit(".", 1)[0] if bare.lower().endswith(".csv") else bare
        bare = bare.rsplit(".", 1)[-1] if "." in bare else bare
        if bare:
            skip.add(bare.lower())

    kept: list[TableSample] = []
    dropped: list[str] = []
    for ts in samples:
        if ts.name.lower() in skip:
            dropped.append(ts.name)
            continue
        kept.append(ts)
    if dropped:
        logger.info(
            "Excluded %d non-data tables from classification: %s",
            len(dropped), dropped,
        )
    return kept


def _install_fit_to_llm_catboost(
    cfg,
    state,
    samples_by_name: dict[str, ColumnSample],
    category_set: HierarchicalCategorySet,
    save_path: Path | None = None,
) -> None:
    """Fit an in-memory CatBoost on LLM-labeled columns and install it.

    Gated by ``cfg.classify_catboost_fit_to_llm``.  No-ops when the LLM
    sweep hasn't produced at least ``fit_to_llm_min_labels`` pairs, when
    all labels collapse to a single class, or when the embedding
    backend is unavailable.  Emits a progress log in either case.

    When ``save_path`` is provided, the trained CatBoost model is also
    persisted to disk (native ``.cbm`` format + sibling ``.classes.json``)
    so downstream consumers can replay inference without retraining.
    This is the hook that makes ML-only reproducibility auditable from
    a run directory alone — pair it with ``svm_frontier.pkl`` and the
    full ML stack for a given run is on disk.
    """
    min_labels = int(getattr(cfg, "classify_catboost_fit_to_llm_min_labels", 30))
    if len(state.labels) < min_labels:
        logger.info(
            "fit_to_llm: only %d LLM labels available (need >= %d) — skipping",
            len(state.labels), min_labels,
        )
        return

    features_list: list = []
    codes: list[str] = []
    for col_name, llm_code in state.labels.items():
        if not llm_code:
            continue
        sample = samples_by_name.get(col_name)
        if sample is None:
            continue
        try:
            feats = extract_features(
                column_name=sample.name,
                column_type=sample.column_type,
                values=sample.values,
                siblings=sample.siblings,
                null_count=sample.null_count,
                total_count=sample.total_count,
                source_table=sample.table_name,
                distinct_count=sample.distinct_count,
            )
        except Exception:
            continue
        features_list.append(feats)
        codes.append(llm_code)

    if len(features_list) < min_labels:
        logger.info(
            "fit_to_llm: %d usable (features, code) pairs after filtering — skipping",
            len(features_list),
        )
        return

    from atelier.classify.ml_train import fit_catboost_to_llm_labels
    from atelier.classify import ml_inference

    classifier = fit_catboost_to_llm_labels(
        features_list, codes,
        iterations=int(cfg.classify_catboost_iterations),
        depth=int(cfg.classify_catboost_depth),
        learning_rate=float(cfg.classify_catboost_learning_rate),
    )
    if classifier is None:
        return

    ml_inference.install_catboost(classifier)
    if save_path is not None:
        try:
            classifier.save(save_path)
            logger.info("fit_to_llm: CatBoost persisted to %s", save_path)
        except Exception as exc:
            logger.warning("fit_to_llm: failed to save CatBoost to %s: %s", save_path, exc)
    logger.info(
        "fit_to_llm: installed CatBoost trained on %d LLM labels across %d classes",
        len(features_list), len(set(codes)),
    )
    if save_path is not None:
        try:
            classifier.save(save_path)
        except Exception as exc:
            logger.warning("fit_to_llm: could not save CatBoost to %s: %s",
                           save_path, exc)


def run_classification_pipeline(
    cfg,
    fsm: AgentFSM,
    *,
    source_id: str | None = None,
    connection_name: str | None = None,
    database: str = "default",
    sample_size: int | None = None,
    tables_limit: int | None = None,
    samples: list[TableSample] | None = None,
    category_set: HierarchicalCategorySet | None = None,
    llm_backend=None,
    vocab_uri: str | None = None,
) -> dict[str, Any]:
    """Run the classification pipeline with LLM-driven convergence.

    The pipeline requires an LLM backend for evidence fusion.  When no
    explicit ``llm_backend`` is provided, one is created from config:
    ``ATELIER_LLM_API_KEY`` takes priority, then ``ANTHROPIC_SUBAGENT_MODEL``
    (backend type inferred from model format).

    For dev/test without hive or real LLM, inject ``samples=`` and
    ``llm_backend=`` explicitly.

    Args:
        cfg: AtelierConfig.
        fsm: AgentFSM instance for state tracking.
        source_id: Data source to classify. When "ootb-sample", loads
            sample CSVs and the expanded vocabulary automatically.
        connection_name: CAI data connection name.
        database: Hive database to classify.
        sample_size: Rows to sample per table.
        tables_limit: Max tables to discover.
        samples: Pre-loaded TableSamples (skip discover/sample phases).
        category_set: Pre-loaded vocabulary (skip vocab loading).
        llm_backend: Injected LLM backend (for testing). Created from
            config when None.

    Returns:
        Pipeline result summary dict.

    Raises:
        ValueError: If no LLM backend is available.
    """
    # ── Runtime overlay (settings page) ───────────────────────
    # Session-level tuning values override cfg here; the overlay
    # is a no-op when empty, so production runs behave normally.
    from atelier.config_overlay import apply_to_config
    cfg = apply_to_config(cfg)

    # ── Design invariants (non-negotiable) ────────────────────
    # These floors exist because the values below them produce output
    # that *looks like* a classification run but isn't the pipeline we
    # publish accuracy numbers for.  Project directive — see
    # docs/src/architecture/classification.md and project memory
    # ``feedback_pipeline_invariants.md``.  Do not widen or remove
    # without an explicit sign-off from the user; they have been
    # suggested as memory-pressure mitigations in the past and are
    # not acceptable.
    min_iterations_floor = 2
    if cfg.classify_bootstrap_max_iterations < min_iterations_floor:
        raise ValueError(
            f"classify.bootstrap.max_iterations = "
            f"{cfg.classify_bootstrap_max_iterations} violates the project "
            f"design directive (minimum {min_iterations_floor}).  The "
            f"bootstrap loop's revisit pass is part of the published-"
            f"accuracy pipeline; setting max_iterations=1 skips it and "
            f"produces a different algorithm.  If you're hitting resource "
            f"pressure, fix the resource budget — do not shrink the pipeline."
        )
    if cfg.classify_bootstrap_min_iterations < min_iterations_floor:
        raise ValueError(
            f"classify.bootstrap.min_iterations = "
            f"{cfg.classify_bootstrap_min_iterations} violates the project "
            f"design directive (minimum {min_iterations_floor}).  Without "
            f"a min_iterations floor the loop can exit on iteration 1 "
            f"when the initial disagreement set happens to be empty — "
            f"the iterative DST-fusion component never runs and the "
            f"'CONVERGED' state is silently misleading.  See "
            f"feedback_pipeline_invariants.md."
        )
    if cfg.classify_bootstrap_min_iterations > cfg.classify_bootstrap_max_iterations:
        raise ValueError(
            f"classify.bootstrap.min_iterations "
            f"({cfg.classify_bootstrap_min_iterations}) must not exceed "
            f"classify.bootstrap.max_iterations "
            f"({cfg.classify_bootstrap_max_iterations})."
        )
    if not cfg.classify_catboost_fit_to_llm:
        raise ValueError(
            "classify.catboost.fit_to_llm = false violates the project "
            "design directive.  Fit-to-LLM is the training regime under "
            "which the published accuracy numbers were obtained; "
            "disabling it swaps in a pre-trained CatBoost whose "
            "attributions do not agree with the current LLM's decisions.  "
            "If you're hitting resource pressure, fix the resource budget "
            "— do not change the training regime."
        )

    # ── Respect HOCON-configured discovery limits ─────────────
    # When callers (gateway, service) don't pass these explicitly,
    # fall back to the configured values instead of the hard-coded
    # function defaults.  Prevents the function-default from silently
    # overriding operator-configured env vars (ATELIER_CLASSIFY_TABLES_LIMIT,
    # ATELIER_CLASSIFY_SAMPLE_SIZE).
    if sample_size is None:
        sample_size = cfg.classify_sample_size
    if tables_limit is None:
        tables_limit = cfg.classify_tables_limit

    # ── Source-based auto-resolution ──────────────────────────
    # When source_id is provided, auto-load samples and vocabulary.
    # The OOTB sample and the local Synthetic corpus both pair with
    # the expanded 316-leaf ICE ontology — their reference codes
    # share that vocabulary, so the LLM prompts and fusion frame are
    # identical.  Synthetic is local-dev only and never shipped OOTB.
    if source_id == "ootb-sample" and samples is None:
        samples = load_sample_source()
        if category_set is None:
            category_set = load_sample_vocabulary(hierarchical=True)
    elif source_id == "synthetic" and samples is None:
        samples = load_synth_source()
        if category_set is None:
            category_set = load_sample_vocabulary(hierarchical=True)
    elif source_id == "meta-tagging" and samples is None:
        # Private reference source — mount path never committed to git.
        from atelier.classify.meta_tagging_source import (
            load_meta_tagging_source,
            load_meta_tagging_vocabulary,
            resolve_meta_tagging_mount,
        )
        mount = resolve_meta_tagging_mount(cfg)
        if mount is None:
            raise RuntimeError(
                "meta-tagging source requested but no mount resolved — "
                "set ATELIER_META_TAGGING_DIR or cfg.classify_meta_tagging_dir "
                "to a directory containing annotations.csv"
            )
        samples = load_meta_tagging_source(mount)
        if category_set is None:
            category_set = load_meta_tagging_vocabulary(mount)
    # ── LLM backend resolution ────────────────────────────────
    # The pipeline cannot function without an LLM.  Resolve early
    # so callers get a clear error before any FSM state is created.
    if llm_backend is None:
        from atelier.classify.llm_backend import create_backend_from_cfg
        # create_backend_from_cfg raises ValueError when no creds
        llm_backend = create_backend_from_cfg(cfg)

    # ── Embedding acceleration ────────────────────────────────
    # Multi-GPU sharding kicks in automatically when preflight_gpu()
    # reports more than one device AND the operator hasn't forced
    # classify_gpu_enabled = "false".
    from atelier.classify.embedding import configure as configure_embeddings
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

    run = fsm.start_run(
        config={
            "connection_name": connection_name,
            "database": database,
            "sample_size": sample_size,
            "tables_limit": tables_limit,
            "source_id": source_id,
        },
        source_id=source_id,
    )
    run_id = run.id

    build_dir = _PROJECT_ROOT / "build"
    results_dir = build_dir / "results" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # Persist the settings-at-start so the UI can show historical vs
    # current in the adaptive focus section even for past runs.
    try:
        from atelier.config_overlay import snapshot as _settings_snapshot
        snapshot_path = results_dir / "settings_snapshot.json"
        snap = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **_settings_snapshot(cfg),
        }
        snapshot_path.write_text(json.dumps(snap, indent=2, default=str) + "\n")
    except Exception as e:
        logger.warning("settings_snapshot.json write failed (non-fatal): %s", e)

    try:
        # Pre-LLM phase observability — proof-of-progress paradigm.
        # Heavy Hive reads (annotations load, table discovery, per-
        # table sampling) have no native timeout via cml.data_v1;
        # phase_heartbeat ensures FSM.updated_at advances every 5s
        # while the work runs so nautilus's stall detector can
        # distinguish "actively waiting on Hive" from "thread died."
        # See atelier.classify.phase_heartbeat for the design.
        from atelier.classify.phase_heartbeat import phase_heartbeat
        from atelier.classify.sampler import probe_connection

        # ── LOADING_VOCAB ────────────────────────────────────────
        fsm.advance(run_id, FSMState.LOADING_VOCAB, progress={"step": "loading_vocab"})

        # Lightweight liveness probe — fails fast (5s budget) on a
        # dead Hive connection BEFORE we invest in the heavy
        # annotations-load query.  Logs and continues on probe
        # failure; the heavy query can still proceed but operators
        # have a clear signal in the log.
        hive_uri_present = bool(connection_name) or bool(getattr(cfg, "cml_data_connection_names", []))
        if hive_uri_present and category_set is None:
            probe_connection(cfg, connection_name)

        with phase_heartbeat(
            fsm, run_id, FSMState.LOADING_VOCAB,
            interval_s=float(getattr(cfg, "classify_phase_heartbeat_interval_s", 5.0)),
            label="loading_vocab",
        ):
            if category_set is None:
                category_set = _load_vocabulary(
                    cfg, build_dir, connection_name,
                    vocab_uri=vocab_uri, database=database,
                )
        logger.info("Loaded %d leaf categories", len(category_set.categories))

        if not isinstance(category_set, HierarchicalCategorySet):
            raise RuntimeError("Expected HierarchicalCategorySet")

        # Vocabulary quality check — flags label collisions like
        # "Web Browser" / "WebBrowser" that cause non-deterministic
        # name-match resolution.  Warn-only by default; gated to
        # raise via classify.taxonomy.strict_validation = true.
        from atelier.classify.taxonomy import validate_taxonomy
        taxonomy_findings = validate_taxonomy(category_set)
        if taxonomy_findings:
            errors = [f for f in taxonomy_findings if f.severity == "error"]
            warnings = [f for f in taxonomy_findings if f.severity == "warning"]

            # Errors surface individually — they're rare, load-bearing,
            # and signal structural vocabulary problems (duplicate
            # codes etc.) the rest of the pipeline can't work around.
            for f in errors:
                logger.warning(
                    "Taxonomy ERROR [%s]: %s", f.kind, f.detail,
                )

            # Warnings get a single summary line + a sidecar JSON
            # alongside the run's other artifacts.  35 individual
            # logger.warning lines per run is noise on the operator's
            # console; the structured findings are the right surface
            # for the vocabulary team to consume from the sidecar.
            if warnings:
                from collections import Counter
                kind_counts = Counter(f.kind for f in warnings)
                kind_summary = ", ".join(
                    f"{n} {kind}" for kind, n in sorted(kind_counts.items())
                )
                logger.info(
                    "Taxonomy validation: %d errors, %d warnings (%s); "
                    "see %s/taxonomy_findings.json",
                    len(errors), len(warnings), kind_summary, results_dir,
                )
                try:
                    findings_payload = {
                        "errors": [
                            {
                                "kind": f.kind,
                                "severity": f.severity,
                                "codes": list(f.codes),
                                "detail": f.detail,
                            }
                            for f in errors
                        ],
                        "warnings": [
                            {
                                "kind": f.kind,
                                "severity": f.severity,
                                "codes": list(f.codes),
                                "detail": f.detail,
                            }
                            for f in warnings
                        ],
                    }
                    (results_dir / "taxonomy_findings.json").write_text(
                        json.dumps(findings_payload, indent=2) + "\n",
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to write taxonomy_findings.json: %s", exc,
                    )

            strict = bool(getattr(cfg, "classify_taxonomy_strict_validation", False))
            if strict and (errors or warnings):
                raise RuntimeError(
                    f"Taxonomy validation failed under strict mode: "
                    f"{len(errors)} error(s) + {len(warnings)} warning(s).  "
                    f"Disable strict_validation or fix the source vocabulary."
                )
            if errors and not strict:
                # Errors (e.g. duplicate codes) always raise — they
                # break category-set invariants the rest of the
                # pipeline assumes.
                raise RuntimeError(
                    f"Taxonomy has {len(errors)} structural error(s); first: {errors[0].detail}"
                )

        frame = FrameOfDiscernment(
            category_set,
            confusable_pairs=_build_confusable_pairs(category_set),
        )
        fsm.advance(run_id, FSMState.DISCOVERING, progress={
            "categories_loaded": len(category_set.categories),
        })

        # ── DISCOVERING + SAMPLING ────────────────────────────────
        heartbeat_interval = float(getattr(cfg, "classify_phase_heartbeat_interval_s", 5.0))
        if samples is not None:
            all_samples = samples
            fsm.advance(run_id, FSMState.SAMPLING, progress={
                "tables_discovered": len(all_samples),
                "injected": True,
            })
        else:
            with phase_heartbeat(
                fsm, run_id, FSMState.DISCOVERING,
                interval_s=heartbeat_interval, label="discovering",
            ):
                table_names = discover_tables(
                    cfg, connection_name, database, limit=tables_limit
                )
            logger.info("Discovered %d tables", len(table_names))

            fsm.advance(run_id, FSMState.SAMPLING, progress={
                "tables_discovered": len(table_names),
            })

            all_samples: list[TableSample] = []
            with phase_heartbeat(
                fsm, run_id, FSMState.SAMPLING,
                interval_s=heartbeat_interval, label="sampling",
            ) as sampling_ctx:
                for i, tname in enumerate(table_names):
                    sampling_ctx["tables_sampled"] = i
                    sampling_ctx["current_table"] = tname
                    try:
                        ts = sample_table_metadata(
                            cfg, tname, connection_name, database, sample_size
                        )
                        all_samples.append(ts)
                    except Exception as exc:
                        logger.warning("Failed to sample %s: %s", tname, exc)
                sampling_ctx["tables_sampled"] = len(table_names)

        # Strip tables that shouldn't be classified (vocabulary tables,
        # internal test leftovers).  The annotations table IS the vocab,
        # not data; classifying it pollutes the accuracy signal.
        tables_before_filter = len(all_samples)
        all_samples = _filter_classifiable_tables(all_samples, vocab_uri)
        tables_after_filter = len(all_samples)
        # Advance the FSM with the post-filter count so the UI shows
        # tables that will ACTUALLY be classified, not the raw Hive
        # discovery count.  Also report the pre-filter count as
        # ``tables_discovered_raw`` for operators who want to see the
        # full enumeration.
        if tables_before_filter != tables_after_filter:
            fsm.advance(run_id, FSMState.SAMPLING, progress={
                "tables_discovered_raw": tables_before_filter,
                "tables_classifiable": tables_after_filter,
                "tables_filtered": tables_before_filter - tables_after_filter,
            })

        # Apply curated-reference CSV (when configured) so evaluation_report
        # gets real accuracy numbers.  Hive-backed runs don't carry a
        # reference through sample_table_metadata, but an operator with an
        # external reference (reviewer xlsx → CSV) can point the pipeline
        # at it via cfg.classify_reference_uri.
        ref_uri = getattr(cfg, "classify_reference_uri", "") or ""
        if ref_uri:
            try:
                from atelier.classify.reference import (
                    apply_reference,
                    load_reference_csv,
                )
                # Pass category_set so rows with only a mnemonic
                # (shape emitted by ingest_reference when the
                # reviewer xlsx has no explicit code column) resolve
                # via the vocabulary rather than being silently dropped.
                ref_map = load_reference_csv(
                    ref_uri, _PROJECT_ROOT, category_set=category_set,
                )
                hits = apply_reference(all_samples, ref_map)
                if hits:
                    logger.info(
                        "Curated reference applied to %d/%d columns from %s",
                        hits,
                        sum(len(t.columns) for t in all_samples),
                        ref_uri,
                    )
                else:
                    logger.warning(
                        "Reference URI %s resolved but matched zero columns — "
                        "check column_name qualifier convention",
                        ref_uri,
                    )
            except Exception as exc:
                logger.warning("Reference injection failed (non-fatal): %s", exc)

        # Reference-column invariant.  Synth-generator answer-key
        # columns (name pattern ``^(attr|code|col|data|field|item|key|
        # ref|val|var)_\d+(_\d+)*$``) encode the paired natural-named
        # column's reference code directly in their name; they must
        # never be classified (trivial by name parse) or appear in
        # sibling contexts (would leak answers into other columns'
        # embeddings).  The meta-tagging loader filters internally;
        # the Hive sampler does not.  Applying the filter here means
        # every loader path ends up with reference columns excluded,
        # with zero behavior change on production data (pattern doesn't
        # match production column names).
        #
        # Gated by ``classify_exclude_reference_columns`` so UAT
        # reviewers can demonstrate accuracy in both configurations
        # (the toggle lives on the Status page).  Default ON; flag
        # exists purely for the UAT synth corpus that motivated it
        # and will be removed once that dataset is retired.
        if cfg.classify_exclude_reference_columns:
            from atelier.classify.meta_tagging_source import exclude_reference_columns
            pre_filter_cols = sum(len(t.columns) for t in all_samples)
            all_samples = exclude_reference_columns(all_samples)
            post_filter_cols = sum(len(t.columns) for t in all_samples)
            if pre_filter_cols != post_filter_cols:
                logger.info(
                    "Reference-column exclusion: %d → %d columns "
                    "(%d answer-key columns dropped from %d tables)",
                    pre_filter_cols, post_filter_cols,
                    pre_filter_cols - post_filter_cols, len(all_samples),
                )
        else:
            logger.info(
                "Reference-column exclusion DISABLED — answer-key "
                "columns will be sent through the classifier (UAT "
                "demonstration mode only; see Status page toggle)."
            )

        # Flatten to column list with table mapping
        all_columns: list[ColumnSample] = []
        column_table: dict[str, str] = {}
        for ts in all_samples:
            for col in ts.columns:
                all_columns.append(col)
                column_table[col.name] = ts.name

        total_columns = len(all_columns)
        logger.info("Sampled %d columns across %d tables", total_columns, len(all_samples))

        samples_by_name: dict[str, ColumnSample] = {c.name: c for c in all_columns}
        column_names = list(samples_by_name.keys())

        # ── Bootstrap config + LLM prompts ────────────────────────
        from atelier.classify.bootstrap import (
            BootstrapConfig,
            BootstrapState,
            FatalLLMError,
            bootstrap_config_from_cfg,
            _coverage,
            _identify_disagreements,
            _identify_uncertain_columns,
            _llm_revisit,
            _llm_sweep,
            _mean_gap,
            _mean_k,
            _run_ml_validation,
            record_iteration_metrics,
            should_stop_early,
        )
        from atelier.classify.llm_backend import (
            build_category_tree,
            build_system_prompt,
        )

        boot_cfg = bootstrap_config_from_cfg(cfg)
        category_table = build_category_tree(category_set)
        system_prompt = build_system_prompt(category_table, category_set=category_set)

        # Wire config → ml_inference model paths
        from atelier.classify import ml_inference
        ml_inference.configure_paths(
            catboost_path=cfg.classify_catboost_model_path,
            svm_path=cfg.classify_svm_model_path,
        )

        discounts = DiscountConfig.from_cfg(cfg)

        # ── Ontology→user-taxonomy alignment for SVM evidence ──────
        # The synth-trained SVM emits ICE.* (bundled-ontology) codes;
        # without a per-vocabulary alignment, those predictions never
        # appear as singletons in the user-taxonomy frame and the SVM
        # contributes nothing.  Build the alignment once per (vocab,
        # model) tuple and stash it on the run for the SVM evidence
        # site to consume.  See ``ontology_alignment.py`` for the
        # independence/discount rationale and known caveats.
        from atelier.classify.ontology_alignment import build_alignment
        try:
            svm_alignment = build_alignment(
                category_set=category_set,
                llm_backend=llm_backend,
                system_prompt=system_prompt,
                model_name=getattr(cfg, "classify_subagent_model", None) or "unknown",
            )
        except Exception as exc:
            logger.warning(
                "ontology_alignment: build failed — proceeding without "
                "(SVM evidence will be vacuous on user-taxonomy runs): %s",
                exc,
            )
            svm_alignment = {}

        # Try sentence-transformers for cosine
        has_embeddings = False
        try:
            from atelier.classify.embedding import classify_cosine
            has_embeddings = True
        except ImportError:
            logger.warning("sentence-transformers not available; using name+pattern only")

        # ── MC Stratification ──────────────────────────────────────
        from atelier.classify.monte_carlo import MCConfig as _MCConfig
        mc_cfg = _MCConfig.from_cfg(cfg)
        mc_plan = None

        # ── Row-Level MC ──────────────────────────────────────────
        from atelier.classify.row_sampler import RowMCConfig
        row_mc_cfg = RowMCConfig.from_cfg(cfg)

        if total_columns >= mc_cfg.min_corpus_size and has_embeddings:
            from atelier.classify.monte_carlo import (
                pre_classify as _pre_classify,
                stratify as _stratify,
                select_sample as _select_sample,
            )
            logger.info(
                "MC sampling: %d columns >= threshold %d",
                total_columns, mc_cfg.min_corpus_size,
            )
            fsm.advance(run_id, FSMState.SAMPLING, progress={
                "phase": "mc_stratification",
                "columns_total": total_columns,
            })
            pre_results = _pre_classify(
                column_names, samples_by_name, category_set, frame,
                has_embeddings,
            )
            strata = _stratify(pre_results, mc_cfg)
            mc_plan = _select_sample(strata, mc_cfg, total=total_columns)
            logger.info(
                "MC plan: %d frontier + %d propagation across %d strata",
                len(mc_plan.frontier_columns),
                len(mc_plan.propagation_columns),
                len(mc_plan.strata),
            )

        # ── LLM SWEEP ────────────────────────────────────────────
        # When MC is active, sweep only frontier columns
        sweep_columns = (
            list(mc_plan.frontier_columns)
            if mc_plan and not mc_plan.is_passthrough
            else column_names
        )

        fsm.advance(run_id, FSMState.LLM_SWEEP, progress={
            "columns_total": total_columns,
            "phase": "llm_sweep",
            "mc_frontier": len(sweep_columns),
        })

        state = BootstrapState()
        if mc_plan and not mc_plan.is_passthrough:
            state.mc_strata_count = len(mc_plan.strata)
            state.mc_sample_fraction = mc_plan.effective_sample_fraction

        # Register the live state with nautilus so it can observe
        # batch_audit + request cooperative cancellation.  Unregistered
        # in the outer ``finally`` regardless of outcome.
        try:
            from atelier.overwatch.nautilus import register_state as _nautilus_register
            _nautilus_register(run_id, state)
        except Exception as exc:
            logger.debug("nautilus registration skipped: %s", exc)

        # Heartbeat: every batch completion advances FSM.updated_at so
        # operators and watchdogs can distinguish a running sweep from
        # one whose thread has died silently (observed in the wild:
        # Bedrock TCP connection hung with no timeout, gateway thread
        # entered LLM_SWEEP and never emitted another progress update).
        #
        # The raw heartbeat dict from bootstrap._llm_sweep uses
        # ``columns_labeled`` / ``llm_calls_total``; the UI (Status.tsx)
        # expects ``llm_labeled`` / ``llm_calls``.  Remap here so the
        # Status card's existing fields light up during the sweep
        # instead of freezing on the pre-sweep values.  Additional
        # ``sweep_*`` fields surface sub-phase detail (batches, elapsed,
        # batch size, truncations, failures) the operator needs to tell
        # a running sweep apart from a stalled one.
        def _sweep_progress(p: dict) -> None:
            try:
                sweep_started = state.sweep_started_at or time.time()
                elapsed_s = round(time.time() - sweep_started, 1)
                fsm.advance(run_id, FSMState.LLM_SWEEP, progress={
                    "columns_total": total_columns,
                    "mc_frontier": len(sweep_columns),
                    "llm_labeled": p.get("columns_labeled", 0),
                    "llm_calls": p.get("llm_calls_total", 0),
                    "sweep_phase": p.get("phase"),
                    "sweep_batches": p.get("batches_attempted", 0),
                    "sweep_truncations": p.get("truncation_count", 0),
                    "sweep_failed": p.get("failed_columns", 0),
                    "sweep_elapsed_s": elapsed_s,
                    "sweep_batch_size": state.effective_batch_size,
                })
            except Exception:
                pass  # never let progress reporting abort the sweep

        _llm_sweep(
            state, boot_cfg, llm_backend, system_prompt,
            sweep_columns, samples_by_name, column_table,
            category_count=len(category_set.categories),
            progress_callback=_sweep_progress,
        )

        # ── Label Propagation ──────────────────────────────────────
        if mc_plan and not mc_plan.is_passthrough:
            from atelier.classify.monte_carlo import propagate_labels as _propagate
            _propagate(state, mc_plan, samples_by_name, mc_cfg)

        coverage = _coverage(state, column_names)
        logger.info(
            "LLM sweep: labeled %d/%d (coverage=%.1f%%, calls=%d, propagated=%d)",
            len(state.labels), total_columns,
            coverage * 100, state.llm_calls_total, state.propagated_count,
        )

        # ── Fit-to-LLM CatBoost ──────────────────────────────────
        # When enabled, train an in-memory CatBoost on the (embedding_text,
        # llm_code) pairs we just produced and install it so downstream
        # evidence fusion + SHAP/SAGE attribute against the model that
        # agrees with the LLM by construction.  Replaces the pre-trained
        # classify_catboost_model_path for the rest of this run only.
        if getattr(cfg, "classify_catboost_fit_to_llm", False):
            try:
                _install_fit_to_llm_catboost(
                    cfg, state, samples_by_name, category_set,
                    save_path=results_dir / "catboost_fit_to_llm.cbm",
                )
            except Exception as exc:
                logger.warning("fit_to_llm install failed (non-fatal): %s", exc)

        # ── VALIDATING ───────────────────────────────────────────
        fsm.advance(run_id, FSMState.VALIDATING, progress={
            "phase": "ml_validation",
            "llm_labeled": len(state.labels),
            "coverage": round(coverage, 4),
        })

        # MC-aware discount: propagated labels get higher discount
        prop_discount = mc_cfg.propagation_discount if mc_plan and not mc_plan.is_passthrough else None

        _run_ml_validation(
            state, boot_cfg, column_names, samples_by_name,
            category_set, frame, has_embeddings, discounts=discounts,
            propagation_discount=prop_discount,
        )

        disagreements = _identify_disagreements(state, column_names, boot_cfg)
        mean_k = _mean_k(state, column_names)
        mean_gap = _mean_gap(state, column_names)

        logger.info(
            "ML validation: mean_gap=%.3f, mean_K=%.3f, disagreements=%d",
            mean_gap, mean_k, len(disagreements),
        )

        # ── TARGETED REVISIT LOOP ────────────────────────────────
        # Record iteration-0 metrics from initial ML validation.
        # No revisits at iteration 0; pass an empty revisited set.
        record_iteration_metrics(
            state, column_names, len(disagreements), boot_cfg,
            revisited_this_iter=set(),
        )

        # Convergence reason carried through both loop flavours and into
        # the final run summary.  Surfaces "how" the run ended to the UI
        # and to overwatch analysis — a green CONVERGED chip that hides
        # "no_revisit_candidates at iteration 1" is a silent failure.
        convergence_reason: str | None = None
        # Free-form prose explanation populated by the agent loop's
        # declare_converged tool; rendered as the tooltip body in the
        # Status UI so operators see the agent's reasoning alongside
        # the structured tag.  None on programmatic-loop runs.
        convergence_reason_detail: str | None = None

        # Agent-driven convergence (when configured and credentials available)
        if cfg.classify_agent_enabled and (cfg.has_anthropic or cfg.has_bedrock):
            from atelier.classify.agent_loop import run_agent_loop
            logger.info("Using agent-driven convergence loop")
            fsm.advance(run_id, FSMState.LLM_SWEEP, progress={
                "phase": "agent_convergence",
                "mean_k": round(mean_k, 4),
                **_convergence_progress(
                    state, column_names, disagreements, boot_cfg,
                ),
            })
            run_agent_loop(
                state, cfg, boot_cfg, llm_backend, system_prompt,
                column_names, samples_by_name, column_table,
                category_set, frame, has_embeddings, discounts,
            )
            # Post-agent fallback: if the agent declared convergence
            # before the min_iterations directive was satisfied (the
            # tool-side gate is the primary defense; this is the
            # secondary), run a programmatic revisit pass over the
            # broader uncertain-columns set.  Same machinery the
            # programmatic loop uses; ensures the directive holds
            # regardless of which loop drove convergence.
            if state.iteration < boot_cfg.min_iterations:
                logger.warning(
                    "Agent declared convergence at iteration=%d but "
                    "min_iterations=%d — running programmatic fallback "
                    "revisit pass to honor the project directive.",
                    state.iteration, boot_cfg.min_iterations,
                )
                fallback_candidates: list[str] = list(disagreements)
                fb_uncertain = _identify_uncertain_columns(
                    state, column_names, boot_cfg,
                )
                seen = set(fallback_candidates)
                for n in fb_uncertain:
                    if n not in seen:
                        fallback_candidates.append(n)
                        seen.add(n)
                while (
                    state.iteration < boot_cfg.min_iterations
                    and fallback_candidates
                ):
                    state.iteration += 1
                    try:
                        _llm_revisit(
                            state, boot_cfg, llm_backend, system_prompt,
                            fallback_candidates, samples_by_name,
                            column_table, category_set,
                        )
                    except FatalLLMError:
                        raise
                    _run_ml_validation(
                        state, boot_cfg, column_names, samples_by_name,
                        category_set, frame, has_embeddings,
                        discounts=discounts,
                    )
                    fallback_candidates = list(_identify_uncertain_columns(
                        state, column_names, boot_cfg,
                    ))
            # Carry the agent's structured tag (if it picked one) and
            # its prose reason as a separate detail field.  Tag drives
            # Status UI rendering; prose lands in the tooltip body.
            convergence_reason = (
                state.agent_converged_tag
                or ("agent_convergence" if state.agent_converged_reason else None)
            )
            convergence_reason_detail = state.agent_converged_reason
        else:
            # Programmatic convergence loop (default).
            #
            # The revisit candidate set is the UNION of two criteria:
            #  (a) K-based disagreements — LLM and ML emit different codes
            #      with enough source conflict to flag.
            #  (b) belief-gap uncertainty — even when LLM and ML coincide,
            #      a column with ``bel < bel_floor`` or ``gap > threshold``
            #      is a weakly-supported prediction that deserves a
            #      second look with enriched context.
            #
            # Pre-``min_iterations`` iterations cannot exit on the empty-
            # revisit-set branch: the algorithm's published accuracy
            # numbers rely on at least one revisit pass actually running.
            # Mirrors the ``max_iterations >= 2`` directive in 0c0170f.
            for iteration in range(1, boot_cfg.max_iterations + 1):
                revisit_candidates = list(disagreements)
                uncertain = _identify_uncertain_columns(state, column_names, boot_cfg)
                # Dedupe while preserving disagreement ordering (highest K first).
                seen = set(revisit_candidates)
                for name in uncertain:
                    if name not in seen:
                        revisit_candidates.append(name)
                        seen.add(name)

                if not revisit_candidates and iteration > boot_cfg.min_iterations:
                    convergence_reason = "no_revisit_candidates"
                    logger.info(
                        "No revisit candidates after min_iterations — converged",
                    )
                    break

                if state.llm_calls_total >= boot_cfg.max_total_llm_calls:
                    convergence_reason = "budget_exhausted"
                    logger.info("Budget exhausted (%d calls)", state.llm_calls_total)
                    break

                if (
                    mean_gap < boot_cfg.gap_threshold
                    and iteration > boot_cfg.min_iterations
                ):
                    convergence_reason = "gap_threshold_met"
                    logger.info(
                        "Mean belief gap=%.3f < threshold=%.3f — converged",
                        mean_gap, boot_cfg.gap_threshold,
                    )
                    break

                # Early termination: belief gap no longer decreasing (plateau).
                # Still honored AFTER the min_iterations floor so a truly
                # plateaued run stops, but an iter-1 coincidence can't
                # trigger it.
                if should_stop_early(state) and iteration > boot_cfg.min_iterations:
                    convergence_reason = "plateau"
                    logger.info(
                        "Belief gap not decreasing for 2 iterations — "
                        "early stop (mean_gap=%.3f)",
                        mean_gap,
                    )
                    break

                # Use the broadened set for the revisit, but preserve the
                # narrow ``disagreements`` variable for existing call sites
                # (Row MC rotation, telemetry, etc.).
                disagreements = revisit_candidates

                state.iteration = iteration

                # Row MC: rotate values for disagreement columns
                if row_mc_cfg.enabled and disagreements:
                    from atelier.classify.row_sampler import select_row_sample
                    rotated = 0
                    for name in disagreements:
                        col = samples_by_name[name]
                        if col.all_values and len(col.all_values) > row_mc_cfg.k:
                            select_row_sample(
                                col, k=row_mc_cfg.k,
                                strategy=row_mc_cfg.strategy,
                                iteration=iteration,
                            )
                            rotated += 1
                    if rotated:
                        logger.info(
                            "Row MC: rotated values for %d/%d disagreement columns "
                            "(k=%d, strategy=%s, iteration=%d)",
                            rotated, len(disagreements),
                            row_mc_cfg.k, row_mc_cfg.strategy, iteration,
                        )

                fsm.advance(run_id, FSMState.LLM_SWEEP, progress={
                    "phase": "revisit",
                    "iteration": iteration,
                    "mean_k": round(mean_k, 4),
                    **_convergence_progress(
                        state, column_names, disagreements, boot_cfg,
                    ),
                })

                # Snapshot the columns that will be revisited THIS
                # iteration so the per-column trajectory append (in
                # record_iteration_metrics below) can mark them
                # ``revisited=True``.  Captured before _llm_revisit
                # mutates state.labels and before disagreements is
                # re-computed for the next iteration.
                revisited_this_iter: set[str] = set(disagreements)

                _llm_revisit(
                    state, boot_cfg, llm_backend, system_prompt,
                    disagreements, samples_by_name, column_table, category_set,
                )

                fsm.advance(run_id, FSMState.VALIDATING, progress={
                    "phase": "revalidation",
                    "iteration": iteration,
                    "llm_calls": state.llm_calls_total,
                })

                _run_ml_validation(
                    state, boot_cfg, column_names, samples_by_name,
                    category_set, frame, has_embeddings, discounts=discounts,
                    propagation_discount=prop_discount,
                )

                # Row MC: record label history for stability tracking
                if row_mc_cfg.enabled:
                    for name in disagreements:
                        if name in state.labels:
                            state.row_labels_history.setdefault(name, []).append(
                                state.labels[name]
                            )

                disagreements = _identify_disagreements(state, column_names, boot_cfg)
                mean_k = _mean_k(state, column_names)
                mean_gap = _mean_gap(state, column_names)
                coverage = _coverage(state, column_names)

                # Row MC: escalate row-unstable columns to full reservoir
                if row_mc_cfg.enabled and row_mc_cfg.adaptive_escalation:
                    from atelier.classify.bootstrap import row_stability as _row_stab
                    escalated = 0
                    for name in list(disagreements):
                        if _row_stab(state, name) < 0.5:
                            col = samples_by_name[name]
                            if col.all_values:
                                col.values = list(col.all_values)
                                escalated += 1
                    if escalated:
                        state.escalated_count += escalated
                        logger.info(
                            "Row MC: escalated %d row-unstable columns to full reservoir",
                            escalated,
                        )

                record_iteration_metrics(
                    state, column_names, len(disagreements), boot_cfg,
                    revisited_this_iter=revisited_this_iter,
                )

            # Loop exited without hitting one of the named break paths —
            # we ran the full max_iterations budget.  Flag that honestly
            # so the UI can show it rather than claiming belief-gap
            # convergence we didn't actually achieve.
            if convergence_reason is None:
                convergence_reason = "max_iterations_reached"

        # ── FINAL CLASSIFICATION PASS ────────────────────────────
        coverage = _coverage(state, column_names)
        mean_k = _mean_k(state, column_names)
        # Convergence uses the belief-gap criterion (primary convergence
        # measure per docs + BootstrapConfig comments).  The previous
        # ``mean_k < k_threshold`` check was Yager-oriented — under the
        # default Dempster fusion, K is normalized out so mean_k stays
        # high (~0.85 in live runs), making the flag permanently false.
        # Overwatch correctly flagged the resulting contradiction
        # between summary.converged=false and FSM state CONVERGED.
        mean_gap = _mean_gap(state, column_names)
        converged = (
            coverage >= boot_cfg.coverage_target
            and mean_gap < boot_cfg.gap_threshold
        )
        # If the loop path never assigned a reason but the run converged
        # by the coverage+gap criteria above, flag it explicitly rather
        # than letting the absence be interpreted as legitimate
        # convergence.
        if convergence_reason is None:
            convergence_reason = (
                "coverage_and_gap_met" if converged else "unknown"
            )

        fsm.advance(run_id, FSMState.CLASSIFYING, progress={
            "phase": "final_classification",
            "converged": converged,
            "mean_k": round(mean_k, 4),
            "coverage": round(coverage, 4),
            **_convergence_progress(
                state, column_names, disagreements, boot_cfg,
            ),
        })

        classifications: list[dict[str, Any]] = []
        for col in all_columns:
            llm_code = state.labels.get(col.name)
            llm_conf = state.confidence.get(col.name, 0.0)
            result = _classify_column(
                col, category_set, frame,
                llm_code=llm_code,
                llm_confidence=llm_conf,
                llm_discount=boot_cfg.llm_discount,
                use_cosine=has_embeddings,
                discounts=discounts,
                fusion_strategy=cfg.classify_fusion_strategy,
                svm_alignment=svm_alignment,
            )
            classifications.append(result)

        fsm.advance(run_id, FSMState.FUSING, progress={
            "columns_classified": len(classifications),
        })

        # ── Feature analysis (SHAP + SAGE, config-gated) ──────────
        _run_feature_analysis(cfg, classifications, all_samples, category_set, results_dir, mc_plan=mc_plan)

        # ── Cautious-code review — agent-mediated backoff ────────
        # Runs between FUSING and EVALUATING so accuracy numbers reflect
        # post-review predictions.  SHAP/SAGE attribute to features-in-
        # general (not per-column codes), so order doesn't disturb them.
        # On by default — see classify.cautious_review.enabled.
        cautious_audit: dict = {"enabled": False}
        if getattr(cfg, "classify_cautious_review_enabled", True):
            from atelier.classify.cautious_review import review_classifications
            def _cautious_progress(p: dict) -> None:
                try:
                    fsm.advance(run_id, FSMState.FUSING, progress={
                        "phase": "cautious_review",
                        **p,
                    })
                except Exception:
                    pass
            try:
                cautious_audit = review_classifications(
                    classifications, cfg,
                    category_set=category_set,
                    progress_callback=_cautious_progress,
                )
                # Persist audit beside other artifacts.
                (results_dir / "cautious_review.json").write_text(
                    json.dumps(cautious_audit, indent=2, default=str) + "\n",
                )
            except Exception as exc:
                logger.warning(
                    "Cautious-code review failed (non-fatal, keeping pre-review predictions): %s",
                    exc,
                )
                cautious_audit = {"enabled": True, "error": str(exc)}

        # ── EVALUATING ───────────────────────────────────────────
        fsm.advance(run_id, FSMState.EVALUATING, progress={
            "columns_fused": len(classifications),
        })

        summary = _evaluate_results(classifications)
        eval_report = evaluate_classifications(
            classifications, category_set, run_id=run_id,
        )
        from atelier.classify.evaluation import epistemic_evaluation
        epistemic = epistemic_evaluation(classifications, category_set)
        summary["converged"] = converged
        summary["convergence_reason"] = convergence_reason
        summary["convergence_reason_detail"] = convergence_reason_detail
        summary["cautious_review"] = {
            k: v for k, v in cautious_audit.items() if k != "decisions"
        }
        summary["epistemic_evaluation"] = epistemic
        from atelier.classify.bootstrap import k_convergence_rate
        summary["bootstrap_iterations"] = state.iteration
        summary["llm_calls"] = state.llm_calls_total
        summary["tokens_input"] = state.tokens_input
        summary["tokens_output"] = state.tokens_output
        summary["mean_k"] = round(mean_k, 4)
        summary["bootstrap_coverage"] = round(coverage, 4)
        summary["k_convergence_rate"] = round(k_convergence_rate(state), 4)
        summary["agent_turns"] = state.agent_turns
        summary["agent_converged_reason"] = state.agent_converged_reason
        summary["agent_reasoning"] = state.agent_reasoning
        # Monte Carlo sampling metadata
        summary["mc_enabled"] = mc_plan is not None and not mc_plan.is_passthrough
        summary["mc_strata"] = len(mc_plan.strata) if mc_plan else 0
        summary["mc_frontier_columns"] = (
            len(mc_plan.frontier_columns) if mc_plan else total_columns
        )
        summary["mc_propagated_columns"] = state.propagated_count
        summary["mc_escalated_columns"] = state.escalated_count
        summary["mc_sample_fraction"] = (
            mc_plan.effective_sample_fraction if mc_plan else 1.0
        )
        summary["iteration_metrics"] = [
            {
                "iteration": m.iteration,
                "mean_k": m.mean_k,
                "max_k": m.max_k,
                "disagreements": m.disagreements,
                "coverage": m.coverage,
                "llm_calls": m.llm_calls,
                # Belief-gap convergence (added 2026-05-03 — see plan
                # how-about-extend-the-golden-sedgewick.md).  Time-major
                # trajectory needed for thesis-defense plots; the older
                # column_trajectories.json is column-major and doesn't
                # substitute.
                "mean_gap": m.mean_gap,
                "mean_bel": m.mean_bel,
                "frac_unclear": m.frac_unclear,
                "gap_contraction_rate": m.gap_contraction_rate,
                "indep_tier_disagreement_frac": m.indep_tier_disagreement_frac,
                # Composite scalar retained for back-compat — see
                # ``residual_norm`` docstring for why it isn't promoted
                # to the live dashboard.
                "residual_norm": m.residual_norm,
                "contraction_rate": m.contraction_rate,
            }
            for m in state.iteration_metrics
        ]

        # Write results
        results_path = results_dir / "classifications.json"
        results_path.write_text(json.dumps(classifications, indent=2, default=str) + "\n")
        eval_report.write_json(results_dir / "evaluation_report.json")

        # Per-column residual trajectories — column-major view of the
        # bootstrap loop's convergence behaviour, complementary to the
        # time-major iteration_history in classifications.  Used for
        # offline analysis (Phase B/C acceleration backtest), operator
        # post-mortem on stuck columns, and audit.  See
        # docs/src/architecture/dst-evidence-independence.md.
        from dataclasses import asdict as _dc_asdict
        trajectories_path = results_dir / "column_trajectories.json"
        trajectories_payload = {
            name: [_dc_asdict(snap) for snap in hist]
            for name, hist in state.column_history.items()
        }
        trajectories_path.write_text(
            json.dumps(trajectories_payload, indent=2, default=str) + "\n",
        )

        parquet_path = _write_parquet(classifications, results_dir / "atelier_embeddings.parquet")

        # Order matters here: the ``datasets`` table has a FK on
        # ``artifact_set_id`` referencing ``ml_artifact_sets(id)``, so the
        # artifact set must be registered BEFORE the dataset row tries to
        # reference it.  Pre-2026-05-04 these blocks were inverted, which
        # produced a silent FK violation on every classify run that
        # trained ML artifacts (every run with f > 0).  See run b7e10711
        # for the captured traceback in ``register_error.json``.

        # 1. Register the ML artifact set so the dataset row's FK has a
        #    target.  Non-fatal — a failure here means Extend is
        #    unavailable for this run until backfilled via
        #    scripts/backfill_dataset.py, AND the dataset row that
        #    follows will fall back to ``artifact_set_id=None``.
        artifact_set_registered = False
        try:
            from atelier.classify.artifact_set import build_artifact_set_record
            from atelier.db.dao import AtelierDao
            spec = build_artifact_set_record(
                run_id=run_id,
                results_dir=results_dir,
                cfg=cfg,
                n_columns=len(classifications),
                source_id=source_id,
                fsm_run_id=run_id,
            )
            if spec is not None:
                AtelierDao().register_artifact_set(**spec)
                artifact_set_registered = True
                logger.info("Registered ML artifact set: %s", run_id)
        except Exception as e:
            logger.exception("Failed to register ML artifact set for run %s", run_id)
            _write_register_error(
                results_dir, "artifact_set", run_id, source_id, e,
            )

        # 2. Auto-register as a dataset so the Embeddings page is
        #    populated.  When the artifact set wasn't registered (no
        #    spec, or registration raised), pass ``artifact_set_id=None``
        #    so the dataset row still lands — the Embeddings panel
        #    surfaces the run even when Extend wiring is incomplete.
        #    Any other failure here leaves a ``register_error.json``
        #    sidecar so an operator can spot the orphan run and
        #    backfill via scripts/backfill_dataset.py.
        if parquet_path:
            try:
                from atelier.db.dao import AtelierDao
                dao = AtelierDao()
                version_number = 1
                if source_id:
                    version_number = dao.next_version_number(source_id)
                dao.upsert_dataset(
                    dataset_id=run_id,
                    name=f"Classification {run_id[:8]}",
                    parquet_path=str(parquet_path),
                    description=f"{len(classifications)} columns classified",
                    row_count=len(classifications),
                    source_id=source_id,
                    version_number=version_number,
                    is_active=True,
                    summary=f"{len(all_samples)} tables, {len(classifications)} columns",
                    fsm_run_id=run_id,
                    artifact_set_id=run_id if artifact_set_registered else None,
                    parent_dataset_id=None,    # classify runs have no parent
                    run_kind="classify",
                )
                if source_id:
                    dao.set_active_version(source_id, run_id)
            except Exception as e:
                logger.exception("Failed to register dataset for run %s", run_id)
                _write_register_error(
                    results_dir, "dataset", run_id, source_id, e,
                )

        # ── Governance sync (Atlas) ──────────────────────────────────
        # When auto_sync is enabled and Atlas is configured, push the
        # taxonomy as classification types and tag entities with results.
        governance_summary: dict = {}
        try:
            if cfg.governance_auto_sync and cfg.has_atlas:
                from atelier.governance.client import GovernanceClient
                from atelier.governance.sync import (
                    TaxonomyNode, sync_taxonomy_to_atlas,
                    ColumnClassification as GovColumnClassification,
                    sync_classifications_to_atlas,
                )
                gc = GovernanceClient.from_atelier_config(cfg)
                dry = cfg.governance_dry_run

                # 1. Sync taxonomy → Atlas classification types
                nodes = [
                    TaxonomyNode(
                        code=cat.code, label=cat.label,
                        notation=getattr(cat, "notation", ""),
                        parent_code=getattr(cat, "parent_code", "") or "",
                    )
                    for cat in category_set.all_categories
                ]
                nodes.sort(key=lambda n: n.code.count("."))
                tax_report = sync_taxonomy_to_atlas(gc.atlas, nodes, dry_run=dry)
                governance_summary["taxonomy"] = {
                    "created": len(tax_report.created),
                    "skipped": len(tax_report.skipped),
                    "failed": len(tax_report.failed),
                }

                # 2. Tag entities with classification results
                by_table: dict[str, list] = {}
                for c in classifications:
                    table = c.get("table_name", "unknown")
                    by_table.setdefault(table, []).append(
                        GovColumnClassification(
                            column_name=c["column_name"],
                            tags=[c.get("predicted_code", "")],
                            confidence=float(c.get("confidence", 0) or 0),
                            reason=c.get("evidence", ""),
                        )
                    )
                tag_success = 0
                tag_errors = 0
                for tbl, cols in by_table.items():
                    results = sync_classifications_to_atlas(
                        gc.atlas, tbl, cols,
                        cluster_name=cfg.governance_cluster_name,
                        dry_run=dry,
                    )
                    tag_success += sum(1 for r in results if r.status in ("success", "dry_run"))
                    tag_errors += sum(1 for r in results if r.status == "error")
                governance_summary["tagging"] = {
                    "success": tag_success, "errors": tag_errors,
                    "tables": len(by_table), "dry_run": dry,
                }
                logger.info(
                    "Governance sync: taxonomy=%d created, tagging=%d/%d success/errors%s",
                    len(tax_report.created), tag_success, tag_errors,
                    " (dry run)" if dry else "",
                )
        except Exception as e:
            logger.warning("Governance sync failed (non-fatal): %s", e)
            governance_summary["error"] = str(e)

        # ── Overwatch analysis ────────────────────────────────────────
        # When overwatch is enabled, run a single-turn analysis of the
        # pipeline results and write recommendations to overwatch.md.
        overwatch_path = None
        try:
            if cfg.has_overwatch:
                from atelier.overwatch.agent import run_overwatch_analysis
                overwatch_path = run_overwatch_analysis(
                    cfg, run_id, summary, results_dir,
                )
        except Exception as e:
            logger.warning("Overwatch analysis failed (non-fatal): %s", e)

        # ── Focus computation ─────────────────────────────────────────
        # Hybrid focus: deterministic drift-from-default rules unioned
        # with the fenced ``focus`` JSON block overwatch may emit.
        # Always runs — overwatch-absence degrades gracefully.
        try:
            from atelier.classify.focus import compute_focus
            compute_focus(run_id, results_dir)
        except Exception as e:
            logger.warning("Focus computation failed (non-fatal): %s", e)

        fsm.advance(run_id, FSMState.CONVERGED, progress={
            **summary,
            "result_path": str(results_path),
            "parquet_path": str(parquet_path) if parquet_path else None,
            "governance": governance_summary or None,
            "overwatch": str(overwatch_path) if overwatch_path else None,
        }, result_path=str(parquet_path) if parquet_path else str(results_path))

        return {
            "run_id": run_id,
            "state": "CONVERGED",
            "classifications": len(classifications),
            "result_path": str(results_path),
            "parquet_path": str(parquet_path) if parquet_path else None,
            "evaluation_report": eval_report.to_dict(),
            "overwatch_report": str(overwatch_path) if overwatch_path else None,
            **summary,
        }

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        try:
            fsm.advance(run_id, FSMState.ERROR, error=str(exc))
        except ValueError:
            pass
        return {
            "run_id": run_id,
            "state": "ERROR",
            "error": str(exc),
        }
    finally:
        # Always tear down the nautilus registration so the watcher
        # thread stops observing a state object the pipeline no longer
        # owns.  Safe to call even if registration never happened.
        try:
            from atelier.overwatch.nautilus import (
                unregister_state as _nautilus_unregister,
            )
            _nautilus_unregister(run_id)
        except Exception:
            pass


_HIVE_DB_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_hive_vocab_uri(vocab_uri: str) -> tuple[str, str]:
    """Parse a hive-style ``vocab_uri`` into ``(database, table)``.

    The only accepted shape is ``{db}.annotations`` where ``db`` is a
    Hive identifier (alnum + underscore, leading letter).  All other
    shapes raise ``ValueError`` so the calling pipeline surfaces a real
    cause to the operator instead of silently routing to the wrong
    database — historically this manifested as runs falling back to the
    16-leaf universal vocabulary while the user's selection was ignored.
    """
    if not vocab_uri or not vocab_uri.strip():
        raise ValueError(
            "vocab_uri is empty; expected \"{db}.annotations\""
        )
    parts = vocab_uri.split(".")
    if len(parts) != 2 or not parts[0]:
        raise ValueError(
            f"vocab_uri={vocab_uri!r} does not match expected "
            f"\"{{db}}.annotations\" form"
        )
    database, table = parts
    if table != "annotations":
        raise ValueError(
            f"vocab_uri={vocab_uri!r}: table must be 'annotations', "
            f"got {table!r}"
        )
    if not _HIVE_DB_IDENT_RE.match(database):
        raise ValueError(
            f"vocab_uri={vocab_uri!r}: unsafe database identifier "
            f"{database!r} (expected alnum + underscore, leading letter)"
        )
    return database, table


def _load_vocabulary(
    cfg,
    build_dir: Path,
    connection_name,
    vocab_uri: str | None = None,
    database: str | None = None,
):
    """Load vocabulary for classification.

    Vocabulary routing by source type:

    - **OOTB sample**: 316-leaf ICE from ``data/sample/ontology.json``
      (handled by caller via ``load_sample_vocabulary``).
    - **Hive/synth with annotations**: Domain vocab loaded directly from
      the annotations table specified by *vocab_uri*.  The customer's
      domain codes are the classification targets — the LLM reads labels
      and descriptions and classifies into hierarchical dot-codes.
    - **Env-default Hive**: when *vocab_uri* is empty but
      *connection_name* and *database* are both set (via
      ``ATELIER_CLASSIFY_CONNECTION`` + ``ATELIER_CLASSIFY_DATABASE``),
      try ``{database}.annotations`` via ``load_annotations_from_hive``
      before falling through.  This is the auto-classification-at-deploy
      path that matches the env-seeded ``data_source`` row.
    - **Fallback**: 16-leaf universal (only when no domain annotations).

    Hive sources always require annotations; the annotations table
    location (``vocab_uri``) is configured per data source, decoupled
    from the data tables being classified.
    """
    log = logging.getLogger(__name__)

    if vocab_uri:
        # Scheme-aware dispatch: file:// reads annotations.csv from a
        # local mount; everything else falls through to the Hive / cache
        # path.  Add s3:// / jdbc:// branches here as they come online.
        if vocab_uri.startswith("file://"):
            from atelier.classify.taxonomy import load_annotations_from_filesystem
            fs_path = Path(vocab_uri[len("file://"):]).expanduser().resolve()
            try:
                domain_cs = load_annotations_from_filesystem(fs_path)
                if not isinstance(domain_cs, HierarchicalCategorySet):
                    domain_cs = HierarchicalCategorySet(
                        name=domain_cs.name,
                        categories=list(domain_cs.categories),
                    )
                log.info(
                    "Loaded filesystem vocabulary: %d leaves (vocab_uri=%s)",
                    len(domain_cs.categories), vocab_uri,
                )
                return domain_cs
            except FileNotFoundError:
                raise RuntimeError(
                    f"Filesystem vocab_uri={vocab_uri!r} points at a missing file"
                )

        try:
            db_from_uri, _ = _parse_hive_vocab_uri(vocab_uri)
            domain_cs = _load_domain_annotations(
                cfg, build_dir, connection_name, database=db_from_uri,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load domain annotations from vocab_uri={vocab_uri!r} "
                f"via connection {connection_name!r}"
            ) from exc
        if domain_cs is None or len(domain_cs.categories) == 0:
            raise RuntimeError(
                f"Domain annotations from vocab_uri={vocab_uri!r} returned "
                f"0 categories via connection {connection_name!r}"
            )
        if not isinstance(domain_cs, HierarchicalCategorySet):
            domain_cs = HierarchicalCategorySet(
                name=domain_cs.name,
                categories=list(domain_cs.categories),
            )
        log.info(
            "Loaded domain vocabulary: %d leaf categories (vocab_uri=%s)",
            len(domain_cs.categories), vocab_uri,
        )
        return domain_cs

    # Env-default Hive: when the operator set ATELIER_CLASSIFY_CONNECTION
    # + ATELIER_CLASSIFY_DATABASE but no explicit vocab_uri was threaded
    # through, try {database}.annotations on that connection before
    # falling through to the universal fixture.  Matches the stable
    # data_source seeded at startup when the env vars are present.
    if connection_name and database:
        try:
            from atelier.classify.taxonomy import load_annotations_from_hive
            domain_cs = load_annotations_from_hive(
                cfg, connection_name, database, hierarchical=True,
            )
            if domain_cs is not None and len(domain_cs.categories) > 0:
                if not isinstance(domain_cs, HierarchicalCategorySet):
                    domain_cs = HierarchicalCategorySet(
                        name=domain_cs.name,
                        categories=list(domain_cs.categories),
                    )
                log.info(
                    "Loaded env-default vocabulary: %d leaf categories "
                    "(%s.annotations via %s)",
                    len(domain_cs.categories), database, connection_name,
                )
                return domain_cs
        except Exception as exc:
            log.warning(
                "Env-default Hive vocab load failed (%s.annotations via %s): %s. "
                "Falling through to universal fixture.",
                database, connection_name, exc,
            )

    # Fallback: universal vocabulary (16 BFO-grounded leaves)
    universal = load_universal_vocabulary(hierarchical=True)
    log.info("Loaded universal vocabulary: %d terms", len(universal.categories))
    return universal


def _load_domain_annotations(
    cfg,
    build_dir: Path,
    connection_name,
    *,
    database: str,
):
    """Load domain-specific annotations from cache or hive.

    Returns a CategorySet of domain terms, or ``None`` if hive returns an
    empty annotations table.  Hive load failures are *not* caught here —
    callers wrap them with chained context (the user-facing vocab_uri),
    so the operator sees the real cause instead of a silent fallback.

    Cache is keyed by ``{connection_name}__{database}.json`` to prevent
    cross-source poisoning when an operator switches between data
    sources within the same project ``build/`` tree.
    """
    log = logging.getLogger(__name__)
    cache_dir = build_dir / "data" / "annotations"
    cache_path = cache_dir / f"{connection_name}__{database}.json"

    if cache_path.exists():
        cs = load_annotations_from_json(cache_path, hierarchical=True)
        if len(cs.categories) > 0:
            log.info(
                "Loaded %d domain categories from cache %s",
                len(cs.categories), cache_path,
            )
            return cs
        log.warning(
            "Cache %s contains 0 categories — treating as corrupt, will re-fetch",
            cache_path,
        )
        cache_path.unlink()

    cs = load_annotations_from_hive(cfg, connection_name, database)
    if len(cs.categories) == 0:
        log.warning(
            "Hive returned 0 domain categories from %s.annotations via %s "
            "— skipping domain layer",
            database, connection_name,
        )
        return None
    log.info(
        "Loaded %d domain categories from hive (%s.annotations via %s)",
        len(cs.categories), database, connection_name,
    )
    save_annotations_json(cs, cache_path)
    return cs


# ── Confusable pairs ──────────────────────────────────────────────
#
# Known category pairs that are structurally similar and commonly confused.
# When DST evidence is split between two members of a pair, mass is
# redistributed to a compound focal element instead of forcing a
# singleton decision — allowing honest representation of ambiguity.

_CONFUSABLE_PAIR_CODES: list[tuple[str, str]] = [
    ("ICE.METADATA.RECID", "ICE.SENSITIVE.TECHNICAL.DEVID"),
    ("ICE.METADATA.TIMESTAMP", "ICE.SENSITIVE.PID.IDENTITY.DOB"),
    ("ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.TXNAMT", "ICE.SENSITIVE.PID.FINANCIAL.ACCOUNT.BAN"),
    ("ICE.SENSITIVE.TECHNICAL.IPADDR", "ICE.SENSITIVE.TECHNICAL.DEVID"),
]


def _build_confusable_pairs(
    category_set: HierarchicalCategorySet,
) -> list[tuple[str, str]]:
    """Filter confusable pairs to those present in the loaded vocabulary."""
    leaf_codes = category_set.leaf_codes
    return [
        (a, b) for a, b in _CONFUSABLE_PAIR_CODES
        if a in leaf_codes and b in leaf_codes
    ]


# Sources whose evidence is genuinely independent of the LLM sweep.
# Used to compute a separate "independent-tier consensus" alongside
# the full fusion so the bootstrap revisit gate can detect when the
# LLM disagrees with the union of LLM-independent signals (cosine,
# pattern, name_match) — see Shafer 1976 §11.3 reliability discount
# and Denoeux 2008 on non-distinct evidence.
#
# CatBoost (``fit_to_llm``) is excluded because its labels are the
# LLM's labels by construction — it cannot tautologically contradict
# the source it was trained on.
#
# SVM is excluded as a conservative call: its features and training
# labels are independent of the LLM, but the per-vocabulary ICE→user-
# code alignment in ``classify.ontology_alignment`` does pass through
# the LLM at vocab-load time.  A contradicting SVM vote could in
# principle be confounded by an alignment error that the LLM would
# also commit on the live sweep.  Membership in this tier is meant
# to be the strict "no shared knowledge with the runtime LLM at all"
# set; SVM's weak vocab-level dependency keeps it out for now.
# Future work to admit SVM here cleanly: switch the alignment to a
# BM25 + transformer-reranker path that doesn't share an LLM with
# the runtime sweep — see ``ontology_alignment.py`` module docstring.
INDEPENDENT_TIER: frozenset[str] = frozenset({"cosine", "pattern", "name_match"})


def _resolved_pattern_map_for(category_set: HierarchicalCategorySet) -> dict[str, str]:
    """Return the pattern-target map resolved against *category_set*, cached."""
    cached = getattr(category_set, "_resolved_pattern_map", None)
    if cached is None:
        cached = resolve_pattern_map(DEFAULT_PATTERN_MAP, category_set)
        try:
            category_set._resolved_pattern_map = cached  # type: ignore[attr-defined]
        except AttributeError:
            return cached
    return cached


def _classify_column(
    col: ColumnSample,
    category_set: HierarchicalCategorySet,
    frame: FrameOfDiscernment,
    *,
    llm_code: str | None = None,
    llm_confidence: float = 0.0,
    llm_alternatives: list[dict] | None = None,
    llm_discount: float = 0.15,
    use_cosine: bool = True,
    discounts: DiscountConfig | None = None,
    fusion_strategy: str = "dempster",
    svm_alignment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Classify a single column using Dempster-Shafer evidence fusion.

    Evidence sources (up to 6): name matching, pattern detection,
    cosine similarity, LLM, CatBoost, SVM.  The pipeline always
    supplies LLM evidence; llm_code may be None only for offline
    use cases such as seed data preparation.

    In addition to the full Dempster fusion, computes an
    *independent-tier consensus* over ``{cosine, pattern, name_match}``
    — the subset of sources that does not derive from the LLM sweep —
    and exposes its top-1 singleton in the result dict.  The bootstrap
    revisit gate at ``_identify_disagreements`` consults this signal so
    cosine/pattern disagreement with LLM can trigger a revisit even
    when the fully-fused prediction (which includes LLM mass) happens
    to match LLM.  See ``docs/src/architecture/dst-evidence-independence.md``.
    """
    if discounts is None:
        discounts = DiscountConfig()

    features = extract_features(
        column_name=col.name,
        column_type=col.column_type,
        values=col.values,
        siblings=col.siblings,
        source_table=col.table_name,
        total_count=col.total_count,
        null_count=col.null_count,
        distinct_count=col.distinct_count,
    )

    # Collect named evidence sources
    source_masses: dict[str, Any] = {}

    # 1. Name matching
    name_mass = name_match_to_mass(
        col.name, frame, category_set,
        exact_mass=discounts.name_match_exact,
        code_mass=discounts.name_match_code,
        alias_mass=discounts.name_match_alias,
        overlap_mass=discounts.name_match_overlap,
    )
    if not _is_vacuous(name_mass):
        source_masses["name_match"] = name_mass

    # 2. Pattern detection (target codes resolved against the active vocab
    # so non-ICE vocabularies don't silently disable the entire source).
    resolved_pattern_map = _resolved_pattern_map_for(category_set)
    pattern_mass = pattern_to_mass(
        features.pattern_signals, frame,
        pattern_category_map=resolved_pattern_map,
        theta_mass=discounts.pattern_theta,
    )
    if not _is_vacuous(pattern_mass):
        source_masses["pattern"] = pattern_mass

    # 3. Cosine similarity (if available)
    if use_cosine:
        try:
            from atelier.classify.embedding import classify_cosine as _cosine
            similarities = _cosine(features, category_set)
            cosine_mass = cosine_to_mass(
                similarities, frame, discount=discounts.cosine,
            )
            source_masses["cosine"] = cosine_mass
        except Exception as exc:
            logger.debug("Cosine similarity unavailable for %s: %s", col.name, exc)

    # 4. LLM evidence (always present in pipeline; absent only in offline seed prep)
    if llm_code:
        llm_mass_val = llm_to_mass(
            llm_code, llm_confidence,
            llm_alternatives or [],
            frame, discount=llm_discount,
        )
        if not _is_vacuous(llm_mass_val):
            source_masses["llm"] = llm_mass_val

    # 5. CatBoost (if model available)
    try:
        from atelier.classify.ml_inference import predict_catboost
        cb_result = predict_catboost(features, category_set)
        if cb_result:
            proba, variance = cb_result
            cb_mass = catboost_to_mass(
                proba, frame, variance,
                base_discount=discounts.catboost_base,
                variance_scale=discounts.catboost_variance_scale,
                max_discount=discounts.catboost_max,
                fallback_discount=discounts.catboost_fallback,
            )
            if not _is_vacuous(cb_mass):
                source_masses["catboost"] = cb_mass
    except Exception as exc:
        logger.debug("CatBoost unavailable for %s: %s", col.name, exc)

    # 6. SVM (if model available).
    # The synth-trained SVM emits ICE.* (bundled-ontology) codes; the
    # per-vocabulary ``svm_alignment`` translates them into the user's
    # taxonomy before svm_to_mass tests frame membership.  Without
    # alignment (or on OOTB-sample runs where user vocab IS ICE.*), the
    # translate_proba helper is an identity pass-through and the SVM
    # behavior is unchanged from the pre-refactor path.  See
    # ``ontology_alignment.py`` for the independence/discount caveats.
    try:
        from atelier.classify.ml_inference import predict_svm
        from atelier.classify.ontology_alignment import translate_proba
        svm_proba = predict_svm(features)
        if svm_proba:
            svm_proba = translate_proba(svm_proba, svm_alignment)
        if svm_proba:
            svm_mass = svm_to_mass(svm_proba, frame, discount=discounts.svm)
            if not _is_vacuous(svm_mass):
                source_masses["svm"] = svm_mass
    except Exception as exc:
        logger.debug("SVM unavailable for %s: %s", col.name, exc)

    # Fuse evidence via HierarchicalClassification
    if not source_masses:
        return _empty_classification(col, features)

    # Independent-tier consensus over LLM-independent sources only.
    # Used by the bootstrap revisit gate to detect "LLM disagrees with
    # the union of cosine/pattern/name_match" — a condition that the
    # fully-fused prediction can mask whenever LLM-derivative ML
    # sources amplify the LLM vote (Shafer 1976 §11.3, Denoeux 2008).
    indep_top1_code: str | None = None
    indep_top1_mass: float = 0.0
    indep_top1_conflict: float = 0.0
    indep_assignments = [
        source_masses[name] for name in INDEPENDENT_TIER if name in source_masses
    ]
    if indep_assignments:
        try:
            indep_combined, indep_top1_conflict = combine_multiple(
                indep_assignments, strategy="dempster",
            )
            indep_top = indep_combined.most_committed_singleton()
            if indep_top is not None:
                indep_top1_code, indep_top1_mass = indep_top
        except ValueError:
            # Total conflict (K=1) collapses Dempster — leave consensus
            # empty; the high-K branch of the revisit gate still fires.
            pass

    hc = HierarchicalClassification.from_combined_evidence(
        source_masses=source_masses,
        frame=frame,
        category_set=category_set,
        fusion_strategy=fusion_strategy,
    )

    best_code = hc.category.code
    bel, pl = hc.interval_at(best_code)
    belief_path = hc.belief_path()

    return {
        "table_name": col.table_name,
        "column_name": col.name,
        "column_type": col.column_type,
        "predicted_code": best_code,
        "predicted_label": hc.category.label,
        # Mnemonic / formal code straight from the annotations table
        # ("BAN", "EMAIL", "PAN").  Operators used to reviewing the
        # annotations table want this tag in the result alongside the
        # label — let both coexist so the UI can render either style.
        "predicted_annotation": getattr(hc.category, "abbrev", "") or "",
        "confidence": hc.confidence,
        "belief": round(bel, 4),
        "plausibility": round(pl, 4),
        "uncertainty": round(pl - bel, 4),
        "conflict": hc.conflict,
        "needs_clarification": hc.needs_clarification,
        "evidence": hc.evidence,
        "evidence_sources": {name: _mass_summary(ba) for name, ba in source_masses.items()},
        "embedding_text": features.to_embedding_text(),
        "pattern_signals": features.pattern_signals,
        # Canonical ICE.* metadata for fired patterns — feeds cosine
        # via the augmented embedding text and the LLM prompt at
        # first pass.  Surfaced here so SAGE/SHAP attribution can
        # treat ontology priors as a discrete feature distinct from
        # raw embedding text.  Universal-substrate codes; never
        # returned as classification targets.
        "ontology_priors": list(features.ontology_priors),
        "belief_path": belief_path,
        "cautious_code": hc.cautious_code(0.7),
        # Codes anywhere in the hierarchy (leaf or internal node)
        # whose belief meets the threshold — surfaces cross-subtree
        # disagreement that ``belief_path`` (confined to the
        # predicted leaf's ancestor chain) cannot show.  When cosine
        # evidence localizes to a subtree the LLM did not pick, the
        # subtree's internal-node parent appears here with its
        # belief mass, even when the predicted leaf sits elsewhere.
        # See docs/src/architecture/dst-evidence-independence.md.
        "cross_subtree_belief": hc.cross_subtree_belief(),
        # Smets' least-commitment promotion — when the predicted
        # leaf is below the commit threshold AND the system flags
        # ``needs_clarification``, this field carries the more-
        # general code where evidence IS unambiguous.  ``predicted_code``
        # retains its leaf-argmax semantics for backward
        # compatibility with Atlas governance sync; operators
        # consult ``cautious_promoted_code`` when the prediction is
        # flagged as uncertain.
        "cautious_promoted_code": hc.cautious_promoted_code(),
        # Curated reference (per-column answer key for accuracy checks)
        # attached at sample-load time by the source loader.  The code
        # is a reference for accuracy checking, not a published
        # human-curated benchmark.  ``reference_label`` is resolved
        # via the category_set to humanize the code for display.
        "reference_code": col.reference_code,
        "reference_label": (
            getattr(category_set.all_by_code.get(col.reference_code), "label", "")
            if col.reference_code and hasattr(category_set, "all_by_code")
            else ""
        ),
        "matches_reference": (
            col.reference_code == best_code
            if col.reference_code and best_code
            else None
        ),
        # Raw LLM vote preserved alongside the fused label so operators
        # can see (a) whether the LLM covered this column at all and
        # (b) whether the final prediction agrees with the LLM.  When
        # fit-to-LLM is on these match by construction on LLM-covered
        # columns; divergence on LLM-missing columns means CatBoost /
        # SVM carried the prediction.  ``llm_code`` is None when the
        # LLM didn't see this column (e.g. batch truncation).
        "llm_code": llm_code,
        "llm_confidence": float(llm_confidence or 0.0),
        # Independent-tier consensus (cosine + pattern + name_match,
        # excluding LLM-derivative sources).  Drives the revisit gate
        # at ``_identify_disagreements``.
        "independent_top1_code": indep_top1_code,
        "independent_top1_mass": round(indep_top1_mass, 4),
        "independent_top1_conflict": round(indep_top1_conflict, 4),
    }


def _empty_classification(col, features) -> dict[str, Any]:
    """Return empty classification when no evidence is available."""
    return {
        "table_name": col.table_name,
        "column_name": col.name,
        "column_type": col.column_type,
        "predicted_code": None,
        "predicted_label": "",
        "predicted_annotation": "",
        "confidence": 0.0,
        "belief": 0.0,
        "plausibility": 1.0,
        "uncertainty": 1.0,
        "conflict": 0.0,
        "needs_clarification": False,
        "evidence": [],
        "evidence_sources": {},
        "embedding_text": features.to_embedding_text(),
        "pattern_signals": features.pattern_signals,
        "ontology_priors": list(features.ontology_priors),
        "reference_code": col.reference_code,
        "reference_label": "",
        "matches_reference": None,
    }


def _is_vacuous(assignment) -> bool:
    """Check if a BeliefAssignment is vacuous (all mass on Theta)."""
    if len(assignment.masses) == 1:
        fe = next(iter(assignment.masses))
        return len(fe.codes) > 1  # Theta has all codes
    return False


def _mass_summary(assignment) -> dict[str, float]:
    """Summarize a BeliefAssignment as top-3 singletons."""
    singletons = sorted(
        [(next(iter(fe.codes)), m) for fe, m in assignment.masses.items() if len(fe.codes) == 1],
        key=lambda x: -x[1],
    )
    return {code: round(m, 4) for code, m in singletons[:3]}


def _run_feature_analysis(
    cfg,
    classifications: list[dict[str, Any]],
    all_samples: list[TableSample],
    category_set: HierarchicalCategorySet,
    results_dir: Path,
    mc_plan=None,
) -> None:
    """Run SHAP and SAGE feature analysis, mutating classifications in-place.

    Both are gated by config (classify_shap_enabled, classify_sage_enabled).
    SAGE uses predicted class indices as supervision — it measures feature
    contribution to the model's own decisions, not a curated reference.

    When MC is active and ``classify_background_analysis`` is true, SHAP runs
    in a background daemon thread. SAGE runs on the frontier sample only
    (representative subset).
    """
    all_features = [
        extract_features(
            column_name=col.name,
            column_type=col.column_type,
            values=col.values,
            siblings=col.siblings,
            source_table=col.table_name,
            total_count=col.total_count,
            null_count=col.null_count,
            distinct_count=col.distinct_count,
        )
        for ts in all_samples for col in ts.columns
    ]

    # ── SHAP (per-item explanations) ────────────────────────
    if cfg.classify_shap_enabled:
        if cfg.classify_background_analysis and mc_plan and not mc_plan.is_passthrough:
            # Background SHAP for large corpora
            import threading
            def _shap_background():
                try:
                    _run_shap(cfg, all_features, category_set, classifications, results_dir)
                except Exception as e:
                    logger.warning("Background SHAP failed: %s", e)
            t = threading.Thread(target=_shap_background, daemon=True)
            t.start()
            logger.info("SHAP started in background thread")
        else:
            _run_shap(cfg, all_features, category_set, classifications, results_dir)

    # ── SAGE (global feature importance) ────────────────────
    # SAGE is critical — it quantifies per-feature contribution to
    # classification. The config flag gates runtime cost for dev/UI
    # testing, not because SAGE is optional.  When a GPU is present
    # we also auto-enable SAGE (kernel runtime is tens of seconds on
    # synth-scale corpora); CPU users keep the opt-in default.
    sage_auto = False
    if not cfg.classify_sage_enabled and cfg.classify_gpu_enabled != "false":
        try:
            from atelier.classify.gpu import preflight_gpu
            if preflight_gpu().available:
                sage_auto = True
                logger.info("SAGE auto-enabled on GPU (kernel runtime acceptable)")
        except Exception:
            pass

    if cfg.classify_sage_enabled or sage_auto:
        # When MC active, run SAGE on frontier sample only (representative)
        sage_features = all_features
        sage_classifications = classifications
        if mc_plan and not mc_plan.is_passthrough:
            frontier_set = mc_plan.frontier_columns
            sage_pairs = [
                (feat, cls) for feat, cls in zip(all_features, classifications)
                if cls["column_name"] in frontier_set
            ]
            if sage_pairs:
                sage_features, sage_classifications = zip(*sage_pairs)
                sage_features = list(sage_features)
                sage_classifications = list(sage_classifications)
                logger.info("SAGE: using %d frontier columns (of %d total)",
                            len(sage_features), len(all_features))

        try:
            import numpy as np
            from atelier.classify.sage import run_sage_analysis

            code_to_idx = {cat.code: i for i, cat in enumerate(category_set.categories)}
            label_idx = np.array([
                code_to_idx.get(c["predicted_code"], 0)
                for c in sage_classifications
            ])

            sage_result = run_sage_analysis(
                sage_features, label_idx, category_set,
                n_permutations=cfg.classify_sage_permutations,
                detect_convergence=True,
            )
            logger.info("SAGE: %d features, %.1fs", len(sage_result.feature_names), sage_result.elapsed_seconds)
            sage_path = results_dir / "sage_importance.json"
            sage_path.write_text(json.dumps(sage_result.to_dict(), indent=2) + "\n")
        except Exception as e:
            logger.warning("SAGE analysis failed: %s", e)


def _run_shap(cfg, all_features, category_set, classifications, results_dir):
    """Run SHAP analysis synchronously."""
    try:
        from atelier.classify.shap_explanations import run_shap_analysis
        shap_method = getattr(cfg, "classify_shap_method", "auto") or "auto"
        shap_result = run_shap_analysis(all_features, category_set, method=shap_method)
        if shap_result:
            shap_records = shap_result.to_records(k=cfg.classify_shap_top_k)
            for cls_dict, shap_row in zip(classifications, shap_records):
                cls_dict.update(shap_row)
            logger.info("SHAP: %s method, %d items", shap_result.method, shap_result.n_items)
            shap_path = results_dir / "shap_summary.json"
            shap_path.write_text(json.dumps(shap_result.to_dict(), indent=2) + "\n")
    except Exception as e:
        logger.warning("SHAP analysis failed: %s", e)


def _evaluate_results(classifications: list[dict]) -> dict[str, Any]:
    """Compute summary statistics for a classification run."""
    total = len(classifications)
    if total == 0:
        return {"total_columns": 0}

    classified = sum(1 for c in classifications if c["predicted_code"])
    with_reference = [c for c in classifications if c["matches_reference"] is not None]
    correct = sum(1 for c in with_reference if c["matches_reference"])

    avg_confidence = sum(c["confidence"] for c in classifications) / total
    avg_conflict = sum(c["conflict"] for c in classifications) / total
    avg_uncertainty = sum(c["uncertainty"] for c in classifications) / total

    # LLM coverage + LLM-fusion agreement — the two metrics that
    # actually reflect pipeline health under the fit-to-llm regime.
    # Coverage = fraction with any LLM evidence at all.  Agreement =
    # of those, fraction where the fused predicted_code equals the
    # LLM's top pick.  With fit-to-LLM on, agreement should track
    # 100% on LLM-covered columns; divergence is a diagnostic signal
    # that CatBoost disagreed with the LLM after fine-tuning.
    llm_covered = [c for c in classifications if c.get("llm_code")]
    llm_matches = sum(
        1 for c in llm_covered
        if c.get("llm_code") == c.get("predicted_code")
    )

    return {
        "total_columns": total,
        "classified": classified,
        "coverage": round(classified / total, 4) if total else 0.0,
        "llm_coverage": round(len(llm_covered) / total, 4) if total else 0.0,
        "llm_agreement": (
            round(llm_matches / len(llm_covered), 4) if llm_covered else None
        ),
        "with_reference": len(with_reference),
        "correct": correct,
        "accuracy": round(correct / len(with_reference), 4) if with_reference else None,
        "avg_confidence": round(avg_confidence, 4),
        "avg_conflict": round(avg_conflict, 4),
        "avg_uncertainty": round(avg_uncertainty, 4),
    }


def _write_register_error(
    results_dir: Path,
    kind: str,
    run_id: str,
    source_id: str | None,
    exc: BaseException,
) -> None:
    """Persist a registration failure as a sidecar in the run directory.

    Both ``dao.upsert_dataset`` and ``dao.register_artifact_set`` are
    non-fatal — the artifacts on disk are the source of truth and a
    later backfill can repair the missing rows.  But the original code
    swallowed the failure with a single warning line, so a transient
    PGlite outage during a multi-hour run produced an orphan: parquet
    on disk, no UI-visible dataset, no obvious signal.  This sidecar
    makes the failure discoverable for both humans (read the file) and
    tools (``scripts/backfill_dataset.py`` keys off it).
    """
    import traceback as _tb

    path = results_dir / "register_error.json"
    existing: list[dict] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, list):
                existing = loaded
        except Exception:
            pass
    existing.append({
        "kind": kind,
        "run_id": run_id,
        "source_id": source_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": _tb.format_exc(),
    })
    try:
        path.write_text(json.dumps(existing, indent=2) + "\n")
    except Exception:
        # If we can't even write to results_dir, the operator already
        # has bigger problems — the artifacts wouldn't have made it
        # this far.  Don't mask the original exception.
        logger.exception("Could not write register_error.json")


def _write_parquet(
    classifications: list[dict],
    output_path: Path,
) -> Path | None:
    """Write classifications to parquet for embedding-atlas.

    Produces atlas-compatible columns: text, x, y (plus classification metadata).
    Uses UMAP on sentence-transformer embeddings when available, otherwise falls
    back to a deterministic PCA-like projection from DST numeric features.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("pyarrow not available; skipping parquet output")
        return None

    if not classifications:
        return None

    # Build the atlas hover/search text.  Format: ``{Ontology} - {Annotation}``
    # so the embedding-atlas tooltip shows e.g. ``"Bank Account - BAN"``
    # rather than a numeric code.  Falls back gracefully when either
    # field is missing.
    texts = []
    for c in classifications:
        ontology = c["predicted_label"] or ""
        annotation = c.get("predicted_annotation", "") or ""
        if ontology and annotation:
            texts.append(f"{ontology} - {annotation}")
        elif ontology:
            texts.append(ontology)
        else:
            texts.append(c["predicted_code"] or "unknown")

    # Compute 2D projection.  ``umap_reducer`` is the fitted CPU-side
    # UMAP suitable for pickling (None when cuml or PCA fallback fired);
    # the caller persists it to umap.pkl so Extend runs can land in the
    # same coordinate space.
    x_vals, y_vals, umap_reducer = _compute_projection(classifications, texts)

    rows = []
    for i, c in enumerate(classifications):
        row = {
            "text": texts[i],
            "x": x_vals[i],
            "y": y_vals[i],
            "table_name": c["table_name"],
            "column_name": c["column_name"],
            "column_type": c["column_type"] or "",
            "predicted_code": c["predicted_code"] or "",
            "predicted_label": c["predicted_label"] or "",
            "predicted_annotation": c.get("predicted_annotation", "") or "",
            "llm_code": c.get("llm_code") or "",
            "llm_confidence": c.get("llm_confidence", 0.0),
            "confidence": c["confidence"],
            "belief": c["belief"],
            "plausibility": c["plausibility"],
            "uncertainty": c["uncertainty"],
            "conflict": c["conflict"],
            "needs_clarification": c.get("needs_clarification", False),
            "evidence": c.get("evidence", ""),
            # Curated reference (per-column answer key from the
            # generator-derived + spot-checked ``curated_reference.csv``).
            # The name reflects that this code is a reference for
            # accuracy checking, not a published human-curated label.
            "reference_code": c.get("reference_code") or "",
            "reference_label": c.get("reference_label") or "",
            # True/False when a reference is available and comparable to
            # the pipeline's prediction; None when no reference exists
            # for this column.
            "matches_reference": c.get("matches_reference"),
            "embedding_text": c.get("embedding_text", ""),
            "pattern_signals": ", ".join(c.get("pattern_signals", {})),
            "dst_belief_path": json.dumps(c.get("belief_path", [])),
            "cautious_code": c.get("cautious_code", ""),
            # Cautious-code review audit.  Empty strings
            # when the column wasn't a review candidate; the
            # pre-review code is the original predicted_code from
            # DST fusion before agent backoff.
            "predicted_code_pre_review": c.get("predicted_code_pre_review", ""),
            "review_decision": c.get("review_decision", ""),
            "review_rationale": c.get("review_rationale", ""),
        }
        # SHAP columns (present when SHAP analysis ran)
        for rank in range(1, 4):
            row[f"shap_top{rank}_name"] = c.get(f"shap_top{rank}_name", "")
            row[f"shap_top{rank}_value"] = c.get(f"shap_top{rank}_value", 0.0)
        rows.append(row)

    table = pa.table({
        k: [r[k] for r in rows]
        for k in rows[0].keys()
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output_path))

    # Persist the fitted CPU-side UMAP reducer alongside the parquet so
    # an Extend Classification run can call ``reducer.transform()`` on
    # new column embeddings and land in the same 2D coordinate space.
    # When ``umap_reducer`` is None (cuml-GPU path or PCA fallback) we
    # skip the save; Extend handles the absence by re-fitting on its
    # own embeddings with a logged warning.
    if umap_reducer is not None:
        try:
            import joblib
            from atelier.classify.artifact_set import UMAP_FILENAME
            umap_path = output_path.parent / UMAP_FILENAME
            joblib.dump(umap_reducer, umap_path)
            logger.info("UMAP reducer saved to %s", umap_path)
        except Exception as e:
            logger.warning("Failed to save UMAP reducer (Extend will re-fit): %s", e)

    return output_path


def _compute_projection(
    classifications: list[dict],
    texts: list[str],
) -> tuple[list[float], list[float], object | None]:
    """Compute 2D x/y coordinates for embedding-atlas.

    Tries UMAP on sentence-transformer embeddings first (best quality).
    Falls back to PCA on DST numeric features (always available).

    Returns ``(x, y, reducer)`` where ``reducer`` is the fitted
    umap-learn ``UMAP`` instance when one was used (suitable for
    pickling via joblib so an Extend Classification run can call
    ``reducer.transform(new_embeddings)`` to land in the same 2D
    coordinate space).  Returns ``None`` for the third element when
    the cuml GPU path was used (cuml.UMAP doesn't round-trip well
    across CPU/GPU pickle boundaries) or when the PCA fallback
    fired — Extend handles the None case by re-fitting with a warning.
    """
    # Try UMAP + sentence-transformers for high-quality projection.
    # When the optional [gpu] extra is installed and a GPU is available,
    # prefer cuml.UMAP (an order of magnitude faster on large corpora);
    # otherwise fall back to umap-learn (CPU).
    try:
        from atelier.classify.embedding import _get_model, get_batch_size
        import numpy as np

        model = _get_model()
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=get_batch_size())
        n_neighbors = min(15, max(2, len(texts) - 1))

        projection = None
        cpu_reducer = None
        try:
            from atelier.classify.gpu import preflight_gpu
            if preflight_gpu().available:
                from cuml.manifold import UMAP as CuUMAP  # type: ignore
                reducer = CuUMAP(
                    n_components=2, n_neighbors=n_neighbors,
                    min_dist=0.1, metric="cosine", random_state=42,
                )
                projection = reducer.fit_transform(embeddings)
                logger.info("UMAP projection: cuml.UMAP (GPU)")
                # cuml.UMAP doesn't pickle reliably across CPU/GPU
                # boundaries, so we don't bundle it for Extend.
        except ImportError:
            logger.debug("cuml not installed; falling back to umap-learn (CPU)")

        if projection is None:
            import umap
            cpu_reducer = umap.UMAP(
                n_components=2, n_neighbors=n_neighbors,
                min_dist=0.1, metric="cosine", random_state=42,
            )
            projection = cpu_reducer.fit_transform(embeddings)

        if hasattr(projection, "values"):
            projection = projection.values
        projection = np.asarray(projection)
        return (
            projection[:, 0].tolist(),
            projection[:, 1].tolist(),
            cpu_reducer,
        )
    except Exception as e:
        logger.debug("UMAP projection unavailable (%s), using DST feature projection", e)

    # Fallback: PCA-like projection from DST numeric features
    import numpy as np

    features = np.array([
        [c["confidence"], c["belief"], c["plausibility"],
         c["uncertainty"], c["conflict"]]
        for c in classifications
    ], dtype=np.float32)

    # Center and project onto first two principal components
    centered = features - features.mean(axis=0)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ vt[:2].T
    except np.linalg.LinAlgError:
        # Degenerate case — use confidence vs belief directly
        proj = features[:, :2]

    return proj[:, 0].tolist(), proj[:, 1].tolist(), None
