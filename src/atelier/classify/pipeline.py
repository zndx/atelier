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
from pathlib import Path
from typing import Any

from atelier.classify.belief import (
    FrameOfDiscernment,
    HierarchicalClassification,
)
from atelier.classify.evaluation import evaluate_classifications
from atelier.classify.features import extract_features
from atelier.classify.fsm import AgentFSM, FSMState
from atelier.classify.mass_functions import (
    DiscountConfig,
    catboost_to_mass,
    cosine_to_mass,
    llm_to_mass,
    name_match_to_mass,
    pattern_to_mass,
    svm_to_mass,
)
from atelier.classify.sampler import (
    ColumnSample,
    TableSample,
    discover_tables,
    sample_table_metadata,
)
from atelier.classify.taxonomy import (
    HierarchicalCategorySet,
    compose_vocabularies,
    load_annotations_from_hive,
    load_annotations_from_json,
    load_universal_vocabulary,
    save_annotations_json,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def run_classification_pipeline(
    cfg,
    fsm: AgentFSM,
    *,
    connection_name: str | None = None,
    database: str = "default",
    sample_size: int = 50,
    tables_limit: int = 100,
    samples: list[TableSample] | None = None,
    category_set: HierarchicalCategorySet | None = None,
    llm_backend=None,
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
    # ── LLM backend resolution ────────────────────────────────
    # The pipeline cannot function without an LLM.  Resolve early
    # so callers get a clear error before any FSM state is created.
    if llm_backend is None:
        from atelier.classify.llm_backend import create_backend_from_cfg
        # create_backend_from_cfg raises ValueError when no creds
        llm_backend = create_backend_from_cfg(cfg)

    run = fsm.start_run(config={
        "connection_name": connection_name,
        "database": database,
        "sample_size": sample_size,
        "tables_limit": tables_limit,
    })
    run_id = run.id

    build_dir = _PROJECT_ROOT / "build"
    results_dir = build_dir / "results" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── LOADING_VOCAB ────────────────────────────────────────
        fsm.advance(run_id, FSMState.LOADING_VOCAB, progress={"step": "loading_vocab"})
        if category_set is None:
            category_set = _load_vocabulary(cfg, build_dir, connection_name)
        logger.info("Loaded %d leaf categories", len(category_set.categories))

        if not isinstance(category_set, HierarchicalCategorySet):
            raise RuntimeError("Expected HierarchicalCategorySet")

        frame = FrameOfDiscernment(category_set)
        fsm.advance(run_id, FSMState.DISCOVERING, progress={
            "categories_loaded": len(category_set.categories),
        })

        # ── DISCOVERING + SAMPLING ────────────────────────────────
        if samples is not None:
            all_samples = samples
            fsm.advance(run_id, FSMState.SAMPLING, progress={
                "tables_discovered": len(all_samples),
                "injected": True,
            })
        else:
            table_names = discover_tables(
                cfg, connection_name, database, limit=tables_limit
            )
            logger.info("Discovered %d tables", len(table_names))

            fsm.advance(run_id, FSMState.SAMPLING, progress={
                "tables_discovered": len(table_names),
            })

            all_samples: list[TableSample] = []
            for tname in table_names:
                try:
                    ts = sample_table_metadata(
                        cfg, tname, connection_name, database, sample_size
                    )
                    all_samples.append(ts)
                except Exception as exc:
                    logger.warning("Failed to sample %s: %s", tname, exc)

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
            bootstrap_config_from_cfg,
            _coverage,
            _identify_disagreements,
            _llm_revisit,
            _llm_sweep,
            _mean_k,
            _run_ml_validation,
        )
        from atelier.classify.llm_backend import (
            build_category_table,
            build_system_prompt,
        )

        boot_cfg = bootstrap_config_from_cfg(cfg)
        category_table = build_category_table(category_set)
        system_prompt = build_system_prompt(category_table)

        # Wire config → ml_inference model paths
        from atelier.classify import ml_inference
        ml_inference.configure_paths(
            catboost_path=cfg.classify_catboost_model_path,
            svm_path=cfg.classify_svm_model_path,
        )

        discounts = DiscountConfig.from_cfg(cfg)

        # Try sentence-transformers for cosine
        has_embeddings = False
        try:
            from atelier.classify.embedding import classify_cosine
            has_embeddings = True
        except ImportError:
            logger.warning("sentence-transformers not available; using name+pattern only")

        # ── LLM SWEEP ────────────────────────────────────────────
        fsm.advance(run_id, FSMState.LLM_SWEEP, progress={
            "columns_total": total_columns,
            "phase": "llm_sweep",
        })

        state = BootstrapState()

        _llm_sweep(
            state, boot_cfg, llm_backend, system_prompt,
            column_names, samples_by_name, column_table,
        )

        coverage = _coverage(state, column_names)
        logger.info(
            "LLM sweep: labeled %d/%d (coverage=%.1f%%, calls=%d)",
            len(state.labels), total_columns,
            coverage * 100, state.llm_calls_total,
        )

        # ── VALIDATING ───────────────────────────────────────────
        fsm.advance(run_id, FSMState.VALIDATING, progress={
            "phase": "ml_validation",
            "llm_labeled": len(state.labels),
            "coverage": round(coverage, 4),
        })

        _run_ml_validation(
            state, boot_cfg, column_names, samples_by_name,
            category_set, frame, has_embeddings, discounts=discounts,
        )

        disagreements = _identify_disagreements(state, column_names, boot_cfg)
        mean_k = _mean_k(state, column_names)

        logger.info(
            "ML validation: mean K=%.3f, disagreements=%d",
            mean_k, len(disagreements),
        )

        # ── TARGETED REVISIT LOOP ────────────────────────────────
        iteration_metrics: list[dict[str, Any]] = [{
            "iteration": 0,
            "mean_k": round(mean_k, 4),
            "disagreements": len(disagreements),
            "coverage": round(coverage, 4),
        }]

        for iteration in range(1, boot_cfg.max_iterations + 1):
            if not disagreements:
                logger.info("No disagreements — converged")
                break

            if state.llm_calls_total >= boot_cfg.max_total_llm_calls:
                logger.info("Budget exhausted (%d calls)", state.llm_calls_total)
                break

            if mean_k < boot_cfg.k_threshold:
                logger.info("Mean K=%.3f < threshold — converged", mean_k)
                break

            state.iteration = iteration

            fsm.advance(run_id, FSMState.LLM_SWEEP, progress={
                "phase": "revisit",
                "iteration": iteration,
                "disagreements": len(disagreements),
                "mean_k": round(mean_k, 4),
            })

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
            )

            disagreements = _identify_disagreements(state, column_names, boot_cfg)
            mean_k = _mean_k(state, column_names)
            coverage = _coverage(state, column_names)

            logger.info(
                "Revisit %d: mean K=%.3f, disagreements=%d, coverage=%.1f%%, calls=%d",
                iteration, mean_k, len(disagreements),
                coverage * 100, state.llm_calls_total,
            )

            iteration_metrics.append({
                "iteration": iteration,
                "mean_k": round(mean_k, 4),
                "disagreements": len(disagreements),
                "coverage": round(coverage, 4),
            })

        # ── FINAL CLASSIFICATION PASS ────────────────────────────
        coverage = _coverage(state, column_names)
        mean_k = _mean_k(state, column_names)
        converged = coverage >= boot_cfg.coverage_target and mean_k < boot_cfg.k_threshold

        fsm.advance(run_id, FSMState.CLASSIFYING, progress={
            "phase": "final_classification",
            "converged": converged,
            "mean_k": round(mean_k, 4),
            "coverage": round(coverage, 4),
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
            )
            classifications.append(result)

        fsm.advance(run_id, FSMState.FUSING, progress={
            "columns_classified": len(classifications),
        })

        # ── Feature analysis (SHAP + SAGE, config-gated) ──────────
        _run_feature_analysis(cfg, classifications, all_samples, category_set, results_dir)

        # ── EVALUATING ───────────────────────────────────────────
        fsm.advance(run_id, FSMState.EVALUATING, progress={
            "columns_fused": len(classifications),
        })

        summary = _evaluate_results(classifications)
        eval_report = evaluate_classifications(classifications, category_set)
        summary["converged"] = converged
        summary["bootstrap_iterations"] = state.iteration
        summary["llm_calls"] = state.llm_calls_total
        summary["tokens_input"] = state.tokens_input
        summary["tokens_output"] = state.tokens_output
        summary["mean_k"] = round(mean_k, 4)
        summary["bootstrap_coverage"] = round(coverage, 4)
        summary["iteration_metrics"] = iteration_metrics

        # Write results
        results_path = results_dir / "classifications.json"
        results_path.write_text(json.dumps(classifications, indent=2, default=str) + "\n")
        eval_report.write_json(results_dir / "evaluation_report.json")

        parquet_path = _write_parquet(classifications, results_dir / "atelier_embeddings.parquet")

        # Auto-register as a dataset so the Embeddings page is populated
        if parquet_path:
            try:
                from atelier.db.dao import AtelierDao
                dao = AtelierDao()
                dao.upsert_dataset(
                    dataset_id=run_id,
                    name=f"Classification {run_id[:8]}",
                    parquet_path=str(parquet_path),
                    description="Classification pipeline results",
                    row_count=len(classifications),
                )
            except Exception as e:
                logger.warning("Failed to register dataset: %s", e)

        fsm.advance(run_id, FSMState.CONVERGED, progress={
            **summary,
            "result_path": str(results_path),
            "parquet_path": str(parquet_path) if parquet_path else None,
        }, result_path=str(parquet_path) if parquet_path else str(results_path))

        return {
            "run_id": run_id,
            "state": "CONVERGED",
            "classifications": len(classifications),
            "result_path": str(results_path),
            "parquet_path": str(parquet_path) if parquet_path else None,
            "evaluation_report": eval_report.to_dict(),
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


def _load_vocabulary(cfg, build_dir: Path, connection_name):
    """Load vocabulary: universal base + optional domain extensions.

    Two-layer composition:
      1. Universal BFO-grounded vocabulary (always loaded, ships in git)
      2. Domain annotations (customer-specific, from hive or cache)

    Domain annotations compose on top of the universal base via ``is_a``
    parent references.  When unavailable, returns universal-only.
    """
    log = logging.getLogger(__name__)

    # Always start with BFO-grounded universal vocabulary
    universal = load_universal_vocabulary(hierarchical=True)
    log.info("Loaded universal vocabulary: %d terms", len(universal.categories))

    # Try domain extensions (customer annotations from cache or hive)
    domain_cs = _load_domain_annotations(cfg, build_dir, connection_name)
    if domain_cs is None:
        return universal

    # Compose: domain terms attach to universal tree via parent_code
    composed = compose_vocabularies(universal, domain_cs)
    log.info(
        "Composed vocabulary: %d universal + %d domain = %d total terms",
        len(universal.categories), len(domain_cs.categories),
        len(composed.categories),
    )
    return composed


def _load_domain_annotations(cfg, build_dir: Path, connection_name):
    """Load domain-specific annotations from cache or hive.

    Returns a CategorySet of domain terms, or None if unavailable.
    """
    log = logging.getLogger(__name__)
    cache_dir = build_dir / "data" / "annotations"
    cache_path = cache_dir / "annotations.json"

    # Try cached first — but reject empty caches (poisoned by prior failures)
    if cache_path.exists():
        cs = load_annotations_from_json(cache_path, hierarchical=True)
        if len(cs.categories) > 0:
            log.info("Loaded %d domain categories from cache %s", len(cs.categories), cache_path)
            return cs
        log.warning("Cache %s contains 0 categories — treating as corrupt, will re-fetch", cache_path)
        cache_path.unlink()

    # Try hive
    try:
        cs = load_annotations_from_hive(cfg, connection_name)
        if len(cs.categories) == 0:
            log.warning("Hive returned 0 domain categories — skipping domain layer")
            return None
        log.info("Loaded %d domain categories from hive", len(cs.categories))
        save_annotations_json(cs, cache_path)
        return cs
    except Exception as exc:
        log.warning("Failed to load domain annotations from hive: %s", exc)

    return None


def _classify_column(
    col: ColumnSample,
    category_set: HierarchicalCategorySet,
    frame: FrameOfDiscernment,
    *,
    llm_code: str | None = None,
    llm_confidence: float = 0.0,
    llm_alternatives: list[dict] | None = None,
    llm_discount: float = 0.10,
    use_cosine: bool = True,
    discounts: DiscountConfig | None = None,
) -> dict[str, Any]:
    """Classify a single column using Dempster-Shafer evidence fusion.

    Evidence sources (up to 6): name matching, pattern detection,
    cosine similarity, LLM, CatBoost, SVM.  The pipeline always
    supplies LLM evidence; llm_code may be None only for offline
    use cases such as seed data preparation.
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

    # 2. Pattern detection
    pattern_mass = pattern_to_mass(
        features.pattern_signals, frame,
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

    # 6. SVM (if model available)
    try:
        from atelier.classify.ml_inference import predict_svm
        svm_proba = predict_svm(features)
        if svm_proba:
            svm_mass = svm_to_mass(svm_proba, frame, discount=discounts.svm)
            if not _is_vacuous(svm_mass):
                source_masses["svm"] = svm_mass
    except Exception as exc:
        logger.debug("SVM unavailable for %s: %s", col.name, exc)

    # Fuse evidence via HierarchicalClassification
    if not source_masses:
        return _empty_classification(col, features)

    hc = HierarchicalClassification.from_combined_evidence(
        source_masses=source_masses,
        frame=frame,
        category_set=category_set,
    )

    best_code = hc.category.code
    bel, pl = hc.interval_at(best_code)

    return {
        "table_name": col.table_name,
        "column_name": col.name,
        "column_type": col.column_type,
        "predicted_code": best_code,
        "predicted_label": hc.category.label,
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
        "ground_truth": col.ground_truth,
        "is_correct": (
            col.ground_truth == best_code
            if col.ground_truth and best_code
            else None
        ),
    }


def _empty_classification(col, features) -> dict[str, Any]:
    """Return empty classification when no evidence is available."""
    return {
        "table_name": col.table_name,
        "column_name": col.name,
        "column_type": col.column_type,
        "predicted_code": None,
        "predicted_label": "",
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
        "ground_truth": col.ground_truth,
        "is_correct": None,
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
) -> None:
    """Run SHAP and SAGE feature analysis, mutating classifications in-place.

    Both are gated by config (classify_shap_enabled, classify_sage_enabled).
    SAGE uses predicted class indices as supervision — it measures feature
    contribution to the model's own decisions, not external ground truth.
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
        try:
            from atelier.classify.shap_explanations import run_shap_analysis
            shap_result = run_shap_analysis(all_features, category_set)
            if shap_result:
                shap_records = shap_result.to_records(k=cfg.classify_shap_top_k)
                for cls_dict, shap_row in zip(classifications, shap_records):
                    cls_dict.update(shap_row)
                logger.info("SHAP: %s method, %d items", shap_result.method, shap_result.n_items)
                shap_path = results_dir / "shap_summary.json"
                shap_path.write_text(json.dumps(shap_result.to_dict(), indent=2) + "\n")
        except Exception as e:
            logger.warning("SHAP analysis failed: %s", e)

    # ── SAGE (global feature importance) ────────────────────
    if cfg.classify_sage_enabled:
        try:
            import numpy as np
            from atelier.classify.sage import run_sage_analysis

            code_to_idx = {cat.code: i for i, cat in enumerate(category_set.categories)}
            gt_indices = np.array([
                code_to_idx.get(c["predicted_code"], 0)
                for c in classifications
            ])

            sage_result = run_sage_analysis(
                all_features, gt_indices, category_set,
                n_permutations=cfg.classify_sage_permutations,
                detect_convergence=True,
            )
            logger.info("SAGE: %d features, %.1fs", len(sage_result.feature_names), sage_result.elapsed_seconds)
            sage_path = results_dir / "sage_importance.json"
            sage_path.write_text(json.dumps(sage_result.to_dict(), indent=2) + "\n")
        except Exception as e:
            logger.warning("SAGE analysis failed: %s", e)


def _evaluate_results(classifications: list[dict]) -> dict[str, Any]:
    """Compute summary statistics for a classification run."""
    total = len(classifications)
    if total == 0:
        return {"total_columns": 0}

    classified = sum(1 for c in classifications if c["predicted_code"])
    with_truth = [c for c in classifications if c["is_correct"] is not None]
    correct = sum(1 for c in with_truth if c["is_correct"])

    avg_confidence = sum(c["confidence"] for c in classifications) / total
    avg_conflict = sum(c["conflict"] for c in classifications) / total
    avg_uncertainty = sum(c["uncertainty"] for c in classifications) / total

    return {
        "total_columns": total,
        "classified": classified,
        "coverage": round(classified / total, 4) if total else 0.0,
        "with_ground_truth": len(with_truth),
        "correct": correct,
        "accuracy": round(correct / len(with_truth), 4) if with_truth else None,
        "avg_confidence": round(avg_confidence, 4),
        "avg_conflict": round(avg_conflict, 4),
        "avg_uncertainty": round(avg_uncertainty, 4),
    }


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

    # Build text column for atlas hover/search
    texts = []
    for c in classifications:
        label = c["predicted_label"] or c["predicted_code"] or "unknown"
        texts.append(f"{c['table_name']}.{c['column_name']} — {label}")

    # Compute 2D projection
    x_vals, y_vals = _compute_projection(classifications, texts)

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
            "confidence": c["confidence"],
            "belief": c["belief"],
            "plausibility": c["plausibility"],
            "uncertainty": c["uncertainty"],
            "conflict": c["conflict"],
            "needs_clarification": c.get("needs_clarification", False),
            "evidence": c.get("evidence", ""),
            "ground_truth": c["ground_truth"] or "",
            "is_correct": c["is_correct"] if c["is_correct"] is not None else False,
            "embedding_text": c.get("embedding_text", ""),
            "pattern_signals": ", ".join(c.get("pattern_signals", [])),
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
    return output_path


def _compute_projection(
    classifications: list[dict],
    texts: list[str],
) -> tuple[list[float], list[float]]:
    """Compute 2D x/y coordinates for embedding-atlas.

    Tries UMAP on sentence-transformer embeddings first (best quality).
    Falls back to PCA on DST numeric features (always available).
    """
    # Try UMAP + sentence-transformers for high-quality projection
    try:
        from sentence_transformers import SentenceTransformer
        import umap
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, show_progress_bar=False, batch_size=256)
        n_neighbors = min(15, max(2, len(texts) - 1))
        reducer = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors,
            min_dist=0.1, metric="cosine", random_state=42,
        )
        projection = reducer.fit_transform(embeddings)
        return projection[:, 0].tolist(), projection[:, 1].tolist()
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

    return proj[:, 0].tolist(), proj[:, 1].tolist()
