"""End-to-end classification pipeline orchestration.

Drives the AgentFSM through:
  LOADING_VOCAB → DISCOVERING → SAMPLING → CLASSIFYING → FUSING → EVALUATING

Writes results to build/results/{run_id}/ as JSON and parquet.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from atelier.classify.belief import FrameOfDiscernment, combine_multiple
from atelier.classify.features import extract_features
from atelier.classify.fsm import AgentFSM, FSMState
from atelier.classify.mass_functions import (
    cosine_to_mass,
    name_match_to_mass,
    pattern_to_mass,
)
from atelier.classify.sampler import (
    ColumnSample,
    TableSample,
    discover_tables,
    load_all_mock_samples,
    load_annotations_from_hive,
    sample_table_metadata,
)
from atelier.classify.taxonomy import (
    HierarchicalCategorySet,
    load_annotations_from_json,
    load_mock_annotations,
    save_annotations_json,
    _build_category_set_from_records,
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
    use_mock: bool = False,
) -> dict[str, Any]:
    """Run the full classification pipeline.

    Args:
        cfg: AtelierConfig.
        fsm: AgentFSM instance for state tracking.
        connection_name: CAI data connection name.
        database: Hive database to classify.
        sample_size: Rows to sample per table.
        tables_limit: Max tables to discover.
        use_mock: Force mock data (for devenv/CI).

    Returns:
        Pipeline result summary dict.
    """
    run = fsm.start_run(config={
        "connection_name": connection_name,
        "database": database,
        "sample_size": sample_size,
        "tables_limit": tables_limit,
        "use_mock": use_mock,
    })
    run_id = run.id

    build_dir = _PROJECT_ROOT / "build"
    results_dir = build_dir / "results" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── LOADING_VOCAB ────────────────────────────────────────
        fsm.advance(run_id, FSMState.LOADING_VOCAB, progress={"step": "loading_vocab"})
        category_set = _load_vocabulary(cfg, build_dir, connection_name, use_mock)
        logger.info("Loaded %d leaf categories", len(category_set.categories))

        if not isinstance(category_set, HierarchicalCategorySet):
            raise RuntimeError("Expected HierarchicalCategorySet")

        frame = FrameOfDiscernment(category_set)
        fsm.advance(run_id, FSMState.DISCOVERING, progress={
            "categories_loaded": len(category_set.categories),
        })

        # ── DISCOVERING ──────────────────────────────────────────
        if use_mock:
            table_names = [t.name for t in load_all_mock_samples()]
        else:
            table_names = discover_tables(
                cfg, connection_name, database, limit=tables_limit
            )
        logger.info("Discovered %d tables", len(table_names))

        fsm.advance(run_id, FSMState.SAMPLING, progress={
            "tables_discovered": len(table_names),
        })

        # ── SAMPLING ─────────────────────────────────────────────
        all_samples: list[TableSample] = []
        if use_mock:
            all_samples = load_all_mock_samples()
        else:
            for tname in table_names:
                try:
                    ts = sample_table_metadata(
                        cfg, tname, connection_name, database, sample_size
                    )
                    all_samples.append(ts)
                except Exception as exc:
                    logger.warning("Failed to sample %s: %s", tname, exc)

        total_columns = sum(len(ts.columns) for ts in all_samples)
        logger.info("Sampled %d columns across %d tables", total_columns, len(all_samples))

        fsm.advance(run_id, FSMState.CLASSIFYING, progress={
            "tables_sampled": len(all_samples),
            "columns_sampled": total_columns,
        })

        # ── CLASSIFYING + FUSING ─────────────────────────────────
        # Try sentence-transformers for cosine; fall back to name+pattern only
        has_embeddings = False
        try:
            from atelier.classify.embedding import classify_cosine
            has_embeddings = True
        except ImportError:
            logger.warning("sentence-transformers not available; using name+pattern only")

        classifications: list[dict[str, Any]] = []
        for ts in all_samples:
            for col in ts.columns:
                result = _classify_column(
                    col, category_set, frame,
                    use_cosine=has_embeddings,
                )
                classifications.append(result)

        fsm.advance(run_id, FSMState.FUSING, progress={
            "columns_classified": len(classifications),
        })

        # ── EVALUATING ───────────────────────────────────────────
        fsm.advance(run_id, FSMState.EVALUATING, progress={
            "columns_fused": len(classifications),
        })

        summary = _evaluate_results(classifications)

        # Write results
        results_path = results_dir / "classifications.json"
        results_path.write_text(json.dumps(classifications, indent=2, default=str) + "\n")

        # Write parquet if pyarrow available
        parquet_path = _write_parquet(classifications, results_dir / "atelier_embeddings.parquet")

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


def _load_vocabulary(cfg, build_dir: Path, connection_name, use_mock: bool):
    """Load vocabulary from hive, cache, or mock."""
    cache_dir = build_dir / "data" / "annotations"
    cache_path = cache_dir / "annotations.json"

    if use_mock:
        cs = load_mock_annotations(hierarchical=True)
        save_annotations_json(cs, cache_path)
        return cs

    # Try cached first
    if cache_path.exists():
        return load_annotations_from_json(cache_path, hierarchical=True)

    # Try hive
    try:
        records = load_annotations_from_hive(cfg, connection_name)
        cs = _build_category_set_from_records(records, hierarchical=True)
        save_annotations_json(cs, cache_path)
        return cs
    except Exception:
        # Fall back to mock
        return load_mock_annotations(hierarchical=True)


def _classify_column(
    col: ColumnSample,
    category_set: HierarchicalCategorySet,
    frame: FrameOfDiscernment,
    *,
    use_cosine: bool = True,
) -> dict[str, Any]:
    """Classify a single column using available evidence sources."""
    features = extract_features(
        column_name=col.name,
        column_type=col.column_type,
        values=col.values,
        siblings=col.siblings,
        source_table=col.table_name,
        total_count=col.total_count,
        null_count=col.null_count,
    )

    # Collect evidence sources
    assignments = []
    evidence_sources: dict[str, Any] = {}

    # 1. Name matching
    name_mass = name_match_to_mass(col.name, frame, category_set)
    if not _is_vacuous(name_mass):
        assignments.append(name_mass)
        evidence_sources["name_match"] = _mass_summary(name_mass)

    # 2. Pattern detection
    pattern_mass = pattern_to_mass(features.pattern_signals, frame)
    if not _is_vacuous(pattern_mass):
        assignments.append(pattern_mass)
        evidence_sources["pattern"] = _mass_summary(pattern_mass)

    # 3. Cosine similarity (if available)
    if use_cosine:
        try:
            from atelier.classify.embedding import classify_cosine as _cosine
            similarities = _cosine(features, category_set)
            cosine_mass = cosine_to_mass(similarities, frame)
            assignments.append(cosine_mass)
            evidence_sources["cosine"] = _mass_summary(cosine_mass)
        except Exception:
            pass

    # Fuse evidence
    if not assignments:
        return _empty_classification(col, features)

    if len(assignments) == 1:
        combined = assignments[0]
        conflict = 0.0
    else:
        combined, conflict = combine_multiple(assignments)

    # Extract predictions
    best_code, best_mass = _best_singleton(combined, frame)
    bel = combined.belief(frame.singleton(best_code)) if best_code else 0.0
    pl = combined.plausibility(frame.singleton(best_code)) if best_code else 0.0

    best_label = ""
    if best_code:
        cat = category_set.by_code.get(best_code)
        if cat:
            best_label = cat.label

    return {
        "table_name": col.table_name,
        "column_name": col.name,
        "column_type": col.column_type,
        "predicted_code": best_code,
        "predicted_label": best_label,
        "confidence": round(best_mass, 4) if best_mass else 0.0,
        "belief": round(bel, 4),
        "plausibility": round(pl, 4),
        "uncertainty": round(pl - bel, 4),
        "conflict": round(conflict, 4),
        "evidence_sources": evidence_sources,
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
        "evidence_sources": {},
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


def _best_singleton(assignment, frame) -> tuple[str | None, float]:
    """Find the singleton with highest mass in a BeliefAssignment."""
    best_code = None
    best_mass = 0.0
    for fe, m in assignment.masses.items():
        if len(fe.codes) == 1 and m > best_mass:
            best_code = next(iter(fe.codes))
            best_mass = m
    return best_code, best_mass


def _mass_summary(assignment) -> dict[str, float]:
    """Summarize a BeliefAssignment as top-3 singletons."""
    singletons = sorted(
        [(next(iter(fe.codes)), m) for fe, m in assignment.masses.items() if len(fe.codes) == 1],
        key=lambda x: -x[1],
    )
    return {code: round(m, 4) for code, m in singletons[:3]}


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
    """Write classifications to parquet for embedding-atlas."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("pyarrow not available; skipping parquet output")
        return None

    rows = []
    for c in classifications:
        rows.append({
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
            "ground_truth": c["ground_truth"] or "",
            "is_correct": c["is_correct"] if c["is_correct"] is not None else False,
            "pattern_signals": ", ".join(c.get("pattern_signals", [])),
        })

    if not rows:
        return None

    table = pa.table({
        k: [r[k] for r in rows]
        for k in rows[0].keys()
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output_path))
    return output_path
