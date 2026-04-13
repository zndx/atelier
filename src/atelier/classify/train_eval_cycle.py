"""End-to-end orchestrator: synth → train → classify → evaluate.

Used by tier-0 BDD tests to verify the full ML pipeline produces
meaningful classifications without external services.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_synth_train_eval(
    category_set,
    *,
    variants_per_category: int = 30,
    seed: int = 42,
    catboost_iterations: int = 500,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate synth → train CatBoost + SVM → classify mock data → evaluate.

    Returns dict with accuracy, micro_f1, macro_f1, evaluation_report,
    models_trained, evidence_sources_fired, catboost_evidence_count,
    svm_evidence_count, total_columns.
    """
    from atelier.classify.belief import FrameOfDiscernment
    from atelier.classify.evaluation import evaluate_classifications
    from atelier.classify.ml_train import train_all
    from atelier.classify.pipeline import _classify_column
    from atelier.classify.sampler import load_all_mock_samples
    from atelier.classify.synth import generate_synth_tables
    from atelier.classify import ml_inference

    cleanup = work_dir is None
    if work_dir is None:
        _tmpdir = tempfile.mkdtemp(prefix="atelier_m3_")
        work_dir = Path(_tmpdir)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)

    synth_dir = work_dir / "synth"
    models_dir = work_dir / "models"

    # 1. Generate synthetic data
    logger.info("Generating synthetic data (seed=%d)", seed)
    generate_synth_tables(
        category_set,
        synth_dir,
        variants_per_category=variants_per_category,
        seed=seed,
    )

    # 2. Train classifiers
    logger.info("Training CatBoost + SVM classifiers")
    model_paths = train_all(
        synth_dir,
        category_set,
        models_dir,
    )

    # 3. Configure inference to use our freshly trained models
    ml_inference.reset()
    ml_inference.configure_paths(
        catboost_path=model_paths["catboost"],
        svm_path=model_paths["svm"],
    )

    try:
        # 4. Classify mock data
        frame = FrameOfDiscernment(category_set)
        all_samples = load_all_mock_samples()
        classifications: list[dict[str, Any]] = []
        all_sources: set[str] = set()
        catboost_count = 0
        svm_count = 0

        for ts in all_samples:
            for col in ts.columns:
                result = _classify_column(col, category_set, frame, use_cosine=True)
                classifications.append(result)

                sources = result.get("evidence_sources", {})
                all_sources.update(sources.keys())
                if "catboost" in sources:
                    catboost_count += 1
                if "svm" in sources:
                    svm_count += 1

        # 5. Evaluate
        report = evaluate_classifications(classifications, category_set)

        return {
            "accuracy": report.exact_accuracy,
            "micro_f1": report.micro_f1,
            "macro_f1": report.macro_f1,
            "hierarchical_accuracy": report.hierarchical_accuracy,
            "evaluation_report": report.to_dict(),
            "models_trained": list(model_paths.keys()),
            "evidence_sources_fired": sorted(all_sources),
            "catboost_evidence_count": catboost_count,
            "svm_evidence_count": svm_count,
            "total_columns": len(classifications),
        }

    finally:
        # 6. Clean up inference state
        ml_inference.reset()


def run_real_data_eval(
    data_dir: Path,
    *,
    mode: str = "ml_only",
    llm_backend=None,
    variants_per_category: int = 30,
    seed: int = 42,
    catboost_iterations: int = 500,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Real vocabulary + templates → synth → train → classify real data → evaluate.

    Loads the real annotation vocabulary and sample data from CSV files,
    generates template-based synthetic training data, trains ML classifiers,
    then classifies the real columns and evaluates against ground truth.

    Args:
        data_dir: Path to directory with annotations.csv and data CSVs.
        mode: "ml_only" for single-pass ML, "bootstrap" for LLM convergence loop.
        llm_backend: Injected LLM backend for bootstrap mode.
        variants_per_category: Synthetic columns per category for training.
        seed: RNG seed.
        work_dir: Working directory for synth/models/results (temp if None).

    Returns:
        Dict with accuracy, micro_f1, hierarchical_accuracy, evaluation_report, etc.
    """
    from atelier.classify.belief import FrameOfDiscernment
    from atelier.classify.evaluation import evaluate_classifications
    from atelier.classify.fsm import AgentFSM
    from atelier.classify.ml_train import train_all
    from atelier.classify.pipeline import _classify_column
    from atelier.classify.real_data_loader import (
        extract_value_templates,
        load_real_samples,
        load_real_vocabulary,
    )
    from atelier.classify.synth import generate_synth_tables
    from atelier.classify import ml_inference

    data_dir = Path(data_dir)

    cleanup = work_dir is None
    if work_dir is None:
        _tmpdir = tempfile.mkdtemp(prefix="atelier_real_")
        work_dir = Path(_tmpdir)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)

    synth_dir = work_dir / "synth"
    models_dir = work_dir / "models"

    # 1. Load real vocabulary
    logger.info("Loading real vocabulary from %s", data_dir)
    category_set = load_real_vocabulary(data_dir)
    logger.info("Vocabulary: %d leaves, %d total", len(category_set.categories), len(category_set.all_categories))

    # 2. Extract value templates from real data
    logger.info("Extracting value templates")
    templates = extract_value_templates(data_dir)
    logger.info("Templates for %d codes", len(templates))

    # 3. Generate synthetic training data
    logger.info("Generating template-based synthetic data (seed=%d)", seed)
    generate_synth_tables(
        category_set,
        synth_dir,
        value_templates=templates,
        variants_per_category=variants_per_category,
        seed=seed,
    )

    # 4. Train classifiers
    logger.info("Training CatBoost + SVM classifiers")
    model_paths = train_all(
        synth_dir,
        category_set,
        models_dir,
    )

    # 5. Configure inference
    ml_inference.reset()
    ml_inference.configure_paths(
        catboost_path=model_paths["catboost"],
        svm_path=model_paths["svm"],
    )

    try:
        # 6. Load real samples
        real_samples = load_real_samples(data_dir)
        total_cols = sum(len(t.columns) for t in real_samples)
        logger.info("Real data: %d columns across %d tables", total_cols, len(real_samples))

        if mode == "bootstrap":
            return _run_real_bootstrap(
                category_set, real_samples, llm_backend,
                work_dir=work_dir,
            )

        # ML-only classification
        frame = FrameOfDiscernment(category_set)
        classifications: list[dict[str, Any]] = []
        all_sources: set[str] = set()

        for ts in real_samples:
            for col in ts.columns:
                result = _classify_column(col, category_set, frame, use_cosine=True)
                classifications.append(result)
                sources = result.get("evidence_sources", {})
                all_sources.update(sources.keys())

        # 7. Evaluate
        report = evaluate_classifications(classifications, category_set)

        # Write results
        results_dir = work_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(results_dir / "evaluation_report.json", "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        with open(results_dir / "classifications.json", "w") as f:
            json.dump(classifications, f, indent=2)

        logger.info(
            "Real data ML-only: accuracy=%.1f%%, hierarchical=%.1f%%, micro_f1=%.3f",
            report.exact_accuracy * 100,
            report.hierarchical_accuracy * 100,
            report.micro_f1,
        )

        return {
            "accuracy": report.exact_accuracy,
            "micro_f1": report.micro_f1,
            "macro_f1": report.macro_f1,
            "hierarchical_accuracy": report.hierarchical_accuracy,
            "evaluation_report": report.to_dict(),
            "models_trained": list(model_paths.keys()),
            "evidence_sources_fired": sorted(all_sources),
            "total_columns": len(classifications),
            "results_dir": str(results_dir),
        }

    finally:
        ml_inference.reset()


def _run_real_bootstrap(
    category_set,
    real_samples,
    llm_backend,
    *,
    work_dir: Path,
) -> dict[str, Any]:
    """Run bootstrap convergence on real data with injected samples."""
    from atelier.classify.pipeline import run_classification_pipeline
    from atelier.classify.fsm import AgentFSM
    from atelier.classify.mock_llm import RealisticMockLLMBackend
    from atelier.config import load_config

    cfg = load_config()
    fsm = AgentFSM()

    # Build ground truth dict for mock LLM
    if llm_backend is None:
        gt = {}
        for ts in real_samples:
            for col in ts.columns:
                if col.ground_truth:
                    gt[col.name] = col.ground_truth
        llm_backend = RealisticMockLLMBackend(
            ground_truth=gt,
            base_accuracy=0.80,
            revisit_correction_rate=0.70,
        )

    result = run_classification_pipeline(
        cfg,
        fsm,
        llm_backend=llm_backend,
        samples=real_samples,
        category_set=category_set,
    )

    # Write results
    results_dir = work_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(results_dir / "bootstrap_result.json", "w") as f:
        json.dump(
            {k: v for k, v in result.items() if not isinstance(v, (set, frozenset))},
            f, indent=2, default=str,
        )

    return result
