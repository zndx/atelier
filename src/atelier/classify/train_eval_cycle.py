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
