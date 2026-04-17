#!/usr/bin/env python3
"""Train CatBoost and SVM classifiers on synthetic data.

Usage:
    uv run python scripts/train_classifiers.py
    uv run python scripts/train_classifiers.py --synth-dir build/data/synth/ --models-dir build/models/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ML classifiers on synthetic data")
    parser.add_argument(
        "--synth-dir", type=Path, default=Path("build/data/synth"),
        help="Directory containing synth CSVs + ground_truth.json",
    )
    parser.add_argument(
        "--models-dir", type=Path, default=Path("build/models"),
        help="Output directory for model files",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate synthetic data first if it doesn't exist",
    )
    args = parser.parse_args()

    # Generate synth data if needed
    if args.generate or not (args.synth_dir / "ground_truth.json").exists():
        logger.info("Generating synthetic data in %s", args.synth_dir)
        from atelier.classify.synth import generate_synth_tables
        from atelier.classify.taxonomy import HierarchicalCategorySet, load_universal_vocabulary

        cats = load_universal_vocabulary(hierarchical=True)
        result = generate_synth_tables(cats, args.synth_dir)
        total_cols = sum(t["column_count"] for t in result)
        logger.info("Generated %d columns in %d files", total_cols, len(result))

    if not (args.synth_dir / "ground_truth.json").exists():
        logger.error("No ground_truth.json found in %s", args.synth_dir)
        return 1

    from atelier.classify.taxonomy import HierarchicalCategorySet, load_universal_vocabulary
    from atelier.classify.ml_train import train_all
    from atelier.config import load_config
    from atelier.config_overlay import apply_to_config

    cfg = apply_to_config(load_config())
    cats = load_universal_vocabulary(hierarchical=True)
    paths = train_all(
        args.synth_dir, cats, args.models_dir,
        embedding_model=cfg.classify_embedding_model,
        catboost_iterations=cfg.classify_catboost_iterations,
        catboost_depth=cfg.classify_catboost_depth,
        catboost_learning_rate=cfg.classify_catboost_learning_rate,
    )

    for name, path in paths.items():
        logger.info("Saved %s → %s", name, path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
