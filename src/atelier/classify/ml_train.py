# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Training orchestrator for CatBoost and SVM classifiers.

Loads synthetic data from CSV + reference_labels.json, extracts features,
and trains both classifiers. Output: model files in build/models/.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_synth_data(synth_dir: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Load synthetic columns and reference labels from a synth directory.

    Returns:
        (columns, reference_labels) where columns = {name: [values]}
        and reference_labels = {name: category_code}.
    """
    ref_path = synth_dir / "reference_labels.json"
    if not ref_path.exists():
        raise FileNotFoundError(f"No reference_labels.json in {synth_dir}")

    with open(ref_path) as f:
        reference_labels: dict[str, str] = json.load(f)

    columns: dict[str, list[str]] = {}
    for csv_path in sorted(synth_dir.glob("synth_*.csv")):
        with open(csv_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            col_data: dict[str, list[str]] = {name: [] for name in header}
            for row in reader:
                for name, val in zip(header, row):
                    col_data[name].append(val)
            columns.update(col_data)

    logger.info("Loaded %d columns from %s", len(columns), synth_dir)
    return columns, reference_labels


def train_svm(
    synth_dir: Path,
    output_path: Path,
) -> Path:
    """Train SVM classifier on synthetic data.

    Args:
        synth_dir: Directory with synth CSVs + reference_labels.json.
        output_path: Where to save the .pkl model file.

    Returns:
        Path to the saved model.
    """
    from atelier.classify.svm_classifier import SVMClassifier, build_svm_text

    columns, reference_labels = _load_synth_data(synth_dir)

    texts: list[str] = []
    labels: list[str] = []
    for col_name, values in columns.items():
        code = reference_labels.get(col_name)
        if not code:
            continue
        text = build_svm_text(col_name, sample_values=values[:5])
        texts.append(text)
        labels.append(code)

    logger.info("Training SVM on %d samples", len(texts))
    classifier = SVMClassifier()
    classifier.fit(texts, labels)
    classifier.save(output_path)
    return output_path


def train_svm_on_frontier_labels(
    state,
    samples_by_name: dict[str, Any],
    output_path: Path,
    *,
    synth_dir: Path | None = None,
    min_frontier_labels: int = 20,
    min_classes: int = 3,
) -> Path | None:
    """Train the incremental SVM on blended synthetic + frontier-tier LLM labels.

    The "incremental" half names the SVM's role in the active-learning
    bootstrap loop — it is retrained as new oracle labels accumulate.
    "Frontier-tier" names the *source* of those labels: the Opus-class
    LLM that runs the sweep + revisit (``label_source in ("llm",
    "llm_revisit")``).  Synth data provides broad vocabulary coverage;
    frontier-tier labels provide corpus-specific signal.

    DST independence is preserved because the SVM operates on sparse
    TF-IDF features and is trained on frontier-tier (Opus) labels,
    while the LLM mass function in DST fusion uses the subagent model
    (Sonnet/Haiku).

    Args:
        state: BootstrapState with labels and label_source.
        samples_by_name: Column samples keyed by name.
        output_path: Where to save the incremental SVM.
        synth_dir: Optional synth data dir for blending.
        min_frontier_labels: Minimum frontier-tier labels to proceed.
        min_classes: Minimum distinct classes to proceed.

    Returns:
        Path to saved model, or None if thresholds not met.
    """
    import time
    from atelier.classify.svm_classifier import SVMClassifier, build_svm_text

    # Collect frontier labels (LLM-sourced, not propagated)
    frontier_texts: list[str] = []
    frontier_labels: list[str] = []
    for name, code in state.labels.items():
        source = state.label_source.get(name, "")
        if source not in ("llm", "llm_revisit"):
            continue
        col = samples_by_name.get(name)
        if col is None:
            continue
        text = build_svm_text(col.name, col.column_type, col.values[:5])
        frontier_texts.append(text)
        frontier_labels.append(code)

    frontier_count = len(frontier_texts)
    if frontier_count < min_frontier_labels:
        logger.info(
            "Incremental SVM skip: %d labels < %d minimum",
            frontier_count, min_frontier_labels,
        )
        return None

    distinct_classes = len(set(frontier_labels))
    if distinct_classes < min_classes:
        logger.info(
            "Incremental SVM skip: %d classes < %d minimum",
            distinct_classes, min_classes,
        )
        return None

    # Blend with synth data for vocabulary coverage
    synth_texts: list[str] = []
    synth_labels: list[str] = []
    if synth_dir and synth_dir.exists():
        try:
            columns, reference_labels = _load_synth_data(synth_dir)
            for col_name, values in columns.items():
                code = reference_labels.get(col_name)
                if not code:
                    continue
                text = build_svm_text(col_name, sample_values=values[:5])
                synth_texts.append(text)
                synth_labels.append(code)
        except Exception as e:
            logger.warning("Failed to load synth data for blending: %s", e)

    texts = synth_texts + frontier_texts
    labels = synth_labels + frontier_labels

    t0 = time.monotonic()
    classifier = SVMClassifier()
    classifier.fit(texts, labels)
    classifier.save(output_path)
    elapsed = time.monotonic() - t0

    logger.info(
        "Incremental SVM trained: %d frontier-tier + %d synth = %d samples, "
        "%d classes, %.1fs → %s",
        frontier_count, len(synth_texts), len(texts),
        len(set(labels)), elapsed, output_path,
    )
    return output_path


def train_catboost(
    synth_dir: Path,
    category_set,
    output_path: Path,
    *,
    embedding_model: str = "all-MiniLM-L6-v2",
    iterations: int = 1000,
    depth: int = 6,
    learning_rate: float = 0.10,
) -> Path:
    """Train CatBoost on the structured per-feature input matrix.

    Each column produces a ``ColumnFeatures``; the classifier encodes
    one SentenceTransformer slice per text feature + scalar slots per
    numeric feature internally, so TreeSHAP attributes natively per
    source feature at inference / explanation time.  See
    :class:`atelier.classify.catboost_classifier.CatBoostColumnClassifier`.

    Args:
        synth_dir: Directory with synth CSVs + reference_labels.json.
        category_set: Reserved for future per-source augmentation
            (unused today; kept for caller-API stability).
        output_path: Where to save the .cbm model file.
        embedding_model: Sentence-transformer model name (encoder for
            the text-shaped feature slices).
        iterations: CatBoost boosting rounds.
        depth: CatBoost tree depth.
        learning_rate: CatBoost learning rate (shrinkage per round).

    Returns:
        Path to the saved model.
    """
    from atelier.classify.catboost_classifier import CatBoostColumnClassifier
    from atelier.classify.embedding import set_model_name
    from atelier.classify.features import extract_features

    set_model_name(embedding_model)
    columns, reference_labels = _load_synth_data(synth_dir)

    # Build ColumnFeatures from each labeled column.
    features_list = []
    labels: list[str] = []
    for col_name, values in columns.items():
        code = reference_labels.get(col_name)
        if not code:
            continue
        features_list.append(extract_features(
            column_name=col_name,
            values=values[:5],
        ))
        labels.append(code)

    logger.info(
        "Training CatBoost on %d samples (encoder=%s, structured per-feature input)",
        len(labels), embedding_model,
    )
    classifier = CatBoostColumnClassifier()
    classifier.fit(
        features_list, labels,
        iterations=iterations, depth=depth, learning_rate=learning_rate,
    )
    classifier.save(output_path)
    return output_path


def fit_catboost_to_llm_labels(
    features_list: list,
    llm_codes: list[str],
    *,
    iterations: int = 1000,
    depth: int = 6,
    learning_rate: float = 0.10,
):
    """Fit an in-memory CatBoost on ``(ColumnFeatures, llm_predicted_code)`` pairs.

    Used by the pipeline's fit-to-LLM mode (REVEAL pattern): after the
    LLM sweep labels the corpus, we fit CatBoost to **agree** with the
    LLM on the columns it labeled.  CatBoost then generalizes to
    columns held out from the LLM pass — same vocabulary, same
    feature space, no oracle dependency at inference time.

    Because CatBoost is trained on the **structured per-feature input**
    (one SentenceTransformer slice per text feature + scalar slots per
    numeric), TreeSHAP attribution at evaluation time attributes
    natively per source feature.  This makes CatBoost the genuine
    explainability surface for the LLM's labeling decisions:
    "predicted code ``X`` because the column_name contributed +0.4
    SHAP, sample_values contributed +0.3, pattern_signals contributed
    +0.2..."

    Args:
        features_list: ``list[ColumnFeatures]`` — one per LLM-labeled
            column.  Length must match ``llm_codes``.
        llm_codes: The LLM's predicted code per column.
        iterations / depth / learning_rate: CatBoost hyperparameters.

    Returns the trained :class:`CatBoostColumnClassifier`, or None when
    input is insufficient (``< 2`` distinct classes or fewer than 10
    samples).  The caller decides how to react to None.
    """
    from atelier.classify.catboost_classifier import CatBoostColumnClassifier

    if len(features_list) != len(llm_codes):
        raise ValueError(
            f"len mismatch: {len(features_list)} features vs {len(llm_codes)} codes"
        )
    if len(features_list) < 10:
        logger.info(
            "fit_to_llm: only %d labels — skipping (need >= 10)",
            len(features_list),
        )
        return None

    pairs = [
        (f, c) for f, c in zip(features_list, llm_codes)
        if f is not None and c
    ]
    if len({c for _, c in pairs}) < 2:
        logger.info("fit_to_llm: only one distinct class — skipping")
        return None

    feats = [f for f, _ in pairs]
    codes = [c for _, c in pairs]

    logger.info(
        "fit_to_llm: training CatBoost (%d samples, %d classes, iter=%d, "
        "structured per-feature input)",
        len(codes), len(set(codes)), iterations,
    )
    classifier = CatBoostColumnClassifier()
    classifier.fit(
        feats, codes,
        iterations=iterations, depth=depth, learning_rate=learning_rate,
    )
    return classifier


def train_all(
    synth_dir: Path,
    category_set,
    models_dir: Path,
    *,
    embedding_model: str = "all-MiniLM-L6-v2",
    catboost_iterations: int = 1000,
    catboost_depth: int = 6,
    catboost_learning_rate: float = 0.10,
) -> dict[str, Path]:
    """Train both CatBoost and SVM classifiers.

    Returns:
        {"catboost": Path, "svm": Path} of saved model files.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    svm_path = train_svm(synth_dir, models_dir / "svm.pkl")
    cb_path = train_catboost(
        synth_dir, category_set, models_dir / "catboost.cbm",
        embedding_model=embedding_model,
        iterations=catboost_iterations,
        depth=catboost_depth,
        learning_rate=catboost_learning_rate,
    )

    return {"catboost": cb_path, "svm": svm_path}
