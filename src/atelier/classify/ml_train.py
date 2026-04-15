"""Training orchestrator for CatBoost and SVM classifiers.

Loads synthetic data from CSV + ground_truth.json, extracts features,
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
    """Load synthetic columns and ground truth from a synth directory.

    Returns:
        (columns, ground_truth) where columns = {name: [values]}
        and ground_truth = {name: category_code}.
    """
    gt_path = synth_dir / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"No ground_truth.json in {synth_dir}")

    with open(gt_path) as f:
        ground_truth: dict[str, str] = json.load(f)

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
    return columns, ground_truth


def train_svm(
    synth_dir: Path,
    output_path: Path,
) -> Path:
    """Train SVM classifier on synthetic data.

    Args:
        synth_dir: Directory with synth CSVs + ground_truth.json.
        output_path: Where to save the .pkl model file.

    Returns:
        Path to the saved model.
    """
    from atelier.classify.svm_classifier import SVMClassifier, build_svm_text

    columns, ground_truth = _load_synth_data(synth_dir)

    texts: list[str] = []
    labels: list[str] = []
    for col_name, values in columns.items():
        code = ground_truth.get(col_name)
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
    """Train SVM on blended synthetic + frontier LLM labels.

    Frontier labels come from the Opus-tier LLM sweep (label_source
    in ("llm", "llm_revisit")).  Synth data provides broad vocabulary
    coverage; frontier labels provide corpus-specific signal.

    DST independence is preserved because the SVM operates on sparse
    TF-IDF features and is trained on frontier-model (Opus) labels,
    while the LLM mass function in DST fusion uses the subagent model
    (Sonnet/Haiku).

    Args:
        state: BootstrapState with labels and label_source.
        samples_by_name: Column samples keyed by name.
        output_path: Where to save the frontier-trained SVM.
        synth_dir: Optional synth data dir for blending.
        min_frontier_labels: Minimum frontier labels to proceed.
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
            "Frontier SVM skip: %d labels < %d minimum",
            frontier_count, min_frontier_labels,
        )
        return None

    distinct_classes = len(set(frontier_labels))
    if distinct_classes < min_classes:
        logger.info(
            "Frontier SVM skip: %d classes < %d minimum",
            distinct_classes, min_classes,
        )
        return None

    # Blend with synth data for vocabulary coverage
    synth_texts: list[str] = []
    synth_labels: list[str] = []
    if synth_dir and synth_dir.exists():
        try:
            columns, ground_truth = _load_synth_data(synth_dir)
            for col_name, values in columns.items():
                code = ground_truth.get(col_name)
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
        "Frontier SVM trained: %d frontier + %d synth = %d samples, "
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
) -> Path:
    """Train CatBoost classifier on sentence-transformer embeddings.

    Args:
        synth_dir: Directory with synth CSVs + ground_truth.json.
        category_set: Used for building embedding text context.
        output_path: Where to save the .cbm model file.
        embedding_model: Sentence-transformer model name.
        iterations: CatBoost training iterations.

    Returns:
        Path to the saved model.
    """
    from atelier.classify.catboost_classifier import CatBoostColumnClassifier
    from atelier.classify.embedding import embed_texts, set_model_name
    from atelier.classify.features import extract_features

    set_model_name(embedding_model)
    columns, ground_truth = _load_synth_data(synth_dir)

    # Build embedding texts using the 12-feature extraction
    embedding_texts: list[str] = []
    labels: list[str] = []
    for col_name, values in columns.items():
        code = ground_truth.get(col_name)
        if not code:
            continue
        features = extract_features(
            column_name=col_name,
            values=values[:5],
        )
        embedding_texts.append(features.to_embedding_text())
        labels.append(code)

    logger.info("Encoding %d columns with %s", len(embedding_texts), embedding_model)
    import numpy as np
    embeddings = np.array(embed_texts(embedding_texts))

    logger.info("Training CatBoost on %d samples (%d dims)", len(labels), embeddings.shape[1])
    classifier = CatBoostColumnClassifier()
    classifier.fit(embeddings, labels, iterations=iterations)
    classifier.save(output_path)
    return output_path


def train_all(
    synth_dir: Path,
    category_set,
    models_dir: Path,
    *,
    embedding_model: str = "all-MiniLM-L6-v2",
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
    )

    return {"catboost": cb_path, "svm": svm_path}
