# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""SVM short-text classifier for column metadata.

Implements a TF-IDF + LinearSVC pipeline as a DST evidence source.
Uses character n-grams (3-6) and word bigrams over column name + sample
value text, producing calibrated probabilities via Platt scaling
(CalibratedClassifierCV).

This classifier is architecturally independent from the sentence-transformer
embedding used by cosine and CatBoost sources — it operates on sparse lexical
features (TF-IDF), providing genuine evidence diversity for Dempster-Shafer
combination.

Design follows the LibShortText principles (Yu et al., 2013) using modern
scikit-learn equivalents rather than the unmaintained original library.

Adopted from signals/src/sigint/svm_classifier.py — the version of record
presented as an independent fifth DST evidence source.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SVMConfig:
    """Configuration for the SVM short-text classifier."""

    # TF-IDF character n-gram range (inclusive)
    char_ngram_min: int = 3
    char_ngram_max: int = 6

    # TF-IDF word n-gram range (inclusive)
    word_ngram_min: int = 1
    word_ngram_max: int = 2

    # Maximum features per vectorizer
    max_features: int = 50_000

    # LinearSVC regularization
    svc_C: float = 1.0

    # Calibration method: "sigmoid" (Platt) or "isotonic"
    calibration_method: str = "sigmoid"

    # Number of CV folds for calibration
    calibration_cv: int = 5


def build_svm_text(
    column_name: str,
    column_type: str | None = None,
    sample_values: list[str] | None = None,
) -> str:
    """Build short text for SVM from column metadata.

    Format: "column_name | column_type | val1, val2, val3"
    """
    parts = [column_name.replace("_", " ")]
    if column_type and column_type.upper() not in ("STRING", "VARCHAR"):
        parts.append(column_type)
    if sample_values:
        parts.append(", ".join(str(v)[:80] for v in sample_values[:5]))
    return " | ".join(parts)


class SVMClassifier:
    """TF-IDF + LinearSVC classifier with calibrated probabilities.

    Combines character n-gram and word n-gram TF-IDF features into a
    single sparse feature matrix, then trains a LinearSVC with Platt
    scaling for probability output.
    """

    def __init__(self, config: SVMConfig | None = None) -> None:
        self._config = config or SVMConfig()
        self._pipeline = None
        self._classes: list[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._pipeline is not None

    def fit(self, texts: list[str], labels: list[str]) -> SVMClassifier:
        """Train the SVM pipeline on labeled short texts.

        Args:
            texts: Column metadata text (e.g., "column_name | values").
            labels: Category codes (e.g., "0070", "0076").

        Returns:
            self, for method chaining.
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.svm import LinearSVC

        from collections import Counter

        cfg = self._config

        # Drop singleton classes — StratifiedKFold requires every class
        # to have >= 2 samples.  With many categories and few tables,
        # some categories will inevitably have only one example.
        min_count = self._min_class_count(labels)
        if min_count < 2:
            counts = Counter(labels)
            singletons = {code for code, n in counts.items() if n < 2}
            logger.warning(
                "SVM: dropping %d singleton classes (< 2 examples): %s",
                len(singletons),
                sorted(singletons)[:20],  # log first 20 to avoid spam
            )
            filtered = [
                (t, l) for t, l in zip(texts, labels)
                if l not in singletons
            ]
            if not filtered:
                raise ValueError(
                    "No classes with >= 2 examples — cannot train SVM"
                )
            texts, labels = [t for t, _ in filtered], [l for _, l in filtered]
            min_count = self._min_class_count(labels)

        # Character n-gram vectorizer — captures subword patterns
        # (abbreviations, camelCase fragments, digit sequences)
        char_tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(cfg.char_ngram_min, cfg.char_ngram_max),
            max_features=cfg.max_features,
            sublinear_tf=True,
        )

        # Word n-gram vectorizer — captures multi-word patterns
        # ("payment card", "email address")
        word_tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(cfg.word_ngram_min, cfg.word_ngram_max),
            max_features=cfg.max_features,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )

        # Union of both feature spaces
        features = FeatureUnion([
            ("char", char_tfidf),
            ("word", word_tfidf),
        ])

        # LinearSVC — maximum-margin classifier on sparse features
        svc = LinearSVC(
            C=cfg.svc_C,
            max_iter=10_000,
            class_weight="balanced",
            dual="auto",
        )

        # Wrap in CalibratedClassifierCV for probability estimates
        calibrated = CalibratedClassifierCV(
            svc,
            cv=min(cfg.calibration_cv, min_count),
            method=cfg.calibration_method,
        )

        self._pipeline = Pipeline([
            ("features", features),
            ("classifier", calibrated),
        ])
        self._pipeline.fit(texts, labels)
        self._classes = list(self._pipeline.classes_)

        logger.info(
            "SVM trained: %d samples, %d classes",
            len(texts), len(self._classes),
        )
        return self

    @staticmethod
    def _min_class_count(labels: list[str]) -> int:
        """Minimum number of samples in any class."""
        from collections import Counter
        counts = Counter(labels)
        return min(counts.values())

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        """Return calibrated probability distributions for each text.

        Args:
            texts: Column metadata texts to classify.

        Returns:
            List of {category_code: probability} dicts.
        """
        if not self.is_fitted:
            raise RuntimeError("SVMClassifier must be fitted before prediction")

        proba_matrix = self._pipeline.predict_proba(texts)
        results = []
        for row in proba_matrix:
            prob_dict = {
                code: float(p)
                for code, p in zip(self._classes, row)
                if p > 1e-6
            }
            results.append(prob_dict)
        return results

    def predict_proba_single(self, text: str) -> dict[str, float]:
        """Return calibrated probability distribution for a single text."""
        return self.predict_proba([text])[0]

    def save(self, path: str | Path) -> None:
        """Persist the trained pipeline to disk.

        Saves:
          - {path}.pkl — sklearn pipeline (joblib)
          - {path}.classes.json — class label mapping
        """
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, str(path))

        classes_path = path.with_suffix(".classes.json")
        classes_path.write_text(json.dumps(self._classes))

        logger.info("SVM saved to %s (%d classes)", path, len(self._classes))

    @classmethod
    def load(cls, path: str | Path, config: SVMConfig | None = None) -> SVMClassifier:
        """Load a persisted SVM pipeline from disk."""
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SVM model not found: {path}")

        obj = cls(config=config)
        obj._pipeline = joblib.load(str(path))
        obj._classes = list(obj._pipeline.classes_)

        # Also try classes JSON for consistency
        classes_path = path.with_suffix(".classes.json")
        if classes_path.exists():
            obj._classes = json.loads(classes_path.read_text())

        logger.info("SVM loaded from %s (%d classes)", path, len(obj._classes))
        return obj

    def feature_importances(self, top_n: int = 20) -> list[tuple[str, float]]:
        """Extract top feature names by absolute SVM weight.

        Only works before calibration wrapping. Returns empty list
        if the underlying estimator is wrapped.
        """
        try:
            # Navigate through CalibratedClassifierCV → LinearSVC
            calibrated = self._pipeline.named_steps["classifier"]
            svc = calibrated.estimator
            if not hasattr(svc, "coef_"):
                return []

            # For multi-class, average absolute coefficients across classes
            coef = np.abs(svc.coef_).mean(axis=0)

            feature_union = self._pipeline.named_steps["features"]
            names = feature_union.get_feature_names_out()

            indices = np.argsort(coef)[::-1][:top_n]
            return [(str(names[i]), float(coef[i])) for i in indices]
        except Exception:
            return []
