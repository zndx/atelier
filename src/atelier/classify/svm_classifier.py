"""SVM column classifier using TF-IDF features.

Dual TF-IDF (character n-grams + word n-grams) → LinearSVC with
CalibratedClassifierCV for Platt-scaled probability estimates.

Ported from signals/src/sigint/svm_classifier.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SVMConfig:
    """Configuration for the SVM classifier pipeline."""

    char_ngram_range: tuple[int, int] = (3, 6)
    word_ngram_range: tuple[int, int] = (1, 2)
    max_features: int = 50_000
    cv_folds: int = 5
    sublinear_tf: bool = True


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
        parts.append(", ".join(str(v) for v in sample_values[:5]))
    return " | ".join(parts)


class SVMClassifier:
    """TF-IDF + LinearSVC classifier with Platt-scaled probabilities."""

    def __init__(self, config: SVMConfig | None = None) -> None:
        self._config = config or SVMConfig()
        self._pipeline = None
        self._classes: list[str] = []

    def fit(self, texts: list[str], labels: list[str]) -> SVMClassifier:
        """Train the SVM pipeline on text inputs + category labels."""
        from scipy.sparse import hstack
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import LinearSVC

        cfg = self._config

        # Dual TF-IDF: character n-grams capture subword patterns,
        # word n-grams capture multi-word patterns
        char_tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=cfg.char_ngram_range,
            max_features=cfg.max_features,
            sublinear_tf=cfg.sublinear_tf,
        )
        word_tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=cfg.word_ngram_range,
            max_features=cfg.max_features,
            sublinear_tf=cfg.sublinear_tf,
        )

        X_char = char_tfidf.fit_transform(texts)
        X_word = word_tfidf.fit_transform(texts)
        X = hstack([X_char, X_word])

        svc = LinearSVC(class_weight="balanced", max_iter=10000, dual="auto")
        calibrated = CalibratedClassifierCV(svc, cv=cfg.cv_folds, method="sigmoid")
        calibrated.fit(X, labels)

        self._pipeline = {
            "char_tfidf": char_tfidf,
            "word_tfidf": word_tfidf,
            "classifier": calibrated,
        }
        self._classes = list(calibrated.classes_)

        logger.info(
            "SVM trained: %d samples, %d classes, %d features",
            len(texts), len(self._classes), X.shape[1],
        )
        return self

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        """Predict class probabilities for a batch of texts."""
        if self._pipeline is None:
            raise RuntimeError("Model not trained or loaded")

        from scipy.sparse import hstack

        char_tfidf = self._pipeline["char_tfidf"]
        word_tfidf = self._pipeline["word_tfidf"]
        classifier = self._pipeline["classifier"]

        X_char = char_tfidf.transform(texts)
        X_word = word_tfidf.transform(texts)
        X = hstack([X_char, X_word])

        proba_matrix = classifier.predict_proba(X)
        results = []
        for row in proba_matrix:
            results.append({
                code: float(prob)
                for code, prob in zip(self._classes, row)
                if prob > 1e-6
            })
        return results

    def predict_proba_single(self, text: str) -> dict[str, float]:
        """Predict for a single text input."""
        return self.predict_proba([text])[0]

    def save(self, path: str | Path) -> None:
        """Save model to disk (joblib + classes JSON)."""
        if self._pipeline is None:
            raise RuntimeError("No model to save")

        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, str(path))

        classes_path = path.with_suffix(".classes.json")
        with open(classes_path, "w") as f:
            json.dump(self._classes, f)

        logger.info("SVM saved to %s (%d classes)", path, len(self._classes))

    @classmethod
    def load(cls, path: str | Path) -> SVMClassifier:
        """Load a saved SVM model from disk."""
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SVM model not found: {path}")

        instance = cls()
        instance._pipeline = joblib.load(str(path))

        classes_path = path.with_suffix(".classes.json")
        if classes_path.exists():
            with open(classes_path) as f:
                instance._classes = json.load(f)
        else:
            instance._classes = list(instance._pipeline["classifier"].classes_)

        logger.info("SVM loaded from %s (%d classes)", path, len(instance._classes))
        return instance
