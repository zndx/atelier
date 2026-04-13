"""CatBoost column classifier using sentence-transformer embeddings.

Trains a CatBoostClassifier on 384-dim embeddings from all-MiniLM-L6-v2.
Uses posterior_sampling for virtual ensemble uncertainty quantification.

Ported from signals/src/sigint/embedding_classifier.py training logic.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CatBoostColumnClassifier:
    """CatBoost classifier over sentence-transformer embeddings."""

    def __init__(self) -> None:
        self._model = None
        self._classes: list[str] = []

    def fit(
        self,
        embeddings,  # numpy ndarray (N, 384)
        labels: list[str],
        *,
        iterations: int = 1000,
        depth: int = 6,
        learning_rate: float = 0.1,
        verbose: int = 0,
    ) -> CatBoostColumnClassifier:
        """Train CatBoost on pre-computed embeddings."""
        from catboost import CatBoostClassifier, Pool

        # Deduplicate classes preserving order
        seen: set[str] = set()
        unique_classes = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                unique_classes.append(label)
        self._classes = sorted(unique_classes)

        pool = Pool(data=embeddings, label=labels)

        self._model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function="MultiClass",
            posterior_sampling=True,
            auto_class_weights="Balanced",
            verbose=verbose,
            random_seed=42,
        )
        self._model.fit(pool)

        logger.info(
            "CatBoost trained: %d samples, %d classes, %d dims",
            len(labels), len(self._classes), embeddings.shape[1],
        )
        return self

    def predict_proba(self, embeddings) -> list[dict[str, float]]:
        """Predict class probabilities for a batch of embeddings."""
        if self._model is None:
            raise RuntimeError("Model not trained or loaded")

        proba_matrix = self._model.predict_proba(embeddings)
        model_classes = list(self._model.classes_)

        results = []
        for row in proba_matrix:
            results.append({
                str(code): float(prob)
                for code, prob in zip(model_classes, row)
                if prob > 1e-6
            })
        return results

    def predict_proba_single(self, embedding) -> dict[str, float]:
        """Predict for a single embedding vector."""
        import numpy as np
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        return self.predict_proba(embedding)[0]

    def virtual_ensemble_variance(self, embeddings) -> list[dict[str, float]]:
        """Get per-class variance from CatBoost virtual ensembles.

        Requires the model to be trained with posterior_sampling=True.
        """
        if self._model is None:
            raise RuntimeError("Model not trained or loaded")

        try:
            ve_preds = self._model.virtual_ensembles_predict(
                embeddings, prediction_type="TotalUncertainty",
            )
        except Exception:
            # Fallback: return empty variance dicts
            return [{} for _ in range(len(embeddings))]

        model_classes = list(self._model.classes_)
        n_classes = len(model_classes)
        results = []

        for row in ve_preds:
            # VirtualEnsembles returns [mean_0..mean_n, var_0..var_n, ...]
            # Variance values start at index n_classes
            variance = {}
            for i, code in enumerate(model_classes):
                var_idx = n_classes + i
                if var_idx < len(row):
                    variance[str(code)] = float(row[var_idx])
            results.append(variance)

        return results

    def save(self, path: str | Path) -> None:
        """Save model to CatBoost native format + classes JSON."""
        if self._model is None:
            raise RuntimeError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))

        classes_path = path.with_suffix(".classes.json")
        with open(classes_path, "w") as f:
            json.dump(self._classes, f)

        logger.info("CatBoost saved to %s (%d classes)", path, len(self._classes))

    @classmethod
    def load(cls, path: str | Path) -> CatBoostColumnClassifier:
        """Load a saved CatBoost model from disk."""
        from catboost import CatBoostClassifier

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CatBoost model not found: {path}")

        instance = cls()
        instance._model = CatBoostClassifier()
        instance._model.load_model(str(path))

        classes_path = path.with_suffix(".classes.json")
        if classes_path.exists():
            with open(classes_path) as f:
                instance._classes = json.load(f)
        else:
            instance._classes = [str(c) for c in instance._model.classes_]

        logger.info("CatBoost loaded from %s (%d classes)", path, len(instance._classes))
        return instance
