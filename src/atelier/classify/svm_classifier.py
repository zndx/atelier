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

    # SVD components for NHSVM Kronecker expansion
    nhsvm_svd_components: int = 200


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


def build_nhsvm_distance_matrix(
    codes: list[str],
    alphas: dict[str, float],
    category_set,
) -> dict[tuple[str, str], float]:
    """Precompute pairwise normalized tree distances for a set of codes.

    Call once per taxonomy + code set (e.g. at pipeline start), then
    pass the result to ``nhsvm_reweight`` via ``distance_matrix=``.
    """
    import math

    ancestor_sets: dict[str, set[str]] = {}
    for c in codes:
        ancestor_sets[c] = set(category_set.ancestors(c)) | {c}

    matrix: dict[tuple[str, str], float] = {}
    for i, a in enumerate(codes):
        for b in codes[i + 1:]:
            sym = (ancestor_sets[a] ^ ancestor_sets[b]) - {None}
            d = math.sqrt(max(sum(alphas.get(n, 0.0) for n in sym), 0.0))
            matrix[(a, b)] = d
            matrix[(b, a)] = d
    return matrix


def nhsvm_reweight(
    proba: dict[str, float],
    alphas: dict[str, float],
    category_set,
    temperature: float = 1.0,
    distance_matrix: dict[tuple[str, str], float] | None = None,
) -> dict[str, float]:
    """Reweight flat SVM probabilities using NHSVM tree-distance normalization.

    For each candidate code, computes its expected normalized tree
    distance to the rest of the probability distribution (Eq. 9,
    Choi et al. 2015).  Codes that are structurally isolated from
    the bulk of the probability mass — like a shallow catch-all
    (INOS) when most mass sits in a deep sensitive subtree — receive
    a higher distance penalty and lose probability to structurally
    coherent alternatives.

    Pass a precomputed ``distance_matrix`` (from
    ``build_nhsvm_distance_matrix``) to avoid recomputing ancestor
    sets on every call.
    """
    import math

    if not proba or not alphas:
        return proba
    codes = list(proba.keys())
    if len(codes) < 2:
        return proba

    if distance_matrix is not None:
        def _dist(a: str, b: str) -> float:
            return distance_matrix.get((a, b), 0.0)
    else:
        ancestor_sets: dict[str, set[str]] = {}
        for c in codes:
            ancestor_sets[c] = set(category_set.ancestors(c)) | {c}

        def _dist(a: str, b: str) -> float:
            sym = (ancestor_sets[a] ^ ancestor_sets[b]) - {None}
            return math.sqrt(max(sum(alphas.get(n, 0.0) for n in sym), 0.0))

    weighted_dist: dict[str, float] = {}
    for c in codes:
        wd = 0.0
        for other in codes:
            if other != c:
                wd += proba[other] * _dist(c, other)
        weighted_dist[c] = wd

    reweighted: dict[str, float] = {}
    for c in codes:
        reweighted[c] = proba[c] * math.exp(-weighted_dist[c] / temperature)

    total = sum(reweighted.values())
    if total > 0:
        reweighted = {c: v / total for c, v in reweighted.items()}
    return reweighted


class HierarchicalFeatureExpander:
    """Kronecker product feature expansion for NHSVM (Choi et al. 2015, Eq. 5).

    Expands a d-dimensional feature vector into d×M dimensions by
    replicating it into per-node blocks scaled by ``sqrt(alpha_n)``.
    Standard L2 regularization on the expanded space equals the
    Structured Shared Frobenius Norm on the original space.
    """

    def __init__(
        self,
        alphas: dict[str, float],
        paths: dict[str, list[str]],
        node_index: dict[str, int],
        n_features_in: int,
    ) -> None:
        self.alphas = alphas
        self.paths = paths
        self.node_index = node_index
        self.n_nodes = len(node_index)
        self.n_features_in = n_features_in
        self._sqrt_alphas = {code: np.sqrt(a) for code, a in alphas.items()}

    def expand_with_labels(self, X: np.ndarray, labels: list[str]):
        """Training-time expansion: populate only the label's path blocks."""
        from scipy.sparse import csr_matrix

        n_samples, d = X.shape
        n_expanded = d * self.n_nodes

        rows, cols, data = [], [], []
        for i in range(n_samples):
            path = self.paths.get(labels[i], [])
            for node in path:
                idx = self.node_index.get(node)
                if idx is None:
                    continue
                scale = self._sqrt_alphas.get(node, 0.0)
                if scale <= 0:
                    continue
                block_start = idx * d
                for j in range(d):
                    val = X[i, j] * scale
                    if abs(val) > 1e-12:
                        rows.append(i)
                        cols.append(block_start + j)
                        data.append(val)

        return csr_matrix(
            (np.array(data, dtype=np.float64),
             (np.array(rows, dtype=np.int32),
              np.array(cols, dtype=np.int32))),
            shape=(n_samples, n_expanded),
        )

    def expand_universal(self, X: np.ndarray):
        """Inference-time expansion: populate all node blocks."""
        from scipy.sparse import csr_matrix

        n_samples, d = X.shape
        n_expanded = d * self.n_nodes

        rows, cols, data = [], [], []
        for i in range(n_samples):
            for code, idx in self.node_index.items():
                scale = self._sqrt_alphas.get(code, 0.0)
                if scale <= 0:
                    continue
                block_start = idx * d
                for j in range(d):
                    val = X[i, j] * scale
                    if abs(val) > 1e-12:
                        rows.append(i)
                        cols.append(block_start + j)
                        data.append(val)

        return csr_matrix(
            (np.array(data, dtype=np.float64),
             (np.array(rows, dtype=np.int32),
              np.array(cols, dtype=np.int32))),
            shape=(n_samples, n_expanded),
        )

    @classmethod
    def from_category_set(cls, category_set, alphas: dict[str, float], n_features_in: int):
        node_index = {cat.code: i for i, cat in enumerate(category_set.all_categories)}
        paths = {cat.code: category_set.path_from_root(cat.code) for cat in category_set.all_categories}
        return cls(alphas, paths, node_index, n_features_in)


class SVMClassifier:
    """TF-IDF + LinearSVC classifier with calibrated probabilities.

    Combines character n-gram and word n-gram TF-IDF features into a
    single sparse feature matrix, then trains a LinearSVC with Platt
    scaling for probability output.

    When ``hierarchical=True`` and a ``category_set`` is provided,
    trains with the Structured Shared Frobenius Norm (Choi et al. 2015):
    TF-IDF → SVD → Kronecker expansion → LinearSVC.  The hierarchy is
    baked into training, not applied post-hoc.
    """

    def __init__(
        self,
        config: SVMConfig | None = None,
        category_set=None,
        hierarchical: bool = False,
    ) -> None:
        self._config = config or SVMConfig()
        self._pipeline = None
        self._classes: list[str] = []
        self._category_set = category_set
        self._hierarchical = hierarchical and category_set is not None
        self._feature_union = None
        self._svd = None
        self._expander: HierarchicalFeatureExpander | None = None

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
        from collections import Counter

        cfg = self._config

        min_count = self._min_class_count(labels)
        if min_count < 2:
            counts = Counter(labels)
            singletons = {code for code, n in counts.items() if n < 2}
            logger.warning(
                "SVM: dropping %d singleton classes (< 2 examples): %s",
                len(singletons),
                sorted(singletons)[:20],
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

        if self._hierarchical:
            return self._fit_hierarchical(texts, labels, min_count)
        return self._fit_flat(texts, labels, min_count)

    def _build_feature_union(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        cfg = self._config
        char_tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(cfg.char_ngram_min, cfg.char_ngram_max),
            max_features=cfg.max_features,
            sublinear_tf=True,
        )
        word_tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(cfg.word_ngram_min, cfg.word_ngram_max),
            max_features=cfg.max_features,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        return FeatureUnion([("char", char_tfidf), ("word", word_tfidf)])

    def _fit_flat(self, texts, labels, min_count) -> SVMClassifier:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC

        cfg = self._config
        features = self._build_feature_union()
        svc = LinearSVC(
            C=cfg.svc_C, max_iter=10_000,
            class_weight="balanced", dual="auto",
        )
        calibrated = CalibratedClassifierCV(
            svc, cv=min(cfg.calibration_cv, min_count),
            method=cfg.calibration_method,
        )
        self._pipeline = Pipeline([
            ("features", features),
            ("classifier", calibrated),
        ])
        self._pipeline.fit(texts, labels)
        self._classes = list(self._pipeline.classes_)
        logger.info("SVM (flat) trained: %d samples, %d classes",
                     len(texts), len(self._classes))
        return self

    def _fit_hierarchical(self, texts, labels, min_count) -> SVMClassifier:
        """Train training-time NHSVM per Choi et al. (2015) Eq. 5.

        Crammer-Singer joint multi-class LinearSVC on Kronecker-expanded
        features.  Joint training optimizes per-class weights against the
        structured multi-class margin — closer to the structured-output
        objective the paper specifies than one-vs-rest training.  Paired
        with per-class expansion at inference (``_predict_proba_hierarchical``)
        for the geometry the paper's ``argmax_y <w, Lambda(y) ⊗ x>`` rule
        implies.

        Roll-forward posture: this is the single hierarchical training
        path.  No one-vs-rest fallback.  Operators tuning the previous
        OvR-trained models will hit a fresh cache miss on the new
        ``_nhsvm_cs`` suffix and retrain into the Crammer-Singer regime.
        """
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.decomposition import TruncatedSVD
        from sklearn.svm import LinearSVC

        cfg = self._config
        cs = self._category_set
        svd_components = getattr(cfg, "nhsvm_svd_components", None) or 200

        # Step 1: TF-IDF
        self._feature_union = self._build_feature_union()
        X_tfidf = self._feature_union.fit_transform(texts)
        logger.info("NHSVM (Crammer-Singer): TF-IDF shape %s", X_tfidf.shape)

        # Step 2: SVD reduction
        n_components = min(svd_components, X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
        if n_components < svd_components:
            logger.warning(
                "NHSVM: SVD components clamped to %d (requested %d) — "
                "small corpus or vocabulary; per-block features will be "
                "lower-dimensional than configured.",
                n_components, svd_components,
            )
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_reduced = self._svd.fit_transform(X_tfidf)
        logger.info("NHSVM: SVD reduced to %d components (%.1f%% variance)",
                     n_components, self._svd.explained_variance_ratio_.sum() * 100)

        # Step 3: Directionally-constrained alphas (Choi Eq. 7)
        alphas = cs.compute_nhsvm_alphas()

        # Step 4: Label-conditional Kronecker expansion (Choi Eq. 5)
        self._expander = HierarchicalFeatureExpander.from_category_set(
            cs, alphas, n_features_in=n_components,
        )
        X_expanded = self._expander.expand_with_labels(X_reduced, labels)
        logger.info("NHSVM: expanded shape %s, nnz %d",
                     X_expanded.shape, X_expanded.nnz)

        # Step 5: Crammer-Singer requires dense data.  Convert.
        X_dense = X_expanded.toarray() if hasattr(X_expanded, "toarray") else X_expanded

        # Step 6: Crammer-Singer joint multi-class LinearSVC + calibration.
        # The CS objective requires loss="hinge", dual=True, and dense
        # input.  Calibration follows the standard sigmoid-on-decision-
        # function path (LinearSVC doesn't expose predict_proba natively).
        svc = LinearSVC(
            C=cfg.svc_C, max_iter=20_000,
            multi_class="crammer_singer",
            loss="hinge",
            dual=True,
            class_weight="balanced",
        )
        calibrated = CalibratedClassifierCV(
            svc, cv=min(cfg.calibration_cv, min_count),
            method=cfg.calibration_method, ensemble=False,
        )
        calibrated.fit(X_dense, labels)

        self._pipeline = calibrated
        self._classes = list(calibrated.classes_)
        logger.info(
            "NHSVM (Crammer-Singer) trained: %d samples, %d classes, "
            "%d expanded features (dense)",
            len(texts), len(self._classes), X_expanded.shape[1],
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

        if self._hierarchical and self._feature_union is not None:
            return self._predict_proba_hierarchical(texts)

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

    def _predict_proba_hierarchical(self, texts: list[str]) -> list[dict[str, float]]:
        """Per-class inference expansion (Choi Eq. 5 ``argmax_y`` rule).

        For each candidate ``y`` the inference feature is built with only
        ``path(y)`` blocks active, scaled by ``sqrt(alpha_n)`` — matching
        the training-time expansion shape for label ``y``.  The model's
        ``p_y`` under that y-specific expansion is the model's answer to
        "if y were the right label, how confident would you be?", and we
        normalize across the candidate set to produce a proper
        distribution.

        This is the inference geometry the paper's structured-output
        rule implies, paired with the Crammer-Singer joint training in
        ``_fit_hierarchical``.  No universal-expansion fallback — the
        roll-forward posture is single-path.
        """
        X_tfidf = self._feature_union.transform(texts)
        X_reduced = self._svd.transform(X_tfidf)

        class_to_index = {c: i for i, c in enumerate(self._classes)}
        results: list[dict[str, float]] = []
        for sample_idx in range(X_reduced.shape[0]):
            X_one = X_reduced[sample_idx:sample_idx + 1]
            raw_p_y: dict[str, float] = {}
            for y in self._classes:
                X_y = self._expander.expand_with_labels(X_one, [y])
                X_y_dense = X_y.toarray() if hasattr(X_y, "toarray") else X_y
                proba_row = self._pipeline.predict_proba(X_y_dense)[0]
                raw_p_y[y] = float(proba_row[class_to_index[y]])
            total = sum(raw_p_y.values())
            if total <= 0:
                # Degenerate — model produced no positive class mass under
                # any expansion.  Roll-forward posture: surface as uniform
                # rather than silently dropping the prediction.
                n = len(self._classes)
                normalized = {y: 1.0 / n for y in self._classes}
            else:
                normalized = {y: p / total for y, p in raw_p_y.items()}
            results.append({code: p for code, p in normalized.items() if p > 1e-6})
        return results

    def predict_proba_single(self, text: str) -> dict[str, float]:
        """Return calibrated probability distribution for a single text."""
        return self.predict_proba([text])[0]

    # Bundle-shape version for hierarchical NHSVM saves.  Bumped on any
    # change to the bundle's structural contract.  Old bundles fail-fast
    # on load (roll-forward posture: no compatibility shims).
    _NHSVM_BUNDLE_VERSION = "crammer_singer.v1"

    def save(self, path: str | Path) -> None:
        """Persist the trained pipeline to disk.

        Flat models save as a bare sklearn pipeline (joblib).
        Hierarchical models save as a dict containing all components
        needed to reconstruct the per-class Kronecker expansion path,
        plus a ``nhsvm_variant`` tag identifying the training-time
        objective (currently always ``crammer_singer.v1``).  Both write
        a ``.classes.json`` sidecar for quick introspection.
        """
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._hierarchical:
            bundle = {
                "hierarchical": True,
                "nhsvm_variant": self._NHSVM_BUNDLE_VERSION,
                "feature_union": self._feature_union,
                "svd": self._svd,
                "expander": self._expander,
                "classifier": self._pipeline,
                "classes": list(self._classes),
            }
            joblib.dump(bundle, str(path))
        else:
            joblib.dump(self._pipeline, str(path))

        classes_path = path.with_suffix(".classes.json")
        classes_path.write_text(json.dumps(self._classes))

        logger.info(
            "SVM saved to %s (%d classes, hierarchical=%s, variant=%s)",
            path, len(self._classes), self._hierarchical,
            self._NHSVM_BUNDLE_VERSION if self._hierarchical else "flat",
        )

    @classmethod
    def load(cls, path: str | Path, config: SVMConfig | None = None) -> SVMClassifier:
        """Load a persisted SVM pipeline from disk.

        Hierarchical bundles must carry the ``nhsvm_variant`` tag matching
        :attr:`_NHSVM_BUNDLE_VERSION`.  Bundles without this tag — or with
        a stale tag — fail loudly with ``RuntimeError``.  This is the
        roll-forward enforcement: legacy ``_nhsvm.pkl`` files from the
        pre-Crammer-Singer era will not silently load as if they were
        compatible.  Operators should clear the cache (or change the
        cache suffix) so a fresh training pass produces a current bundle.

        Flat (non-hierarchical) ``.pkl`` files load unchanged.
        """
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SVM model not found: {path}")

        loaded = joblib.load(str(path))

        obj = cls(config=config)
        if isinstance(loaded, dict) and loaded.get("hierarchical"):
            # Roll-forward: bundle must declare the current variant tag.
            # Pre-Crammer-Singer bundles (no ``nhsvm_variant`` key) are
            # rejected here — silently loading them would let an OvR-
            # trained pipeline drive the per-class inference path with
            # weights that were never fit for that geometry.
            variant = loaded.get("nhsvm_variant")
            if variant != cls._NHSVM_BUNDLE_VERSION:
                raise RuntimeError(
                    f"NHSVM bundle at {path} has nhsvm_variant={variant!r}; "
                    f"expected {cls._NHSVM_BUNDLE_VERSION!r}.  Clear the "
                    f"cache and retrain — legacy bundles are not loaded "
                    f"under the roll-forward Crammer-Singer regime."
                )
            obj._hierarchical = True
            obj._feature_union = loaded["feature_union"]
            obj._svd = loaded["svd"]
            obj._expander = loaded["expander"]
            obj._pipeline = loaded["classifier"]
            obj._classes = loaded.get("classes", list(obj._pipeline.classes_))
        else:
            obj._pipeline = loaded
            obj._classes = list(obj._pipeline.classes_)

        classes_path = path.with_suffix(".classes.json")
        if classes_path.exists():
            obj._classes = json.loads(classes_path.read_text())

        logger.info("SVM loaded from %s (%d classes, hierarchical=%s)",
                     path, len(obj._classes), obj._hierarchical)
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
