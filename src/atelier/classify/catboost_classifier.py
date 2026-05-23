# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""CatBoost column classifier — per-feature structured input for TreeSHAP.

Each column's 12 ``ColumnFeatures`` are encoded into a **structured**
CatBoost input matrix where every feature occupies its own named slice
(7 SentenceTransformer-embedded text features × 384 dims, plus 5 scalar
features × 1 dim each).  CatBoost is given ``feature_names`` aligned to
those slices via :func:`atelier.classify.embedding.feature_input_groups`.

Why structured instead of one big embedding: TreeSHAP attributes to a
model's native input space.  When the model receives a single vector
that's the sentence-transformer encoding of all 12 features
concatenated into one string, the 384 raw dimensions are a learned
nonlinear compression — TreeSHAP can only report aggregate
"embedding" contribution, never per-feature.  By giving each feature
its own dedicated slice, TreeSHAP attributes natively to each named
group; ``shap_explanations._build_feature_groups`` sums the SHAP
values within each slice to produce per-feature attribution.

Uses ``posterior_sampling`` on CPU for virtual-ensemble uncertainty
quantification; GPU trainer disables it (no GPU support in upstream
CatBoost) in exchange for a ~5× speedup.

Ported from signals/src/sigint/embedding_classifier.py and refactored
to the structured-input shape.
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
        # Per-dim feature names (e.g. column_name_d0...column_name_d383,
        # cardinality, ...) written into CatBoost at fit time.  Used for
        # TreeSHAP feature_names alignment.
        self._feature_names: list[str] = []
        # (name, start, end) slices from feature_input_groups() at fit
        # time, persisted to the .classes.json sidecar so SHAP can
        # group raw SHAP values into per-feature buckets at inference
        # without re-deriving the grouping from string-parsing
        # feature_names.
        self._feature_groups: list[tuple[str, int, int]] = []

    def fit(
        self,
        features_list,  # list[ColumnFeatures]
        labels: list[str],
        *,
        iterations: int = 1000,
        depth: int = 6,
        learning_rate: float = 0.1,
        verbose: int = 0,
        progress_ctx: dict | None = None,
    ) -> CatBoostColumnClassifier:
        """Train CatBoost on structured per-feature inputs.

        Each ``ColumnFeatures`` is encoded into the structured matrix by
        :func:`atelier.classify.embedding.embed_features`: 7 named
        text-feature slices (SentenceTransformer-embedded) + 5 scalar
        slots, with feature_names aligned to
        :func:`atelier.classify.embedding.feature_input_groups` so
        TreeSHAP attributes natively per source feature.

        Uses GPU when preflight_gpu reports availability.  CatBoost's GPU
        trainer does not support ``posterior_sampling``, so on GPU we
        disable it and accept the trade-off (no virtual-ensemble variance
        estimate) in exchange for a ~5× speedup.  The CPU path keeps
        posterior_sampling and the uncertainty it provides.
        """
        from catboost import CatBoostClassifier, Pool
        from atelier.classify.embedding import (
            embed_features, feature_input_groups,
        )

        # Deduplicate classes preserving order
        seen: set[str] = set()
        unique_classes = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                unique_classes.append(label)
        self._classes = sorted(unique_classes)

        # Encode features → structured (N, 7*384 + 5) matrix.
        embeddings = embed_features(features_list)

        # Build per-dim feature_names: text features get
        # "<feat>_d<idx>" so TreeSHAP attribution is interpretable
        # at debug time, but the per-feature group sum (via
        # _build_feature_groups slice math) is what gets reported as
        # the headline number.
        groups = feature_input_groups()
        feature_names: list[str] = []
        for name, start, end in groups:
            if end - start == 1:
                feature_names.append(name)
            else:
                feature_names.extend(f"{name}_d{i}" for i in range(end - start))
        # Persist for inference-side validation; saved/loaded with the
        # model so train+inference cannot drift.
        self._feature_names = feature_names
        self._feature_groups = [(name, int(s), int(e)) for name, s, e in groups]

        pool = Pool(
            data=embeddings,
            label=labels,
            feature_names=feature_names,
        )

        use_gpu = False
        try:
            from atelier.classify.gpu import preflight_gpu
            use_gpu = preflight_gpu().available
        except Exception:
            pass

        params: dict = dict(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            verbose=verbose,
            random_seed=42,
        )
        if use_gpu:
            params["task_type"] = "GPU"
            params["devices"] = "0"
        else:
            params["posterior_sampling"] = True

        self._model = CatBoostClassifier(**params)

        # Build per-iteration progress callback when progress_ctx is
        # supplied AND we're on CPU.  CatBoost's GPU trainer doesn't
        # accept the Python callbacks API; on GPU the phase_heartbeat
        # outer tick is the only signal (fine — GPU runs are fast).
        fit_kwargs: dict = {}
        if progress_ctx is not None and not use_gpu:
            class _ProgressCallback:
                """Writes per-iteration counts into the phase_heartbeat ctx
                so the UI tree-builder renders a determinate sub-phase bar
                while CatBoost trains.  No-op on GPU (callbacks unsupported).
                """
                def __init__(self, ctx: dict, total: int) -> None:
                    self._ctx = ctx
                    self._total = total
                def after_iteration(self, info) -> bool:
                    try:
                        self._ctx["current"] = int(getattr(info, "iteration", 0)) + 1
                        self._ctx["total"] = self._total
                    except Exception:
                        pass
                    return True  # continue training
            fit_kwargs["callbacks"] = [_ProgressCallback(progress_ctx, iterations)]

        try:
            self._model.fit(pool, **fit_kwargs)
        except Exception as exc:
            if use_gpu:
                # GPU trainer can fail for version/feature combinations —
                # fall back to CPU with the richer posterior_sampling path.
                logger.warning(
                    "CatBoost GPU training failed (%s); falling back to CPU", exc,
                )
                params.pop("task_type", None)
                params.pop("devices", None)
                params["posterior_sampling"] = True
                self._model = CatBoostClassifier(**params)
                # CPU fallback path: attach the progress callback (we're
                # no longer on GPU so it's safe to install).
                if progress_ctx is not None and "callbacks" not in fit_kwargs:
                    fit_kwargs["callbacks"] = [_ProgressCallback(progress_ctx, iterations)]
                self._model.fit(pool, **fit_kwargs)
            else:
                raise

        logger.info(
            "CatBoost trained (%s): %d samples, %d classes, %d dims",
            "GPU" if use_gpu else "CPU",
            len(labels), len(self._classes), embeddings.shape[1],
        )
        return self

    def predict_proba(self, features_list) -> list[dict[str, float]]:
        """Predict class probabilities for a batch of ``ColumnFeatures``.

        Encodes the features through :func:`embed_features` so the input
        shape exactly matches what was seen at train time.  Inference-
        side use ``predict_from_matrix`` if you have a pre-encoded
        matrix in hand (e.g. from a SHAP call) and want to avoid the
        re-encode cost.
        """
        if self._model is None:
            raise RuntimeError("Model not trained or loaded")

        from atelier.classify.embedding import embed_features
        embeddings = embed_features(features_list)
        return self.predict_from_matrix(embeddings)

    def predict_from_matrix(self, embeddings) -> list[dict[str, float]]:
        """Predict from a pre-encoded structured input matrix.

        The matrix must have the shape produced by
        :func:`atelier.classify.embedding.embed_features` —
        ``(N, 7 * emb_dim + 5)`` with the column ordering described in
        :func:`atelier.classify.embedding.feature_input_groups`.  Used
        by SHAP code that already has the matrix in hand and wants to
        skip the re-encode round-trip.
        """
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

    def predict_proba_single(self, features) -> dict[str, float]:
        """Predict for a single ``ColumnFeatures``."""
        return self.predict_proba([features])[0]

    def virtual_ensemble_variance(self, features_list) -> list[dict[str, float]]:
        """Get per-class variance from CatBoost virtual ensembles.

        Requires the model to be trained with posterior_sampling=True
        (CPU only; GPU trainer doesn't support it).
        """
        if self._model is None:
            raise RuntimeError("Model not trained or loaded")

        from atelier.classify.embedding import embed_features
        embeddings = embed_features(features_list)
        try:
            ve_preds = self._model.virtual_ensembles_predict(
                embeddings, prediction_type="TotalUncertainty",
            )
        except Exception:
            # Fallback: return empty variance dicts
            return [{} for _ in range(len(features_list))]

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
        """Save model to CatBoost native format + classes/groups JSON.

        The sidecar JSON carries both ``classes`` (for label decoding)
        and ``feature_groups`` (for SHAP per-feature attribution).  Old
        sidecars that only have classes are still loadable; the
        feature_groups will be re-derived from
        :func:`feature_input_groups` at load time as a fall-forward
        convenience for transitional reload of pre-refactor models.
        """
        if self._model is None:
            raise RuntimeError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))

        sidecar = {
            "classes": self._classes,
            "feature_groups": [
                {"name": name, "start": s, "end": e}
                for name, s, e in self._feature_groups
            ],
        }
        classes_path = path.with_suffix(".classes.json")
        with open(classes_path, "w") as f:
            json.dump(sidecar, f)

        logger.info(
            "CatBoost saved to %s (%d classes, %d feature groups)",
            path, len(self._classes), len(self._feature_groups),
        )

    @classmethod
    def load(cls, path: str | Path) -> CatBoostColumnClassifier:
        """Load a saved CatBoost model from disk.

        Loads the model and the sidecar JSON.  When the sidecar
        contains ``feature_groups`` (post-refactor format), the
        per-feature group bounds are restored as-saved.  When the
        sidecar is the legacy classes-only list (pre-refactor),
        feature_groups falls forward to the canonical
        :func:`feature_input_groups` shape — only safe if the legacy
        model happened to be trained at the same input dimensionality;
        otherwise the loaded model should be retrained.
        """
        from catboost import CatBoostClassifier
        from atelier.classify.embedding import feature_input_groups

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CatBoost model not found: {path}")

        instance = cls()
        instance._model = CatBoostClassifier()
        instance._model.load_model(str(path))

        classes_path = path.with_suffix(".classes.json")
        if classes_path.exists():
            with open(classes_path) as f:
                payload = json.load(f)
            if isinstance(payload, list):
                # Legacy sidecar: just the class list.
                instance._classes = payload
                instance._feature_groups = [
                    (name, int(s), int(e))
                    for name, s, e in feature_input_groups()
                ]
            else:
                instance._classes = payload.get("classes", [])
                instance._feature_groups = [
                    (g["name"], int(g["start"]), int(g["end"]))
                    for g in payload.get("feature_groups", [])
                ] or [
                    (name, int(s), int(e))
                    for name, s, e in feature_input_groups()
                ]
        else:
            instance._classes = [str(c) for c in instance._model.classes_]
            instance._feature_groups = [
                (name, int(s), int(e))
                for name, s, e in feature_input_groups()
            ]

        logger.info(
            "CatBoost loaded from %s (%d classes, %d feature groups)",
            path, len(instance._classes), len(instance._feature_groups),
        )
        return instance
