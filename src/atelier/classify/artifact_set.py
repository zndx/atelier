# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Helpers for ML Artifact Set lifecycle.

An ML Artifact Set is the bundle of trained-model files produced by an
Atelier classify run: the CatBoost classifier (.cbm + .classes.json),
the optional incremental SVM classifier (.pkl + .classes.json), the
fitted UMAP projection (.pkl), and metadata that lets a downstream
Extend Classification run replay the model on new data without
re-running the full pipeline.

The on-disk layout is fixed by :mod:`atelier.classify.pipeline`:

    build/results/{run_id}/
        catboost_fit_to_llm.cbm
        catboost_fit_to_llm.cbm.classes.json
        svm_frontier.pkl                     (optional)
        svm_frontier.pkl.classes.json        (optional)
        umap.pkl                             (added 20260427)
        settings_snapshot.json               (used for embedding model identity)

This module is the single point of knowledge about that layout — when
the pipeline writes new artifact files or the layout changes, only the
helpers here need updating; the DAO + UI surface stay decoupled.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Filenames within a run's results dir.  Constants so the pipeline,
# the DAO, and the Extend runner agree on what to write / read.
CATBOOST_FILENAME = "catboost_fit_to_llm.cbm"
SVM_FILENAME = "svm_frontier.pkl"
UMAP_FILENAME = "umap.pkl"


def _classes_sidecar(model_path: Path) -> Path:
    """Return the conventional ``.classes.json`` sidecar next to ``model_path``.

    Uses ``Path.with_suffix(".classes.json")`` to match the convention
    in :class:`CatBoostColumnClassifier.save` and
    :class:`SVMClassifier.save` — both replace the model extension
    rather than append, so ``catboost_fit_to_llm.cbm`` →
    ``catboost_fit_to_llm.classes.json``.
    """
    return model_path.with_suffix(".classes.json")


def compute_vocab_signature(classes: list[str]) -> str:
    """Stable sha256 of the sorted, JSON-encoded class list.

    Used as a fast equality check between an artifact set's training
    classes and a candidate runtime vocab.  The signature is canonical
    under reordering — clients can recompute it from any source vocab
    and compare to the persisted signature with a string ==.
    """
    canonical = json.dumps(sorted(classes), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class CompatibilityReport:
    """Result of comparing an artifact's classes with a candidate vocab.

    ``status`` is the coarse verdict the UI surfaces:

    - ``ok``        : artifact classes == candidate classes (signatures match)
    - ``superset``  : artifact ⊂ candidate (model can predict everything
                       it learned; some candidate codes will simply
                       never be predicted — fine in practice)
    - ``partial``   : artifact has codes the candidate doesn't define
                       (predictions may emit codes the source can't
                       interpret) — warn but allow
    - ``disjoint``  : zero overlap — almost certainly the wrong artifact
                       for this source; warn loudly but still allow
    """

    status: str  # one of: ok | superset | partial | disjoint
    artifact_signature: str
    candidate_signature: str
    missing_codes: list[str]   # in artifact but not in candidate
    extra_codes: list[str]     # in candidate but not in artifact

    @property
    def is_match(self) -> bool:
        """True iff signatures are identical (status='ok')."""
        return self.status == "ok"


def check_compatibility(
    artifact_classes: list[str],
    candidate_classes: list[str],
) -> CompatibilityReport:
    """Compare an artifact set's training classes against a candidate vocab.

    Used at Extend Classification time to surface vocabulary drift
    between the training run and the source the operator is targeting.
    Per the project decision, mismatches WARN rather than block — the
    artifact's classes drive the runtime taxonomy of the Extend run.
    """
    artifact_set = set(artifact_classes)
    candidate_set = set(candidate_classes)
    missing = sorted(artifact_set - candidate_set)
    extra = sorted(candidate_set - artifact_set)
    artifact_sig = compute_vocab_signature(artifact_classes)
    candidate_sig = compute_vocab_signature(candidate_classes)

    if artifact_sig == candidate_sig:
        status = "ok"
    elif not artifact_set & candidate_set:
        status = "disjoint"
    elif artifact_set <= candidate_set:
        status = "superset"
    else:
        status = "partial"

    return CompatibilityReport(
        status=status,
        artifact_signature=artifact_sig,
        candidate_signature=candidate_sig,
        missing_codes=missing,
        extra_codes=extra,
    )


def _read_classes_sidecar(path: Path) -> tuple[list[str], list[dict]]:
    """Read CatBoost's ``.classes.json`` sidecar.

    Returns (classes, feature_groups).  Handles both:
    - new format ``{"classes": [...], "feature_groups": [...]}`` and
    - legacy format ``[...]`` (just the class list, no groups).
    """
    if not path.exists():
        return [], []
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [str(c) for c in payload], []
    classes = [str(c) for c in payload.get("classes", [])]
    groups = list(payload.get("feature_groups", []))
    return classes, groups


def build_artifact_set_record(
    run_id: str,
    results_dir: Path,
    cfg: Any,
    *,
    n_columns: int = 0,
    source_id: str | None = None,
    fsm_run_id: str | None = None,
    parent_artifact_set_id: str | None = None,
) -> dict[str, Any] | None:
    """Inspect ``results_dir``, return a dict ready for ``register_artifact_set``.

    Returns None when CatBoost output is missing — the caller should
    log and skip registration in that case (rare; indicates the run
    failed before training finished).  Optional artifacts (SVM, UMAP)
    register as None when absent.

    ``n_columns`` is the number of classified columns the run produced;
    used purely for the cosmetic summary string.  Defaults to 0, in
    which case the summary reports the class count instead.
    """
    results_dir = Path(results_dir)
    catboost_path = results_dir / CATBOOST_FILENAME
    catboost_classes_path = _classes_sidecar(catboost_path)

    if not catboost_path.is_file() or not catboost_classes_path.is_file():
        logger.warning(
            "Artifact set build skipped: CatBoost output missing in %s",
            results_dir,
        )
        return None

    svm_path = results_dir / SVM_FILENAME
    svm_classes_path = _classes_sidecar(svm_path)
    has_svm = svm_path.is_file() and svm_classes_path.is_file()

    umap_path = results_dir / UMAP_FILENAME
    has_umap = umap_path.is_file()

    classes, feature_groups = _read_classes_sidecar(catboost_classes_path)
    if not classes:
        logger.warning(
            "Artifact set build skipped: empty classes sidecar at %s",
            catboost_classes_path,
        )
        return None

    embedding_model = getattr(
        cfg, "embedding_model", "sentence-transformers/all-MiniLM-L6-v2",
    )
    embedding_dim = int(getattr(cfg, "embedding_dim", 384))

    # Display-string conventions: short id + column count + creation date
    # (date populated by the DB DEFAULT, but the in-record summary is a
    # quick scan for the UI before the row is reloaded).
    n_cols = int(n_columns) or len(classes)
    display_name = f"Run {run_id[:8]} — {len(classes)} classes"
    summary = f"{n_cols} cols, {len(classes)} classes"

    facets = {
        "schema": {
            "fields": [
                {"name": name, "type": "embedding|scalar"}
                for name in [g.get("name") for g in feature_groups]
                if name
            ],
        },
        "zndx_ml_artifact": {
            "framework": "catboost",
            "has_svm": has_svm,
            "has_umap": has_umap,
            "trained_at_run": run_id,
        },
    }

    return {
        "artifact_set_id": run_id,
        "source_id": source_id,
        # Caller decides whether to set the FK: pipeline passes the
        # producing run_id; backfill paths pass None when no fsm_runs
        # row exists for the historical run.
        "fsm_run_id": fsm_run_id,
        "parent_artifact_set_id": parent_artifact_set_id,
        "catboost_path": str(catboost_path),
        "catboost_classes_path": str(catboost_classes_path),
        "svm_path": str(svm_path) if has_svm else None,
        "svm_classes_path": str(svm_classes_path) if has_svm else None,
        "umap_path": str(umap_path) if has_umap else None,
        "classes": json.dumps(classes),
        "feature_groups": json.dumps(feature_groups) if feature_groups else None,
        "vocab_signature": compute_vocab_signature(classes),
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "display_name": display_name,
        "summary": summary,
        "facets": json.dumps(facets),
    }


@dataclass
class ExtendBundle:
    """Loaded inference handles for an Extend Classification run.

    Holds the CatBoost classifier (always present), the optional SVM,
    the optional UMAP projection model, and the training classes.  The
    Extend runner consumes this dataclass directly; nothing else does.
    """

    catboost: Any                 # CatBoostColumnClassifier
    svm: Any | None               # SVMClassifier | None
    umap_model: Any | None        # umap.UMAP | None
    classes: list[str]
    feature_groups: list[dict]
    embedding_model: str
    embedding_dim: int
    artifact_set_id: str


def load_for_extend(artifact_set: dict) -> ExtendBundle:
    """Load CatBoost / SVM / UMAP referenced by the artifact-set row.

    Validates each path on disk; raises FileNotFoundError early so the
    Extend FSM run aborts in LOADING_VOCAB rather than partway through
    classification.
    """
    from atelier.classify.catboost_classifier import CatBoostColumnClassifier

    catboost_path = Path(artifact_set["catboost_path"])
    if not catboost_path.is_file():
        raise FileNotFoundError(
            f"Artifact set {artifact_set['id']}: CatBoost missing at "
            f"{catboost_path}"
        )
    catboost = CatBoostColumnClassifier.load(catboost_path)

    svm = None
    if artifact_set.get("svm_path"):
        from atelier.classify.svm_classifier import SVMClassifier
        svm_path = Path(artifact_set["svm_path"])
        if not svm_path.is_file():
            raise FileNotFoundError(
                f"Artifact set {artifact_set['id']}: SVM path recorded "
                f"but file missing at {svm_path}"
            )
        svm = SVMClassifier.load(svm_path)

    umap_model = None
    if artifact_set.get("umap_path"):
        umap_path = Path(artifact_set["umap_path"])
        if not umap_path.is_file():
            raise FileNotFoundError(
                f"Artifact set {artifact_set['id']}: UMAP path recorded "
                f"but file missing at {umap_path}"
            )
        import joblib
        umap_model = joblib.load(umap_path)

    classes_raw = artifact_set.get("classes", "[]")
    classes = json.loads(classes_raw) if isinstance(classes_raw, str) else list(classes_raw)
    groups_raw = artifact_set.get("feature_groups")
    feature_groups = (
        json.loads(groups_raw) if isinstance(groups_raw, str) and groups_raw
        else (groups_raw or [])
    )

    return ExtendBundle(
        catboost=catboost,
        svm=svm,
        umap_model=umap_model,
        classes=classes,
        feature_groups=feature_groups,
        embedding_model=artifact_set.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2",
        ),
        embedding_dim=int(artifact_set.get("embedding_dim", 384)),
        artifact_set_id=artifact_set["id"],
    )
