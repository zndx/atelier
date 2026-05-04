# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Reconcile DB state against the on-disk run artifacts.

The classify pipeline writes per-run artifacts under
``build/results/<run_id>/`` (parquet, classifications.json, model
files, settings_snapshot.json) and registers a corresponding row
in ``ml_artifact_sets`` and ``datasets``.  When either DB write
fails — transient PGlite hiccup, a code regression like the FK
ordering bug fixed in commit 1db0c6e, or a crash between writes —
the artifacts stay on disk but the UI loses the run.

This module makes the filesystem authoritative.  ``sync_filesystem_to_db``
walks ``build/results/`` and, for each complete run, idempotently
registers whatever DB rows are missing.  Designed to be called once
at gateway lifespan startup so a fresh AMP boot reconciles
automatically; also exposed via ``scripts/backfill_dataset.py`` as
a CLI seam for operator-driven runs against a single ``run_id``.

Source-id resolution priority (first hit wins):

  1. Explicit override (``source_id_override`` parameter or ``--source-id``
     CLI flag).
  2. ``settings_snapshot.json::source_id`` — populated by the
     pipeline at run start since 2026-05-04 (commits in the same
     series as this module).
  3. ``ml_artifact_sets`` row's ``source_id`` — only available when
     the artifact set was registered before the dataset row failed
     (the typical orphan-shape under the M9-era FK bug; less likely
     going forward).
  4. None.  The dataset row still lands but with ``source_id=NULL``;
     the operator can fix it later via the Settings page or by
     re-running the classify pipeline against the right source.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atelier.db.dao import AtelierDao

logger = logging.getLogger(__name__)


@dataclass
class _RunOutcome:
    """Per-run reconcile outcome — the unit of truth in SyncReport."""

    run_id: str
    artifact_set: str = "skipped"   # registered | already_registered | skipped | error
    dataset: str = "skipped"        # registered | already_registered | skipped | error
    source_id: str | None = None
    note: str = ""


@dataclass
class SyncReport:
    """Aggregate outcome of a sync pass.

    Counts are derived from ``outcomes`` so the per-run detail stays
    available for logging or operator-facing surfacing.  A clean
    sync produces ``errors == 0`` and ``newly_registered == 0`` on
    a steady-state system.
    """

    outcomes: list[_RunOutcome] = field(default_factory=list)

    @property
    def scanned(self) -> int:
        return len(self.outcomes)

    @property
    def newly_registered(self) -> int:
        return sum(
            1 for o in self.outcomes
            if o.artifact_set == "registered" or o.dataset == "registered"
        )

    @property
    def errors(self) -> int:
        return sum(
            1 for o in self.outcomes
            if o.artifact_set == "error" or o.dataset == "error"
        )

    def summary_line(self) -> str:
        return (
            f"sync: scanned={self.scanned}, newly_registered={self.newly_registered}, "
            f"errors={self.errors}"
        )


def _is_complete_run(run_dir: Path) -> bool:
    """A run is sync-eligible iff its observable outputs are present."""
    return (
        (run_dir / "atelier_embeddings.parquet").is_file()
        and (run_dir / "classifications.json").is_file()
    )


def _resolve_source_id(
    run_dir: Path,
    *,
    override: str | None,
    existing_artifact_set: dict | None,
) -> str | None:
    """Apply the priority ladder documented in the module docstring."""
    if override:
        return override
    snap = run_dir / "settings_snapshot.json"
    if snap.is_file():
        try:
            blob = json.loads(snap.read_text())
            sid = blob.get("source_id")
            if sid:
                return str(sid)
        except Exception:
            pass
    if existing_artifact_set is not None:
        sid = existing_artifact_set.get("source_id")
        if sid:
            return str(sid)
    return None


def _classification_count(run_dir: Path) -> int:
    """Row count from classifications.json — used as ``row_count`` on
    the dataset row.  Returns 0 on parse failure (the upsert still
    lands; the count just shows as zero in the UI)."""
    try:
        c = json.loads((run_dir / "classifications.json").read_text())
        if isinstance(c, list):
            return len(c)
    except Exception:
        pass
    return 0


def _is_duplicate_key_error(exc: BaseException) -> bool:
    """Detect benign 'already registered' DB errors across drivers."""
    msg = str(exc).lower()
    return any(token in msg for token in (
        "duplicate", "unique", "already exists",
    ))


def _ensure_fsm_run(run_id: str, dao: AtelierDao, source_id: str | None) -> None:
    """Guarantee an ``fsm_runs`` row exists so FK-dependent inserts succeed.

    ``ml_artifact_sets.fsm_run_id REFERENCES fsm_runs(id)`` — without a
    target row, the artifact-set insert raises ``IntegrityError``.  When
    reconciling an orphaned run the FSM row may never have been written
    (PGlite was unreachable when the pipeline tried) so we insert a
    synthetic CONVERGED row.  Idempotent — skips if the row exists.
    """
    existing = dao.get_fsm_run(run_id)
    if existing is not None:
        return
    import datetime as _dt
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    dao.upsert_fsm_run(
        run_id=run_id,
        state="CONVERGED",
        started_at=now,
        updated_at=now,
        progress="",
        error=None,
        result_path=f"build/results/{run_id}",
        source_id=source_id,
    )
    logger.info("sync: created synthetic fsm_runs row for %s", run_id)


def _reconcile_run(
    run_dir: Path,
    dao: AtelierDao,
    *,
    source_id_override: str | None,
    cfg,
) -> _RunOutcome:
    """Reconcile a single run — idempotent."""
    run_id = run_dir.name
    outcome = _RunOutcome(run_id=run_id)

    if not _is_complete_run(run_dir):
        outcome.note = "incomplete (missing parquet or classifications.json)"
        return outcome

    # ── Step 0: ensure fsm_runs row exists (FK target) ────────────
    # ml_artifact_sets.fsm_run_id REFERENCES fsm_runs(id) — the
    # pipeline creates the fsm_runs row early, but if PGlite was
    # unreachable at that point the row is missing and every
    # downstream insert fails with IntegrityError.
    source_id_early = _resolve_source_id(
        run_dir, override=source_id_override, existing_artifact_set=None,
    )
    try:
        _ensure_fsm_run(run_id, dao, source_id_early)
    except Exception as exc:
        if not _is_duplicate_key_error(exc):
            outcome.artifact_set = "error"
            outcome.note = f"fsm_run ensure: {type(exc).__name__}: {exc}"
            logger.exception("sync: fsm_run ensure failed for %s", run_id)
            return outcome

    # ── Step 1: artifact set ───────────────────────────────────────
    existing_as = dao.get_artifact_set(run_id)
    if existing_as is not None:
        outcome.artifact_set = "already_registered"
        artifact_set_present = True
    else:
        artifact_set_present = False
        try:
            from atelier.classify.artifact_set import build_artifact_set_record
            spec = build_artifact_set_record(
                run_id=run_id,
                results_dir=run_dir,
                cfg=cfg,
                n_columns=_classification_count(run_dir),
                source_id=source_id_early,
                fsm_run_id=run_id,
            )
            if spec is None:
                outcome.artifact_set = "skipped"
                outcome.note = "no ML artifacts to register"
            else:
                try:
                    dao.register_artifact_set(**spec)
                    outcome.artifact_set = "registered"
                    artifact_set_present = True
                except Exception as exc:
                    if _is_duplicate_key_error(exc):
                        outcome.artifact_set = "already_registered"
                        artifact_set_present = True
                    else:
                        outcome.artifact_set = "error"
                        outcome.note = f"artifact_set: {type(exc).__name__}: {exc}"
                        logger.exception(
                            "sync: artifact_set register failed for %s", run_id,
                        )
                        return outcome
        except Exception as exc:
            outcome.artifact_set = "error"
            outcome.note = f"artifact_set build: {type(exc).__name__}: {exc}"
            logger.exception("sync: artifact_set build failed for %s", run_id)
            return outcome

    # ── Step 2: dataset ────────────────────────────────────────────
    existing_ds = dao.get_dataset(run_id)
    if existing_ds is not None:
        outcome.dataset = "already_registered"
        outcome.source_id = existing_ds.get("source_id")
        return outcome

    refreshed_as = existing_as or dao.get_artifact_set(run_id)
    source_id = _resolve_source_id(
        run_dir,
        override=source_id_override,
        existing_artifact_set=refreshed_as,
    )
    outcome.source_id = source_id

    parquet_path = run_dir / "atelier_embeddings.parquet"
    n_rows = _classification_count(run_dir)
    payload = {
        "dataset_id": run_id,
        "name": f"Classification {run_id[:8]}",
        "parquet_path": str(parquet_path),
        "description": f"{n_rows} columns classified",
        "row_count": n_rows,
        "source_id": source_id,
        "is_active": False,        # sync never auto-activates; operator promotes
        "summary": f"{n_rows} columns (sync-recovered)",
        "fsm_run_id": run_id,
        "artifact_set_id": run_id if artifact_set_present else None,
        "parent_dataset_id": None,
        "run_kind": "classify",
    }
    if source_id:
        try:
            payload["version_number"] = dao.next_version_number(source_id)
        except Exception:
            payload["version_number"] = 1
    else:
        payload["version_number"] = 1

    try:
        dao.upsert_dataset(**payload)
        outcome.dataset = "registered"
    except Exception as exc:
        if _is_duplicate_key_error(exc):
            outcome.dataset = "already_registered"
        else:
            outcome.dataset = "error"
            outcome.note = f"dataset upsert: {type(exc).__name__}: {exc}"
            logger.exception("sync: dataset upsert failed for %s", run_id)
            return outcome

    # ── Step 3: tidy any stale register_error.json sidecar ─────────
    err_path = run_dir / "register_error.json"
    if err_path.is_file():
        try:
            err_path.rename(err_path.with_suffix(".json.resolved"))
        except Exception:
            pass

    return outcome


def sync_filesystem_to_db(
    results_root: Path,
    *,
    dao: AtelierDao | None = None,
    cfg=None,
    source_id_override: str | None = None,
    only_run_id: str | None = None,
) -> SyncReport:
    """Reconcile DB rows against ``build/results/`` artifacts.

    Idempotent.  Safe to call repeatedly; runs already represented in
    the DB cost one read per row and produce ``already_registered``
    outcomes.

    Args:
        results_root: Typically ``build/results``.  When the directory
            doesn't exist or is empty, returns an empty report.
        dao: Optional pre-built ``AtelierDao``.  Defaults to a fresh
            instance.
        cfg: Optional ``AtelierConfig``.  Defaults to ``load_config()``.
        source_id_override: Force this source_id for every reconciled
            run — only useful when ``only_run_id`` is set (CLI form).
        only_run_id: Reconcile a single run instead of the whole tree
            (CLI form, drives ``scripts/backfill_dataset.py``).

    Returns:
        ``SyncReport`` with one ``_RunOutcome`` per scanned run.
    """
    report = SyncReport()
    if not results_root.is_dir():
        return report

    if dao is None:
        from atelier.db.dao import AtelierDao
        dao = AtelierDao()

    if cfg is None:
        from atelier.config import load_config
        cfg = load_config()

    if only_run_id is not None:
        candidates = [results_root / only_run_id]
    else:
        candidates = sorted(p for p in results_root.iterdir() if p.is_dir())

    for run_dir in candidates:
        if not run_dir.is_dir():
            report.outcomes.append(_RunOutcome(
                run_id=run_dir.name, note="run directory not found",
            ))
            continue
        outcome = _reconcile_run(
            run_dir, dao,
            source_id_override=source_id_override,
            cfg=cfg,
        )
        report.outcomes.append(outcome)

    return report
