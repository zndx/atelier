"""Handlers for the evolve-classification post-restart task chain.

Each handler is idempotent: it checks for its expected output artifact
before doing work and returns ``TaskResult(skipped=True, ...)`` when
the work has already been done.  This lets the queue be re-driven safely
after a crash, restart, or operator-triggered retry — re-running a
completed task is a no-op.

Handlers registered:

  apply_enrichment_transforms   → scripts/apply_enrichment_transforms.py
  verify_transform_apply        → scripts/verify_transform_apply.py
  render_change_management_guide → scripts/render_change_management_guide.py
  trigger_pipeline_run          → fsm_start with the operator's last source
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

from atelier.task_queue import TaskResult, register

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────


def _run_subprocess(cmd: list[str]) -> tuple[int, str, str]:
    """Run a script as a subprocess; return (returncode, stdout, stderr)."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _find_manifest_for_cohort(cohort_path: str) -> Path | None:
    """Locate the most-recent manifest produced for this cohort directory."""
    cohort_name = Path(cohort_path).name  # e.g. cohort_umbrellas_v3
    manifests_dir = Path("build/data/transforms/manifests")
    if not manifests_dir.is_dir():
        return None
    candidates = sorted(
        manifests_dir.glob(f"{cohort_name}_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _manifest_already_promoted(manifest_path: Path) -> bool:
    """Has the manifest's target collection been promoted to 'current'?"""
    try:
        manifest = json.loads(manifest_path.read_text())
        target = manifest.get("target_collection")
    except (json.JSONDecodeError, OSError):
        return False
    try:
        from atelier.db.dao import AtelierDao
        dao = AtelierDao()
        cur = dao.get_current_taxonomy_collection("default")
        return cur is not None and cur.get("qdrant_collection") == target
    except Exception:  # noqa: BLE001
        return False


# ── apply_enrichment_transforms ──────────────────────────────────


@register("apply_enrichment_transforms")
def _handle_apply(params: dict) -> TaskResult:
    """Run scripts/apply_enrichment_transforms.py for a cohort.

    Idempotent: if a manifest produced from this cohort already exists
    AND its target collection is the registry's current row, skip.

    Params:
      cohort_dir (required): path to build/enrichment_evolution/cohort_*
      acceptance (optional): path to acceptance JSON
      dry_run (optional bool, default False)
      skip_promote (optional bool, default False)
      allow_duplicate (optional bool, default False)
    """
    cohort_dir = params.get("cohort_dir")
    if not cohort_dir:
        raise ValueError("cohort_dir is required")
    if not Path(cohort_dir).is_dir():
        raise FileNotFoundError(f"cohort_dir not found: {cohort_dir}")

    # Idempotency: existing manifest + promoted
    existing = _find_manifest_for_cohort(cohort_dir)
    if existing and _manifest_already_promoted(existing):
        return TaskResult(skipped=True, outcome={
            "manifest_path": str(existing),
            "reason": "manifest exists and target collection is current",
        })

    cmd = [sys.executable, "scripts/apply_enrichment_transforms.py", cohort_dir]
    if params.get("acceptance"):
        cmd += ["--acceptance", params["acceptance"]]
    if params.get("dry_run"):
        cmd += ["--dry-run"]
    if params.get("skip_promote"):
        cmd += ["--skip-promote"]
    if params.get("allow_duplicate"):
        cmd += ["--allow-duplicate"]

    rc, out, err = _run_subprocess(cmd)
    if rc != 0:
        raise RuntimeError(
            f"apply_enrichment_transforms.py exit {rc}\n--- stdout\n{out}\n"
            f"--- stderr\n{err}"
        )
    # Locate the manifest the run produced (latest matching cohort)
    manifest = _find_manifest_for_cohort(cohort_dir)
    return TaskResult(skipped=False, outcome={
        "manifest_path": str(manifest) if manifest else None,
        "stdout_tail": out[-2000:],
    })


# ── verify_transform_apply ───────────────────────────────────────


@register("verify_transform_apply")
def _handle_verify(params: dict) -> TaskResult:
    """Run scripts/verify_transform_apply.py for the cohort's manifest.

    Idempotent: verify output file already exists for this manifest.

    Params:
      cohort_dir (required): used to locate the matching manifest
      baseline_run (optional): run_id under build/results/; defaults
        to manifest.source_run
      k (optional int, default 25)
    """
    cohort_dir = params.get("cohort_dir")
    if not cohort_dir:
        raise ValueError("cohort_dir is required")

    manifest = _find_manifest_for_cohort(cohort_dir)
    if manifest is None:
        raise RuntimeError(
            f"No manifest found for cohort {cohort_dir} — apply must run first"
        )
    manifest_data = json.loads(manifest.read_text())
    manifest_id = manifest_data.get("manifest_id")
    verify_path = Path(f"build/data/transforms/verify_{manifest_id}.json")
    if verify_path.exists():
        return TaskResult(skipped=True, outcome={
            "verify_path": str(verify_path),
            "manifest_id": manifest_id,
            "reason": "verify artifact already exists",
        })

    cmd = [sys.executable, "scripts/verify_transform_apply.py", str(manifest)]
    if params.get("baseline_run"):
        cmd += ["--baseline-run", f"build/results/{params['baseline_run']}"]
    if params.get("k"):
        cmd += ["--k", str(params["k"])]
    rc, out, err = _run_subprocess(cmd)
    if rc != 0:
        raise RuntimeError(
            f"verify_transform_apply.py exit {rc}\n--- stdout\n{out}\n"
            f"--- stderr\n{err}"
        )
    return TaskResult(skipped=False, outcome={
        "manifest_path": str(manifest),
        "verify_path": str(verify_path) if verify_path.exists() else None,
        "stdout_tail": out[-2000:],
    })


# ── render_change_management_guide ───────────────────────────────


@register("render_change_management_guide")
def _handle_guide(params: dict) -> TaskResult:
    """Run scripts/render_change_management_guide.py for the cohort.

    Idempotent: the .md guide alongside the manifest already exists.

    Params:
      cohort_dir (required)
    """
    cohort_dir = params.get("cohort_dir")
    if not cohort_dir:
        raise ValueError("cohort_dir is required")

    manifest = _find_manifest_for_cohort(cohort_dir)
    if manifest is None:
        raise RuntimeError(
            f"No manifest found for cohort {cohort_dir} — apply must run first"
        )
    guide_path = manifest.with_suffix(".md")
    if guide_path.exists():
        return TaskResult(skipped=True, outcome={
            "guide_path": str(guide_path),
            "reason": "guide already rendered",
        })

    manifest_data = json.loads(manifest.read_text())
    manifest_id = manifest_data.get("manifest_id")
    verify_path = Path(f"build/data/transforms/verify_{manifest_id}.json")
    audit_path = Path("build/diag/cosine_signal_audit.json")

    cmd = [sys.executable, "scripts/render_change_management_guide.py",
           str(manifest)]
    if verify_path.exists():
        cmd += ["--verify", str(verify_path)]
    if audit_path.exists():
        cmd += ["--audit", str(audit_path)]
    rc, out, err = _run_subprocess(cmd)
    if rc != 0:
        raise RuntimeError(
            f"render_change_management_guide.py exit {rc}\n--- stdout\n{out}\n"
            f"--- stderr\n{err}"
        )
    return TaskResult(skipped=False, outcome={
        "manifest_path": str(manifest),
        "guide_path": str(guide_path) if guide_path.exists() else None,
        "stdout_tail": out[-1500:],
    })


# ── trigger_pipeline_run ─────────────────────────────────────────


@register("trigger_pipeline_run")
def _handle_pipeline(params: dict) -> TaskResult:
    """Trigger a new classification pipeline run via fsm_start.

    Idempotent against the *active* collection: if a non-terminal run
    is already in flight, skip.  This handler is meant to be the last
    in the chain — kick off a fresh pipeline run against the freshly-
    promoted collection.

    Params:
      source_id (optional): override; defaults to last user-selected
        source from existing FSM runs.
    """
    try:
        from atelier.gateway import fsm_start, _last_user_selected_source_id
    except ImportError as exc:
        raise RuntimeError(f"gateway not importable: {exc}")
    from atelier.fsm import get_state

    # In-flight guard: skip if a run is currently active
    try:
        state = get_state()
        active_states = {"LOADING_VOCAB", "DISCOVERING", "SAMPLING",
                         "LLM_SWEEP", "VALIDATING", "CLASSIFYING",
                         "FUSING", "EVALUATING"}
        if state and state.get("state") in active_states:
            return TaskResult(skipped=True, outcome={
                "fsm_state": state.get("state"),
                "reason": "pipeline already running",
            })
    except Exception:  # noqa: BLE001
        # If we can't probe FSM state, proceed anyway — fsm_start has
        # its own in-flight check via the lifespan_start guard.
        pass

    source_id = params.get("source_id") or _last_user_selected_source_id()
    if not source_id:
        raise RuntimeError(
            "No source_id available — pass via params or seed via prior FSM runs"
        )
    result = fsm_start(source_id=source_id)
    return TaskResult(skipped=False, outcome={
        "source_id": source_id,
        "fsm_start_result": str(result),
    })
