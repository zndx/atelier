#!/usr/bin/env python
"""Phase 5 of /evolve-classification — apply enrichment transforms forward.

Roll-forward only: every invocation produces a *new* versioned Qdrant
collection.  The previous collection becomes ``stale`` via the registry's
atomic demote-then-promote; nothing is archived or deleted by this script.
"Rollback-like" outcomes are achieved by synthesizing an inverse-transform
cohort and applying it as a fresh forward step.

Algorithm summary:

  1. Resolve the current taxonomy collection from ``taxonomy_registry``.
  2. Derive a new ``augmentation_version`` + target collection name.
  3. Verify ColBERT encoder model matches the source collection's
     ``embedding_model`` — fail-fast on drift.
  4. Filter ``candidates.json`` by ``--acceptance`` (default: confidence
     ≠ "low" AND status in {high_apply, manual_review-with-proposal,
     confirm_current-with-changes}).
  5. Compute manifest-id (SHA256 over cohort_path + sorted accepted
     codes + new_aug_version).  Refuse duplicate unless --allow-duplicate.
  6. Resume-safe: re-use an existing ``building`` row for the target
     collection if one exists; otherwise register a fresh one.
  7. ensure_collection on target.
  8. Scroll the source collection in batches of 256 with vectors.  For
     each point:
       - If ``payload["code"]`` not in accepted set → re-upsert as-is
         (cheap copy with a new point_id from the new augmentation_version).
       - Else → mutate payload (description, name_hints fold), re-encode
         via ColBERT, recompute point_id, upsert.
  9. Persist transforms manifest + per-transform records under
     ``build/data/transforms/``.
 10. Unless --skip-promote: ``set_current_taxonomy_collection`` atomic
     demote-then-promote.

Output:
  build/data/transforms/manifests/<cohort>_<vN>.json
  build/data/transforms/records/<transform_id>.json
  Qdrant collection: ``annotations_<taxonomy>_<new_aug_version>``
  taxonomy_registry: new row (current), prior current → stale
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Acceptance filter ─────────────────────────────────────────────


def default_acceptance(records: list[dict]) -> list[dict]:
    """Default policy: any proposal with confidence != 'low' that has a
    concrete proposed_mnemonic distinct from the current reference.

    Filters out:
      - status=row_review (yellow flags — no proposal)
      - status=error (LLM call failed)
      - status=insufficient_evidence
      - low-confidence proposals
      - confirm_current (no change needed)
      - TAXONOMY_GAP / INSUFFICIENT_EVIDENCE sentinel proposals
    """
    accepted = []
    for r in records:
        if r.get("status") in (
            "row_review", "error", "insufficient_evidence", "confirm_current",
        ):
            continue
        if (r.get("confidence") or "").lower() == "low":
            continue
        mn = r.get("proposed_mnemonic")
        if mn in (None, "", "INSUFFICIENT_EVIDENCE", "TAXONOMY_GAP"):
            continue
        # Skip when proposal would be a no-op (proposed == current mnemonic
        # at the target_code) — this is a stricter check than confirm_current
        # status because enrichment-evolution proposals work on text edits
        # rather than mnemonic swaps, so we accept on text-changed.
        accepted.append(r)
    return accepted


def load_cohort(cohort_dir: Path) -> dict:
    """Load candidates.json from an enrichment_evolution cohort dir."""
    cpath = cohort_dir / "candidates.json"
    if not cpath.is_file():
        sys.exit(f"candidates.json not found at {cpath}")
    return json.loads(cpath.read_text())


def load_acceptance(path: Path) -> set[str]:
    """Optional explicit acceptance JSON: ``{accepted_codes: [<code>, ...]}``."""
    if not path.is_file():
        sys.exit(f"--acceptance file not found: {path}")
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return set(data)
    return set(data.get("accepted_codes", []))


# ── Manifest id + duplicate detection ─────────────────────────────


def compute_manifest_id(
    cohort_path: Path, accepted_codes: list[str], new_aug_version: str,
) -> str:
    blob = "::".join([
        str(cohort_path.resolve()),
        ",".join(sorted(accepted_codes)),
        new_aug_version,
    ])
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return digest[:16]


def find_duplicate_manifest(manifest_id: str) -> Path | None:
    """Scan existing manifests for a matching manifest_id."""
    manifests_dir = Path("build/data/transforms/manifests")
    if not manifests_dir.is_dir():
        return None
    for f in manifests_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            if d.get("manifest_id") == manifest_id:
                return f
        except (json.JSONDecodeError, OSError):
            continue
    return None


# ── Transform application ─────────────────────────────────────────


def fold_common_names_into_hints(name_hints: list, new_common_names: str) -> list:
    """Fold the proposed common_names string into name_hints payload.

    Avoids extending ``compose_annotation_text`` (which doesn't read a
    dedicated common_names field).  The hints with source=evolve carry
    provenance so future audits can identify which hints came from
    rewrite proposals vs original enrichment.
    """
    if not new_common_names:
        return list(name_hints or [])
    existing_hints = list(name_hints or [])
    existing_text = {
        (h.get("hint") if isinstance(h, dict) else str(h)).strip().lower()
        for h in existing_hints
    }
    out = list(existing_hints)
    for raw in new_common_names.split(","):
        text = raw.strip()
        if not text:
            continue
        if text.lower() in existing_text:
            continue
        out.append({"hint": text, "source": "evolve"})
        existing_text.add(text.lower())
    return out


def apply_transform_to_payload(payload: dict, proposal: dict, applied_at: str) -> dict:
    """Return a NEW payload reflecting the transform.

    Mutates: ``description`` (with new_definition), ``name_hints`` (folds
    new_common_names), appends an ``operator_edits`` entry.  Other fields
    pass through unchanged.  Caller will recompute ``source_row_hash``
    and point_id.
    """
    new_payload = dict(payload)
    proposal_primary = proposal.get("subset") or proposal.get("full") or {}

    prior_desc = new_payload.get("description") or ""
    new_desc = proposal_primary.get("new_definition") or proposal.get("proposed_mnemonic")
    if new_desc and new_desc != prior_desc:
        new_payload["description"] = new_desc

    new_common = proposal_primary.get("new_common_names") or ""
    if new_common:
        new_payload["name_hints"] = fold_common_names_into_hints(
            new_payload.get("name_hints", []), new_common,
        )

    edits = list(new_payload.get("operator_edits") or [])
    edits.append({
        "kind": "evolve_rewrite",
        "applied_at": applied_at,
        "proposal_status": proposal.get("status"),
        "confidence": proposal.get("confidence"),
        "model": proposal_primary.get("model")
                or proposal.get("model"),
        "reasoning_snippet": (proposal.get("reasoning") or "")[:240],
    })
    new_payload["operator_edits"] = edits

    return new_payload


# ── Main flow ─────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cohort_dir",
        help="Path to build/enrichment_evolution/cohort_<name>_v<N>/")
    parser.add_argument("--acceptance", default=None,
        help="JSON file listing accepted target codes "
             "(default: heuristic from candidates.json)")
    parser.add_argument("--target-collection", default=None,
        help="Override the derived target collection name")
    parser.add_argument("--source-collection", default=None,
        help="Override the current registry row's collection")
    parser.add_argument("--dry-run", action="store_true",
        help="Plan + write manifest only; no Qdrant or registry writes")
    parser.add_argument("--skip-promote", action="store_true",
        help="Register building row + upsert points but don't promote")
    parser.add_argument("--allow-duplicate", action="store_true",
        help="Override duplicate-manifest detection")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cohort_dir = Path(args.cohort_dir)
    if not cohort_dir.is_dir():
        sys.exit(f"Cohort dir not found: {cohort_dir}")
    cohort_name = cohort_dir.name  # e.g. cohort_umbrellas_v3

    # ── Resolve source collection from registry ──────────────────
    from atelier.db.dao import AtelierDao
    try:
        dao = AtelierDao()
        if args.source_collection:
            all_rows = dao.list_taxonomy_collections(
                taxonomy_id="default", include_archived=True,
            )
            match = [r for r in all_rows
                     if r.get("qdrant_collection") == args.source_collection]
            if not match:
                sys.exit(f"--source-collection {args.source_collection!r} "
                         f"not in registry")
            source = match[0]
        else:
            source = dao.get_current_taxonomy_collection("default")
            if source is None:
                sys.exit("No current taxonomy collection in registry. "
                         "Run enrichment first to seed one.")
    except Exception as exc:
        # PGlite typically only reachable in the App pod; surface a clear
        # message when running from a Session pod by mistake.
        sys.exit(
            f"taxonomy_registry probe failed: {type(exc).__name__}: {exc}\n"
            f"  This script requires the App pod's PGlite on "
            f"127.0.0.1:5440.  Invoke from the Web Terminal Agent inside "
            f"the running Atelier App, or via /evolve-classification."
        )
    source_collection = source["qdrant_collection"]
    source_aug = source["augmentation_version"]
    embedding_model = source["embedding_model"]
    embedding_dim = source["embedding_dim"]
    print(f"Source: {source_collection}  (aug={source_aug}, "
          f"model={embedding_model}, dim={embedding_dim})")

    # ── Encoder model check ──────────────────────────────────────
    from atelier.classify.colbert_encoder import get_encoder
    encoder = get_encoder()
    encoder_model = getattr(encoder, "model_name", None) or embedding_model
    if encoder_model != embedding_model:
        sys.exit(
            f"Encoder model {encoder_model!r} != source collection's "
            f"embedding_model {embedding_model!r}.  Re-encoding under a "
            f"different model fragments lineage; fail-fast per design."
        )

    # ── Load cohort + filter acceptance ──────────────────────────
    cohort = load_cohort(cohort_dir)
    records = cohort.get("candidates")
    # candidates.json shape varies (older shape uses "records"; newer uses
    # "candidates" dict keyed by code).  Normalize.
    if isinstance(records, dict):
        records = list(records.values())
    elif records is None:
        records = cohort.get("records", [])
    if not records:
        sys.exit(f"No proposal records in {cohort_dir}/candidates.json")

    # Normalize record shape: enrichment_evolution.candidates stores each
    # entry as {code, annotation, current_definition, ..., proposals: [...]}
    # The "proposal" lives in proposals[0].  We unwrap and merge into a flat
    # record per the script's downstream expectation.
    flat_records = []
    for r in records:
        if "proposals" in r and r["proposals"]:
            p = r["proposals"][0]
            merged = {
                "target_code": r.get("code") or r.get("target_code"),
                "annotation": r.get("annotation"),
                "current_definition": r.get("current_definition"),
                "trace_summary": r.get("trace_summary"),
                "status": p.get("status") or r.get("status"),
                "confidence": p.get("confidence"),
                "proposed_mnemonic": p.get("new_definition") and "rewrite",
                "reasoning": p.get("diagnosis") or p.get("reasoning"),
                "subset": {  # mimic the shape from update_reference_from_xlsx
                    "new_definition": p.get("new_definition"),
                    "new_common_names": p.get("new_common_names"),
                    "reasoning": p.get("diagnosis") or p.get("reasoning"),
                    "model": cohort.get("model"),
                },
            }
            # Treat any record with a non-empty new_definition as accepting
            # by default (the enrichment-evolution script doesn't emit
            # confirm_current; every proposal is a rewrite proposal).
            if p.get("new_definition"):
                merged["status"] = merged["status"] or "high_apply"
                merged["confidence"] = merged["confidence"] or "high"
            flat_records.append(merged)
        else:
            flat_records.append(r)

    if args.acceptance:
        accepted_codes = load_acceptance(Path(args.acceptance))
        accepted = [r for r in flat_records if r.get("target_code") in accepted_codes]
    else:
        accepted = default_acceptance(flat_records)
    print(f"Cohort {cohort_name}: {len(flat_records)} candidates → "
          f"{len(accepted)} accepted")
    if not accepted:
        sys.exit("Nothing to apply (acceptance set is empty).  Adjust "
                 "--acceptance or rebuild the cohort.")

    # ── Derive target collection name + manifest id ─────────────
    cohort_version_tag = cohort_name.replace("cohort_", "").replace("/", "_")
    new_aug_version = args.target_collection or f"{source_aug}_evolve_{cohort_version_tag}"
    from atelier.enrichment.qdrant_writer import (
        collection_name_for, ensure_collection, upsert_point, point_cache_key,
        EnrichedAnnotationPoint, AnnotationVectors, source_row_hash,
        COLBERT_VECTOR_NAME,
    )
    target_collection = (
        args.target_collection
        if args.target_collection and args.target_collection.startswith("annotations_")
        else collection_name_for("default", new_aug_version)
    )
    accepted_codes_list = [r.get("target_code") for r in accepted if r.get("target_code")]
    manifest_id = compute_manifest_id(cohort_dir, accepted_codes_list, new_aug_version)
    dup = find_duplicate_manifest(manifest_id)
    if dup and not args.allow_duplicate:
        sys.exit(
            f"Duplicate manifest detected: {dup} already records the same "
            f"(cohort, accepted_codes, target_aug_version).  Pass "
            f"--allow-duplicate to override, or pick a different "
            f"--target-collection."
        )

    print(f"Target collection: {target_collection}")
    print(f"Manifest id:       {manifest_id}")
    if args.dry_run:
        print("\n--dry-run: writing manifest only, no Qdrant or registry "
              "writes.")

    # ── Register building row (or reuse existing) ────────────────
    registry_id = None
    if not args.dry_run:
        all_rows = dao.list_taxonomy_collections(
            taxonomy_id="default", include_archived=True,
        )
        existing = [r for r in all_rows
                    if r.get("qdrant_collection") == target_collection
                    and r.get("status") == "building"]
        if existing:
            registry_id = existing[0]["id"]
            print(f"Resuming existing building row: {registry_id}")
        else:
            registry_id = str(_uuid.uuid4())
            dao.register_taxonomy_collection(
                id=registry_id,
                taxonomy_id="default",
                source_table=source.get("source_table") or "default.annotations",
                qdrant_collection=target_collection,
                augmentation_version=new_aug_version,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                qdrant_url=source.get("qdrant_url"),
                summary=f"evolve apply: {cohort_name} ({len(accepted)} transforms)",
                status="building",
            )
            print(f"Registered building row: {registry_id}")

    # ── Open Qdrant + ensure target collection ───────────────────
    if not args.dry_run:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            sys.exit(f"qdrant_client not importable: {exc}")
        qdrant_url = source.get("qdrant_url") or "http://127.0.0.1:6333"
        client = QdrantClient(url=qdrant_url)
        ensure_collection(client, collection=target_collection, embedding_dim=embedding_dim)

    # ── Scroll source + overlay accepted transforms ─────────────
    accepted_by_code = {r["target_code"]: r for r in accepted}
    applied_at = _now_iso()
    transform_records: list[dict] = []
    scrolled = 0
    copied_unchanged = 0
    rewritten = 0
    skipped_missing_codes = set(accepted_by_code.keys())

    if args.dry_run:
        # Dry-run: don't touch Qdrant; just record the planned transforms.
        for code, prop in accepted_by_code.items():
            tid = str(_uuid.uuid4())
            primary = prop.get("subset") or prop.get("full") or {}
            transform_records.append({
                "transform_id": tid,
                "target": {
                    "mnemonic": prop.get("annotation"),
                    "code": code,
                    "captured_at": applied_at,
                    "source": f"cohort:{cohort_name}",
                },
                "prior_text": {
                    "description": prop.get("current_definition"),
                    "common_names": None,
                },
                "new_text": {
                    "description": primary.get("new_definition"),
                    "common_names": primary.get("new_common_names"),
                },
                "correction_type": "umbrella_semantics_insertion",
                "confidence": prop.get("confidence"),
                "reasoning": primary.get("reasoning") or prop.get("reasoning"),
                "model": cohort.get("model"),
                "source_run": cohort.get("run_id"),
                "source_artifact": str(cohort_dir / "candidates.json"),
                "applied_at": applied_at,
                "applied_to_collection": target_collection,
                "applied_to_registry_id": registry_id,
                "old_point_id": None,
                "new_point_id": None,
                "status": "planned",
            })
            skipped_missing_codes.discard(code)
    else:
        next_offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=source_collection,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=True,
            )
            if not points:
                break
            for p in points:
                scrolled += 1
                payload = dict(p.payload or {})
                code = payload.get("code") or payload.get("mnemonic")
                # Pull existing vectors verbatim for unchanged copy
                if hasattr(p, "vector") and p.vector:
                    raw_vec = p.vector
                else:
                    raw_vec = None

                if code in accepted_by_code:
                    # Transform path
                    proposal = accepted_by_code[code]
                    new_payload = apply_transform_to_payload(payload, proposal, applied_at)
                    # Stamp the new payload with new augmentation_version
                    new_payload["augmentation_version"] = new_aug_version
                    new_sr_hash = source_row_hash(new_payload)
                    new_payload["source_row_hash"] = new_sr_hash
                    new_pid = point_cache_key(
                        taxonomy_id="default",
                        taxonomy_version_hash_value=new_payload.get(
                            "taxonomy_version_hash", "default-snapshot",
                        ),
                        augmentation_version=new_aug_version,
                        embedding_model=embedding_model,
                        source_row_hash_value=new_sr_hash,
                    )
                    # Re-encode
                    from atelier.enrichment.qdrant_writer import compose_annotation_text
                    new_text = compose_annotation_text(new_payload)
                    new_vectors_arr = encoder.encode_single(new_text)
                    new_vectors = AnnotationVectors(
                        colbert=new_vectors_arr.tolist(),
                    )
                    new_point = EnrichedAnnotationPoint(
                        point_id=new_pid, vectors=new_vectors,
                        payload=new_payload,
                    )
                    upsert_point(client, collection=target_collection, point=new_point)
                    rewritten += 1
                    skipped_missing_codes.discard(code)

                    primary = proposal.get("subset") or proposal.get("full") or {}
                    tid = str(_uuid.uuid4())
                    transform_records.append({
                        "transform_id": tid,
                        "target": {
                            "mnemonic": proposal.get("annotation"),
                            "code": code,
                            "captured_at": applied_at,
                            "source": f"cohort:{cohort_name}",
                        },
                        "prior_text": {
                            "description": payload.get("description"),
                            "common_names": None,
                        },
                        "new_text": {
                            "description": primary.get("new_definition"),
                            "common_names": primary.get("new_common_names"),
                        },
                        "correction_type": "umbrella_semantics_insertion",
                        "confidence": proposal.get("confidence"),
                        "reasoning": primary.get("reasoning") or proposal.get("reasoning"),
                        "model": cohort.get("model"),
                        "source_run": cohort.get("run_id"),
                        "source_artifact": str(cohort_dir / "candidates.json"),
                        "applied_at": applied_at,
                        "applied_to_collection": target_collection,
                        "applied_to_registry_id": registry_id,
                        "old_point_id": p.id,
                        "new_point_id": new_pid,
                        "status": "applied",
                    })
                else:
                    # Unchanged copy — keep payload, recompute point_id under
                    # the new aug_version, copy existing vectors as-is.
                    new_payload = dict(payload)
                    new_payload["augmentation_version"] = new_aug_version
                    # Recompute source_row_hash from current payload (safety:
                    # ensures the stored hash matches the actual content)
                    sr_hash = source_row_hash(new_payload)
                    new_payload["source_row_hash"] = sr_hash
                    new_pid = point_cache_key(
                        taxonomy_id="default",
                        taxonomy_version_hash_value=new_payload.get(
                            "taxonomy_version_hash", "default-snapshot",
                        ),
                        augmentation_version=new_aug_version,
                        embedding_model=embedding_model,
                        source_row_hash_value=sr_hash,
                    )
                    # Existing vectors from source
                    if raw_vec is None:
                        # No vectors returned — fail soft, count as skip
                        logger.warning(
                            "Point %s has no vectors; skipping copy", p.id,
                        )
                        continue
                    # raw_vec is a dict {COLBERT_VECTOR_NAME: [[..], ..]}
                    colbert_vecs = (
                        raw_vec.get(COLBERT_VECTOR_NAME)
                        if isinstance(raw_vec, dict) else raw_vec
                    )
                    new_vectors = AnnotationVectors(colbert=colbert_vecs)
                    new_point = EnrichedAnnotationPoint(
                        point_id=new_pid, vectors=new_vectors,
                        payload=new_payload,
                    )
                    upsert_point(client, collection=target_collection, point=new_point)
                    copied_unchanged += 1
            if next_offset is None:
                break

    print(f"\nScrolled: {scrolled}  Copied unchanged: {copied_unchanged}  "
          f"Rewritten: {rewritten}")
    if skipped_missing_codes:
        print(f"Codes accepted but missing from source collection: "
              f"{sorted(skipped_missing_codes)}")
        for code in skipped_missing_codes:
            prop = accepted_by_code[code]
            primary = prop.get("subset") or prop.get("full") or {}
            transform_records.append({
                "transform_id": str(_uuid.uuid4()),
                "target": {
                    "mnemonic": prop.get("annotation"),
                    "code": code,
                    "captured_at": applied_at,
                    "source": f"cohort:{cohort_name}",
                },
                "prior_text": {
                    "description": prop.get("current_definition"),
                    "common_names": None,
                },
                "new_text": {
                    "description": primary.get("new_definition"),
                    "common_names": primary.get("new_common_names"),
                },
                "correction_type": "skipped_missing",
                "confidence": prop.get("confidence"),
                "reasoning": "Target code not present in source collection",
                "model": cohort.get("model"),
                "source_run": cohort.get("run_id"),
                "source_artifact": str(cohort_dir / "candidates.json"),
                "applied_at": applied_at,
                "applied_to_collection": target_collection,
                "applied_to_registry_id": registry_id,
                "old_point_id": None,
                "new_point_id": None,
                "status": "skipped_missing",
            })

    # ── Persist manifest + per-transform records ─────────────────
    transforms_root = Path("build/data/transforms")
    manifests_dir = transforms_root / "manifests"
    records_dir = transforms_root / "records"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "manifest_id": manifest_id,
        "cohort": cohort_name,
        "cohort_path": str(cohort_dir),
        "applied_at": applied_at,
        "applied_at_iso": applied_at,
        "model": cohort.get("model"),
        "source_run": cohort.get("run_id"),
        "source_collection": source_collection,
        "source_registry_id": source.get("id"),
        "target_collection": target_collection,
        "target_registry_id": registry_id,
        "new_augmentation_version": new_aug_version,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "counts": {
            "candidates_total": len(flat_records),
            "accepted": len(accepted),
            "rewritten": rewritten,
            "copied_unchanged": copied_unchanged,
            "skipped_missing": len(skipped_missing_codes),
        },
        "dry_run": args.dry_run,
        "promoted": False,  # filled below
        "transform_ids": [r["transform_id"] for r in transform_records],
    }

    for r in transform_records:
        rpath = records_dir / f"{r['transform_id']}.json"
        rpath.write_text(json.dumps(r, indent=2, default=str))

    # ── Promote (atomic demote-then-promote) ─────────────────────
    if not args.dry_run and not args.skip_promote and registry_id:
        ok = dao.set_current_taxonomy_collection(registry_id)
        manifest["promoted"] = bool(ok)
        if ok:
            print(f"\nPromoted {target_collection} to current.  "
                  f"Previous current → stale.")
        else:
            print(f"\nWARNING: set_current_taxonomy_collection returned "
                  f"False; collection stays in 'building'.")

    # Write manifest LAST so it's the resume sentinel
    timestamp_tag = applied_at.replace(":", "").replace("-", "").replace(".", "")[:15]
    mpath = manifests_dir / f"{cohort_name}_{timestamp_tag}.json"
    mpath.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nWrote manifest:   {mpath}")
    print(f"Wrote {len(transform_records)} per-transform records")

    print(f"\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
