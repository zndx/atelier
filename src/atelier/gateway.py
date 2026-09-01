"""HTTP gateway — serves React frontend and bridges REST to gRPC.

In production (CML), this is the process that binds to CDSW_APP_PORT.
It serves the compiled React build from ui/dist/ and proxies /api/*
requests to the co-located gRPC server.
"""

import asyncio
import time
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from atelier.client import AtelierClient
from atelier.proto import atelier_pb2

_log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Seed OOTB sample + discover Hive sources + start cleanup on boot."""

    # Seed + discover with retry — the database may still be starting
    # (PGlite takes a few seconds after the shell probe passes).
    async def _seed_with_retry() -> None:
        # Every seed step that touches the DB, Hive JDBC, or ``fsm_start``
        # runs via ``asyncio.to_thread`` — otherwise a slow Hive probe or
        # the nautilus attach spin-loop blocks the event loop and the
        # HTTP gateway becomes unresponsive before it finishes booting.
        for attempt in range(5):
            try:
                await asyncio.to_thread(_seed_sample_source)
                _log.info("OOTB sample seed: OK")
                break
            except Exception as exc:
                if attempt < 4:
                    _log.info("DB not ready for seed (attempt %d/5): %s", attempt + 1, exc)
                    await asyncio.sleep(2)
                else:
                    _log.warning("Sample source seeding skipped after 5 attempts: %s", exc)
        try:
            await asyncio.to_thread(_seed_meta_tagging_source)
        except Exception as exc:
            _log.warning("Meta-tagging source seeding skipped: %s", exc)
        try:
            await asyncio.to_thread(_seed_sdg_source)
        except Exception as exc:
            _log.warning("SDG corpora source seeding skipped: %s", exc)
        try:
            await asyncio.to_thread(_discover_and_register_hive_sources)
        except Exception as exc:
            _log.warning("Hive source discovery skipped: %s", exc)
        try:
            await asyncio.to_thread(_seed_classify_data_source)
        except Exception as exc:
            _log.warning("Classify data source seeding skipped: %s", exc)
        # Reconcile DB against ``build/results/`` BEFORE auto-start so
        # any orphaned runs (DB writes that failed mid-pipeline) become
        # visible in the UI alongside the new run that auto-start
        # produces.  Idempotent — runs already represented in the DB
        # cost one read each.
        try:
            await asyncio.to_thread(_sync_orphaned_runs)
        except Exception as exc:
            _log.warning("Run-state sync skipped: %s", exc)
        # Drain the restart-ready task queue BEFORE auto-start.  Tasks
        # pre-enqueued from the Session pod or via the Web Terminal Agent
        # (apply forward transforms, verify, render change-management
        # guide) must land before the pipeline runs so it reads the
        # post-apply Qdrant collection rather than racing it.  fsm_start
        # itself gates on a clean queue (so warm-state enqueues from
        # the Agent SDK also serialize correctly), but draining here at
        # cold-start keeps the gateway-startup logs legible and means
        # the first pipeline run doesn't pay the drain cost.
        try:
            await asyncio.to_thread(_kick_task_queue)
        except Exception as exc:
            _log.warning("Task queue dispatch skipped: %s", exc)
        try:
            await asyncio.to_thread(_maybe_auto_start_classify)
        except Exception as exc:
            _log.warning("Classify auto-start skipped: %s", exc)

    seed_task = asyncio.create_task(_seed_with_retry())

    # Background task: clean up idle terminal sessions every 60s.
    async def _session_cleanup_loop() -> None:
        from atelier.terminal import cleanup_idle_sessions
        while True:
            await asyncio.sleep(60)
            try:
                await cleanup_idle_sessions()
            except Exception:
                pass

    cleanup_task = asyncio.create_task(_session_cleanup_loop())

    # Forensics sampler: append memory/load/FSM/queue/RSS state to
    # .app/forensics/samples.jsonl every 10s (override via
    # ATELIER_FORENSICS_INTERVAL_S).  Survives across runs (appends;
    # rotates at 50 MB).  Reader: .app/forensics/digest.py.  See
    # src/atelier/forensics.py.
    try:
        from atelier import forensics as _forensics
        forensics_task = _forensics.start_sampling_task()
    except Exception as exc:  # noqa: BLE001
        _log.warning("Forensics sampler skipped: %s", exc)
        forensics_task = None

    yield
    if forensics_task is not None:
        forensics_task.cancel()
    seed_task.cancel()
    cleanup_task.cancel()


from atelier import __version__ as _atelier_version

app = FastAPI(title="Atelier", version=_atelier_version, lifespan=_lifespan)

_project_root = Path(__file__).resolve().parent.parent.parent

_client: AtelierClient | None = None
_engine = None  # Cached SQLAlchemy engine for /api/status health checks


def _get_client() -> AtelierClient:
    global _client
    if _client is None:
        _client = AtelierClient()
    return _client


def _reset_client() -> None:
    """Drop the cached gRPC client so the next call opens a fresh channel."""
    global _client
    _client = None


def _get_status_engine():
    """Return a cached SQLAlchemy engine for /api/status health checks.

    Creating a new engine per request hits a pglite-socket@0.0.13 bug where
    rapid connect/disconnect cycles cause "server closed the connection
    unexpectedly". Reusing a pooled engine with pool_pre_ping lets us both
    detect stale connections and avoid the setup-loop hazard.
    """
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine
        from atelier.config import load_config
        _engine = create_engine(
            load_config().db_url,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=2,
            max_overflow=0,
            connect_args={"connect_timeout": 10},
        )
    return _engine


def _reset_status_engine() -> None:
    """Drop the cached engine so the next call rebuilds the pool."""
    global _engine
    if _engine is not None:
        try:
            _engine.dispose()
        except Exception:
            pass
    _engine = None


def _error_envelope(detail: str, *, status: int = 503) -> "JSONResponse":
    """Return a JSON error envelope so the frontend never has to parse
    ``Internal Server Error`` as JSON."""
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=status, content={"error": detail})


def _seed_sample_source() -> None:
    """Register OOTB sample data as version 1 with the bundled reference parquet.

    Called once at gateway startup.  Uses a stable dataset_id
    ``ootb-sample-v1`` so that (a) fresh installs get the bundled
    pre-classified parquet auto-registered, and (b) existing deployments
    whose v1 row still has ``parquet_path=""`` (pre-bundling) are
    converged on the bundled artifact without trampling any
    user-generated v2+ runs.

    The bundled parquet lives at ``data/sample/atelier_embeddings.parquet``
    and is produced from a reference pipeline run against the same
    ``data/sample/tables/*.csv`` that OOTB classification uses.  Ships
    with the repo so first-time users can explore the Embeddings page
    before their first real run completes.  If the file is missing
    (e.g. source checkout without the bundle), we fall back to
    registering v1 with an empty parquet path — same behaviour as
    pre-bundling.
    """
    try:
        from atelier.db.dao import AtelierDao
        from atelier.classify.sampler import sample_source_stats
    except Exception:
        return  # DB not available yet (e.g., migrations haven't run)

    try:
        dao = AtelierDao()
        from atelier.classify.sampler import _SAMPLE_DIR  # type: ignore[attr-defined]
        sample_dir = Path(_SAMPLE_DIR).resolve()
        dao.force_upsert_data_source(
            source_id="ootb-sample",
            source_type="filesystem",
            display_name="Sample",
            source_uri=f"file://{sample_dir}",
            vocabulary_mode="universal",
        )

        stats = sample_source_stats()
        if not stats["has_data"]:
            return  # No sample data to seed

        # Stable ID lets us idempotently converge v1 onto the bundled
        # parquet path across restarts and upgrades.  Distinct from the
        # uuid-based IDs that user-generated runs (v2+) create.
        stable_v1_id = "ootb-sample-v1"
        bundled_parquet = sample_dir / "atelier_embeddings.parquet"
        bundled_path_str = str(bundled_parquet) if bundled_parquet.exists() else ""

        versions = dao.list_dataset_versions("ootb-sample")
        existing_v1 = next((v for v in versions if v.get("id") == stable_v1_id), None)
        user_versions = [v for v in versions if v.get("id") != stable_v1_id]

        # Refresh source metadata unconditionally — stats are live mount
        # state and must converge even when the v1 dataset row is already
        # correct (a stale-metadata row otherwise persists forever behind
        # the short-circuit below).
        dao.update_data_source_metadata("ootb-sample", json.dumps({
            "table_count": stats["table_count"],
            "column_count": stats["column_count"],
            "bundled_parquet": bool(bundled_path_str),
        }))

        if (
            existing_v1
            and existing_v1.get("parquet_path") == bundled_path_str
            and existing_v1.get("row_count") == stats["column_count"]
        ):
            return  # Already correctly seeded; nothing to do

        # Only claim is_active when no user-generated run already owns
        # it — otherwise we'd silently demote the user's latest run on
        # every gateway restart.
        any_user_active = any(v.get("is_active") for v in user_versions)
        row_count = stats["column_count"]
        description = (
            f"{stats['table_count']} tables, {stats['column_count']} columns "
            f"from expanded ontology"
        )
        if bundled_path_str:
            description += " (pre-classified reference bundle)"
        dao.upsert_dataset(
            dataset_id=stable_v1_id,
            name="Sample v1",
            parquet_path=bundled_path_str,
            description=description,
            row_count=row_count,
            source_id="ootb-sample",
            version_number=1,
            is_active=not any_user_active,
            summary=f"{stats['table_count']} tables, {stats['column_count']} columns",
        )


        _log.info(
            "Seeded OOTB sample v1: %d tables, %d columns, bundled_parquet=%s",
            stats["table_count"], stats["column_count"], bool(bundled_path_str),
        )
    except Exception as exc:
        _log.warning("Sample source seeding failed: %s", exc)


def _seed_meta_tagging_source() -> None:
    """Register the Meta-tagging source when a local mount is available.

    Meta-tagging data is a private reference corpus.  Resolution prefers
    ``<repo>/build/meta-tagging/`` (UAT snapshot, gitignored) and falls
    back to ``~/local/tmp/meta-tagging/``; both can be overridden via
    ``ATELIER_META_TAGGING_DIR`` / ``cfg.classify_meta_tagging_dir``.
    Its contents — annotation labels, sample values, any
    numeric codes beyond what's already in the pipeline — must never
    land in git.  This seeder reads stats only (row counts, mount
    path) and registers the source so the UI selector can offer it.

    Silent no-op when no mount is found — the source stays hidden
    on deployments that don't have the reference data.
    """
    try:
        from atelier.config import load_config
        from atelier.db.dao import AtelierDao
        from atelier.classify.meta_tagging_source import (
            meta_tagging_stats,
            resolve_meta_tagging_mount,
        )
    except Exception:
        return

    try:
        cfg = load_config()
        mount = resolve_meta_tagging_mount(cfg)
        if mount is None:
            return

        stats = meta_tagging_stats(mount)
        if not stats["has_data"]:
            return

        dao = AtelierDao()
        mount_abs = Path(mount).resolve()
        dao.force_upsert_data_source(
            source_id="meta-tagging",
            source_type="filesystem",
            display_name="Meta-tagging",
            source_uri=f"file://{mount_abs}",
            vocabulary_mode="universal",
            # vocab_uri pins the annotations.csv path so
            # pipeline._load_vocabulary's file:// branch activates.
            vocab_uri=f"file://{mount_abs / 'annotations.csv'}",
        )
        dao.update_data_source_metadata("meta-tagging", json.dumps({
            "table_count": stats["table_count"],
            "column_count": stats["column_count"],
            "mount": stats["mount"],
        }))

        versions = dao.list_dataset_versions("meta-tagging")
        if not versions:
            import uuid
            dataset_id = str(uuid.uuid4())[:8]
            dao.upsert_dataset(
                dataset_id=dataset_id,
                name="Meta-tagging v1",
                parquet_path="",
                description=(
                    f"{stats['table_count']} tables, {stats['column_count']} "
                    f"columns (private reference corpus — not committed)"
                ),
                row_count=stats["column_count"],
                source_id="meta-tagging",
                version_number=1,
                is_active=True,
                summary=f"{stats['table_count']} tables, {stats['column_count']} columns",
            )

        _log.info(
            "Seeded Meta-tagging source: %d tables, %d columns",
            stats["table_count"], stats["column_count"],
        )
    except Exception as exc:
        _log.warning("Meta-tagging source seeding failed: %s", exc)


def _seed_sdg_source() -> None:
    """Register the SDG corpora sample when the submodule is present.

    ``external/sdg-corpora`` ships per-collection relational CSV bundles
    (``corpus/collections/<name>/tables/*.csv``) plus the SKOS-derived
    annotations vocabulary (``vocabulary/annotations.csv``).  Seeding
    one collection as a filesystem source gives a turn-key
    classification target: the pipeline classifies the collection's
    columns blind against the SKOS vocabulary — the per-column
    reference codes are withheld upstream as the scoring key (see the
    submodule README).

    Collection selection: ``cfg.sdg_collection`` is a name or prefix;
    the first sorted match with a ``tables/`` directory wins.  Silent
    no-op when the submodule isn't initialized — the source stays
    hidden on checkouts without it.
    """
    try:
        from atelier.config import load_config
        from atelier.db.dao import AtelierDao
    except Exception:
        return

    try:
        cfg = load_config()
        root = Path(__file__).resolve().parent.parent.parent
        corpora = root / "external" / "sdg-corpora"

        # Preferred substrate: the RI-verified, taxonomy-sound sample
        # built by `just sdg-sample` (atelier.sdg.sample).  Its manifest
        # carries the corpus pin, profile, and vocab signature — the
        # alignment record for every downstream artifact.
        pointer = root / cfg.artifact_root / "sdg_sample" / "current.json"
        if not pointer.exists() and (
            corpora / "corpus" / "collections"
        ).is_dir():
            # First boot on a fresh clone: derive the sample in-line.
            # Pure-python (manifest reads + CSV copies), a few seconds
            # — this is what makes `devenv up` → "start classification"
            # sufficient with no manual `just sdg-sample` step.
            try:
                from atelier.sdg.sample import PROFILES, build_sample
                build_sample(PROFILES["macbook"])
                _log.info("Built SDG sample (macbook profile) on first boot")
            except Exception as exc:
                _log.error(
                    "SDG sample auto-build FAILED (%s) — falling back to "
                    "the raw collection bundle.  NOTE: the raw fallback "
                    "carries the full vocabulary; first-run "
                    "pre-conditioning will be much heavier than the "
                    "sampled path.  Fix with `just sdg-sample`.", exc,
                )
        if pointer.exists():
            try:
                ptr = json.loads(pointer.read_text())
                sample_dir = Path(ptr["path"])
                manifest = json.loads(
                    (sample_dir / "manifest.json").read_text())
                tables_dir = (sample_dir / "tables").resolve()
                vocab_csv_s = (sample_dir / "annotations.csv").resolve()
                if not tables_dir.is_dir() or not vocab_csv_s.is_file():
                    raise FileNotFoundError(
                        f"sample artifacts missing under {sample_dir}")
            except Exception as exc:
                _log.error(
                    "SDG sample pointer %s is broken (%s) — refusing to "
                    "seed a half-valid source; rebuild with `just "
                    "sdg-sample` or remove the pointer.", pointer, exc,
                )
                return
            dao = AtelierDao()
            entity_count = (
                manifest["table_count"] + manifest["column_count"])
            dao.force_upsert_data_source(
                source_id="sdg-corpora",
                source_type="filesystem",
                display_name=f"SDG sample ({entity_count} entities)",
                source_uri=f"file://{tables_dir}",
                vocabulary_mode="universal",
                vocab_uri=f"file://{vocab_csv_s}",
            )
            dao.update_data_source_metadata("sdg-corpora", json.dumps({
                "table_count": manifest["table_count"],
                "column_count": manifest["column_count"],
                "mount": str(tables_dir),
                "corpus_commit": manifest["corpus_commit"],
                "profile": manifest["profile"]["name"],
                "vocab_sig": manifest["vocab_sig"],
                "term_count": manifest["taxonomy"]["term_count"],
                "ri_verified": manifest["referential_integrity"]["verified"],
            }))
            versions = dao.list_dataset_versions("sdg-corpora")
            if not versions:
                import uuid
                dao.upsert_dataset(
                    dataset_id=str(uuid.uuid4())[:8],
                    name="SDG sample v1",
                    parquet_path="",
                    description=(
                        f"{manifest['table_count']} tables, "
                        f"{manifest['column_count']} columns, "
                        f"{manifest['taxonomy']['term_count']}-term SKOS "
                        f"subset (RI-verified, pin "
                        f"{manifest['corpus_commit'][:12]})"
                    ),
                    row_count=manifest["column_count"],
                    source_id="sdg-corpora",
                    version_number=1,
                    is_active=True,
                    summary=(
                        f"{manifest['table_count']} tables, "
                        f"{manifest['column_count']} columns"
                    ),
                )
            _log.info(
                "Seeded SDG sample source: %d tables, %d columns, %d "
                "terms (pin %s, profile %s)",
                manifest["table_count"], manifest["column_count"],
                manifest["taxonomy"]["term_count"],
                manifest["corpus_commit"][:12],
                manifest["profile"]["name"],
            )
            return

        # Fallback: raw collection bundle (no derived sample yet).
        vocab_csv = corpora / "vocabulary" / "annotations.csv"
        collections_dir = corpora / "corpus" / "collections"
        if not vocab_csv.exists() or not collections_dir.is_dir():
            return

        want = cfg.sdg_collection
        match = None
        for d in sorted(collections_dir.iterdir()):
            if d.is_dir() and (d.name == want or d.name.startswith(want)):
                if (d / "tables").is_dir():
                    match = d
                    break
        if match is None:
            _log.info(
                "SDG corpora present but no collection matches %r", want,
            )
            return

        tables_dir = (match / "tables").resolve()
        csvs = sorted(tables_dir.glob("*.csv"))
        if not csvs:
            return
        column_count = 0
        for f in csvs:
            try:
                with open(f, encoding="utf-8") as fh:
                    header = fh.readline().rstrip("\n")
                column_count += len(header.split(",")) if header else 0
            except Exception:
                pass

        dao = AtelierDao()
        dao.force_upsert_data_source(
            source_id="sdg-corpora",
            source_type="filesystem",
            display_name=f"SDG: {match.name}",
            source_uri=f"file://{tables_dir}",
            vocabulary_mode="universal",
            # vocab_uri pins the SKOS annotations.csv so
            # pipeline._load_vocabulary's file:// branch activates —
            # classification targets the SDG vocabulary, not the
            # universal fixture.
            vocab_uri=f"file://{vocab_csv.resolve()}",
        )
        # The pinned submodule commit IS the corpus version — each
        # upstream commit is a reproducible convergence snapshot, and
        # upstream iterates rapidly.  Stamp it so every run's artifacts
        # trace to the exact corpus iteration they saw.
        corpus_commit = ""
        try:
            import subprocess
            corpus_commit = subprocess.run(
                ["git", "-C", str(corpora), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        except Exception:
            pass

        dao.update_data_source_metadata("sdg-corpora", json.dumps({
            "table_count": len(csvs),
            "column_count": column_count,
            "mount": str(tables_dir),
            "collection": match.name,
            "corpus_commit": corpus_commit,
        }))

        versions = dao.list_dataset_versions("sdg-corpora")
        if not versions:
            import uuid
            dao.upsert_dataset(
                dataset_id=str(uuid.uuid4())[:8],
                name="SDG v1",
                parquet_path="",
                description=(
                    f"{len(csvs)} tables, {column_count} columns "
                    f"({match.name}, SKOS annotations vocabulary)"
                ),
                row_count=column_count,
                source_id="sdg-corpora",
                version_number=1,
                is_active=True,
                summary=f"{len(csvs)} tables, {column_count} columns",
            )

        _log.info(
            "Seeded SDG corpora source: %s (%d tables, %d columns)",
            match.name, len(csvs), column_count,
        )
    except Exception as exc:
        _log.warning("SDG corpora source seeding failed: %s", exc)


def _classify_source_id(connection: str, database: str) -> str:
    """Return the canonical data-source id for a Hive (conn, db) pair.

    Single source of truth shared by env-seeded and discovery paths so
    the Data Platform panel shows exactly one row per Hive endpoint.
    Matches the discovery format (``{connection}/{database}``) — which
    is what ``fsm_start`` already splits on "/" — so whichever seed
    function runs first, the other is a no-op via
    ``get_or_create_data_source``.
    """
    return f"{connection}/{database}"


def _seed_classify_data_source() -> None:
    """Seed a ``data_source`` row from ATELIER_CLASSIFY_CONNECTION
    + ATELIER_CLASSIFY_DATABASE env vars so the env-driven default
    appears in the Data Platform panel as an editable row.

    Makes the env vars behave like *defaults* — visible via
    ``/api/data-sources``, selectable as ``activeSourceId``, and
    overridable from the UI (vocab_uri edits, archival, etc).

    Idempotent via ``get_or_create_data_source`` keyed on
    ``_classify_source_id`` — unified with ``_discover_and_register_
    hive_sources`` so a connection named in both the env-seed path
    and the discovery path produces exactly one row.
    """
    try:
        from atelier.config import load_config
        from atelier.db.dao import AtelierDao
    except Exception:
        return  # DB or config not available yet

    cfg = load_config()
    connection = (getattr(cfg, "classify_connection_name", "") or "").strip()
    database = (getattr(cfg, "classify_database", "") or "").strip()
    if not connection or not database:
        return

    source_id = _classify_source_id(connection, database)
    source_uri = source_id  # identical shape; fsm_start splits on "/"
    display_name = f"Hive: {connection}/{database}"

    try:
        dao = AtelierDao()
        dao.get_or_create_data_source(
            source_id=source_id,
            source_type="hive",
            display_name=display_name,
            source_uri=source_uri,
            vocabulary_mode="hive",
            vocab_uri=f"{database}.annotations",
            metadata=json.dumps({
                "connection": connection,
                "database": database,
                "seeded_from_env": True,
            }),
        )
        _log.info(
            "Seeded classify data source from env: %s (%s → %s.annotations)",
            source_id, source_uri, database,
        )
    except Exception as exc:
        _log.warning("Classify data source seed failed: %s", exc)


def _active_source_state_path() -> "Path":
    """Return the path of the persistent last-active-source state file.

    Lives at ``build/state/last_active_source.txt`` under the project's
    build directory — on CAI ``/home/cdsw/build/...`` is part of the
    user's persistent home volume, so the file survives container
    restarts.  Independent of DAO availability and PGlite container
    ephemerality.
    """
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "build" / "state" / "last_active_source.txt"


def _persist_active_source_id(source_id: str | None) -> None:
    """Write *source_id* to the persistent state file.

    Called from :func:`fsm_start` after a successful dispatch so the
    user's last expressed classification intent survives DAO outages
    and container restarts.  Failures are logged at DEBUG and
    swallowed — persistence is best-effort, never blocks dispatch.
    """
    if not source_id:
        return
    try:
        path = _active_source_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source_id.strip() + "\n")
        _log.debug("Persisted last-active source_id %r → %s", source_id, path)
    except Exception as exc:
        _log.debug("Failed to persist active source_id: %s", exc)


def _read_persisted_source_id() -> str | None:
    """Read the persisted last-active source_id, or None.

    Read in :func:`_last_user_selected_source_id` before the DAO
    query so the auto-start path works even when the FSM runs
    table is empty / unreachable (PGlite ephemeral or DAO not yet
    attached at lifespan startup).
    """
    try:
        path = _active_source_state_path()
        if not path.exists():
            return None
        content = path.read_text().strip()
        return content or None
    except Exception as exc:
        _log.debug("Failed to read persisted active source_id: %s", exc)
        return None


def _last_user_selected_source_id() -> str | None:
    """Return the source_id of the user's last-expressed classification intent.

    Two-step resolution, ordered by reliability:

    1. **Persistent state file**
       (``build/state/last_active_source.txt``) — written by
       :func:`fsm_start` on every successful dispatch, so the
       user's last selection survives DAO unavailability, PGlite
       ephemerality, and container restarts.  Read first because
       it's the most authoritative signal of *recent intent*.

    2. **DAO FSM-runs query** —
       ``AtelierDao.list_fsm_runs()`` (ordered by ``started_at``
       desc, see ``db/dao.py:list_fsm_runs``).  Falls back to this
       when the state file is missing — e.g. initial deploy of a
       newer binary onto an environment that previously ran an
       older binary without the persistence path.

    Returns ``None`` only when both signals fail; callers fall
    back to env-driven defaults.  Logs at INFO on resolution and
    at WARNING on full fallback so AMP operator logs reveal which
    path fired without source-tree introspection.
    """
    persisted = _read_persisted_source_id()
    if persisted:
        _log.info(
            "Auto-start: resolved last user-selected source_id from state file: %s",
            persisted,
        )
        return persisted

    try:
        from atelier.db.dao import AtelierDao
        runs = AtelierDao().list_fsm_runs()
    except Exception as exc:
        _log.warning(
            "Auto-start: state file empty AND DAO unreachable (%s: %s) — "
            "falling back to env-driven defaults.  Subsequent manual "
            "/api/fsm/start calls will populate the state file.",
            type(exc).__name__, exc,
        )
        return None

    for run in runs:
        sid = run.get("source_id")
        if sid:
            _log.info(
                "Auto-start: resolved last user-selected source_id from "
                "FSM run history: %s (run %s)",
                sid, str(run.get("id", ""))[:8],
            )
            return sid

    _log.warning(
        "Auto-start: state file empty AND DAO has no runs with a "
        "source_id (saw %d run(s)) — falling back to env-driven "
        "defaults.",
        len(runs),
    )
    return None


def _sync_orphaned_runs() -> None:
    """Reconcile DB rows against ``build/results/`` at gateway startup.

    Idempotent — see ``atelier.db.sync.sync_filesystem_to_db``.  When
    a previous run's DB writes failed mid-pipeline (transient PGlite
    hiccup, FK ordering bug, etc.), this picks up the orphaned
    artifacts on disk and registers whatever rows are missing so the
    UI shows the run.  Runs already represented in the DB pay one
    read per row and are reported as ``already_registered``.
    """
    from atelier.db.sync import sync_filesystem_to_db
    results_root = _project_root / "build" / "results"
    if not results_root.is_dir():
        return
    report = sync_filesystem_to_db(results_root)
    _log.info(report.summary_line())
    for outcome in report.outcomes:
        if outcome.artifact_set == "registered" or outcome.dataset == "registered":
            _log.info(
                "sync: %s — artifact_set=%s dataset=%s source_id=%s%s",
                outcome.run_id, outcome.artifact_set, outcome.dataset,
                outcome.source_id,
                f" ({outcome.note})" if outcome.note else "",
            )
        elif outcome.artifact_set == "error" or outcome.dataset == "error":
            _log.warning(
                "sync: %s — artifact_set=%s dataset=%s%s",
                outcome.run_id, outcome.artifact_set, outcome.dataset,
                f" ({outcome.note})" if outcome.note else "",
            )


def _maybe_auto_start_classify() -> None:
    """Kick off a classification run on boot when configured.

    Gated by ``ATELIER_CLASSIFY_AUTO_START`` (HOCON: ``classify.auto_start``).
    Dispatches unconditionally — if the environment isn't ready the
    pipeline itself will error out and the FSM transitions to ERROR
    with the underlying Bedrock/Anthropic exception in its ``error``
    field.  An honest failure visible in the Status panel is a better
    operator signal than a "skipped" log line that might go unnoticed.

    Source-of-truth precedence on auto-start:

      1. **Last user-selected source** — the ``source_id`` of the
         most recent FSM run, if any.  This captures whatever the
         operator last picked via the Status / Data Platform UI; an
         AMP restart that re-fires auto-start should honor the
         user's current configuration rather than regress to the
         deployment-time env defaults.

      2. **Env-driven default** — ``ATELIER_CLASSIFY_CONNECTION`` +
         ``ATELIER_CLASSIFY_DATABASE``, used only when there are no
         prior runs (initial deploy) or when DAO lookup fails.

    The pipeline's LLM backends set explicit boto3 timeouts
    (connect=15s, read=180s) so a cold-boot egress blackhole surfaces
    as a catchable ``ConnectTimeoutError`` in ~15 seconds rather than
    hanging indefinitely.  The halving retry then engages; if the
    hang is systemic the pipeline exhausts its retries and lands on
    ERROR with a clear message.

    ``fsm_start`` dispatches the pipeline to a daemon thread and
    returns quickly; the existing in-flight guard
    (IDLE/CONVERGED/ERROR) prevents double-launch if the lifespan
    replays on reload.
    """
    try:
        from atelier.config import load_config
    except Exception:
        return
    cfg = load_config()
    if not getattr(cfg, "classify_auto_start", False):
        return

    # Prefer the user's last-selected source over env defaults so a
    # restart honors the operator's current configuration.
    last_source_id = _last_user_selected_source_id()
    if last_source_id:
        result = fsm_start(source_id=last_source_id)
        _log.info(
            "Classify auto-start dispatched (last user-selected): %s → %s",
            last_source_id, result,
        )
        return

    connection = (getattr(cfg, "classify_connection_name", "") or "").strip()
    database = (getattr(cfg, "classify_database", "") or "").strip()
    if not connection or not database:
        # Fresh environment with no Hive config: the seeded SDG sample
        # is the canonical first-run substrate — auto-start against it
        # when present so a zero-config boot converges unattended.
        try:
            from atelier.db.dao import AtelierDao
            if AtelierDao().get_data_source("sdg-corpora"):
                result = fsm_start(source_id="sdg-corpora")
                _log.info(
                    "Classify auto-start dispatched (fresh env → SDG "
                    "sample): %s", result,
                )
                return
        except Exception as exc:
            _log.warning("SDG auto-start probe failed: %s", exc)
        _log.warning(
            "classify_auto_start=true but CONNECTION/DATABASE unset and "
            "no SDG sample source; skipping"
        )
        return

    source_id = _classify_source_id(connection, database)
    result = fsm_start(source_id=source_id)
    _log.info(
        "Classify auto-start dispatched (env default — no prior runs): %s → %s",
        source_id, result,
    )


def _kick_task_queue() -> None:
    """Drain the restart-ready task queue SYNCHRONOUSLY at lifespan boot.

    See src/atelier/task_queue.py.  Tasks pre-enqueued from the Session
    pod or the Web Terminal Agent (apply forward transforms, verify,
    render change-management guide, etc.) drain here BEFORE auto-start
    fires the classification pipeline — otherwise the pipeline races
    the apply step and reads the pre-apply Qdrant collection.
    Idempotent: handlers detect already-completed work and short-circuit.
    Crash-safe: orphaned ``running`` entries left behind by an AMP
    restart are recovered back to ``pending`` on the next boot.

    Blocking: the gateway-startup logs are legible (the drain runs in
    a single thread under to_thread, finishes, auto-start proceeds).
    FastAPI's lifespan reaches ``yield`` after this returns; the
    server doesn't accept requests until the drain completes.  For
    long-running cohort applies that's tens of seconds of additional
    boot time — acceptable for the consistency guarantee.

    Failed tasks are recorded under ``build/data/task_queue/failed/``
    for operator review; surface via ``python -m atelier.task_queue list``.
    Subsequent ``fsm_start`` calls also run a drain+check via the
    ``drain_then_check`` gate — handles the case where the operator
    enqueues tasks via the Web Terminal Agent after boot.
    """
    try:
        from atelier import task_handlers  # noqa: F401  — registers handlers
        from atelier.task_queue import drain, list_handlers
    except Exception as exc:  # noqa: BLE001
        _log.warning("Task queue module unavailable: %s", exc)
        return
    _log.info(
        "Task queue: %d handlers registered; draining pending tasks "
        "synchronously before auto-start", len(list_handlers()),
    )
    try:
        result = drain()
        _log.info("Task queue drain complete: %s", result)
    except Exception as exc:  # noqa: BLE001
        _log.error("Task queue drain crashed: %s", exc, exc_info=True)


def _discover_and_register_hive_sources() -> None:
    """Probe configured Hive connections and register discovered annotation sources.

    For each connection × database that contains an ``annotations`` table
    with the expected schema, registers a data source via the DAO.
    Idempotent — ``get_or_create_data_source`` is a no-op for existing IDs.
    """
    try:
        from atelier.config import load_config
        from atelier.data.connections import discover_hive_sources
        from atelier.db.dao import AtelierDao
    except Exception:
        return  # DB or config not available yet

    cfg = load_config()
    if not cfg.cml_data_connection_names:
        return

    discoveries = discover_hive_sources(cfg)
    if not discoveries:
        return

    try:
        dao = AtelierDao()
        for d in discoveries:
            dao.get_or_create_data_source(
                source_id=d["source_id"],
                source_type="hive",
                display_name=d["display_name"],
                source_uri=d["source_id"],
                vocabulary_mode="hive",
                vocab_uri=f"{d['database']}.annotations",
                metadata=json.dumps({
                    "connection": d["connection"],
                    "database": d["database"],
                    "annotation_count": d["annotation_count"],
                    "schema_format": d["schema_format"],
                }),
            )
            _log.info(
                "Registered Hive source: %s (%d annotations, %s format)",
                d["source_id"], d["annotation_count"], d["schema_format"],
            )
    except Exception as exc:
        _log.warning("Hive source registration failed: %s", exc)


# ── REST → gRPC bridge ────────────────────────────────────────────


# NOTE: endpoints below are declared `def` (not `async def`) so FastAPI runs
# them in its threadpool. The gRPC stub calls are synchronous blocking
# operations; running them on the event loop would serialize every request
# and hang the gateway if any one call stalls.


@app.get("/api/atelier/v1/federation/surfaces")
def federation_surfaces(request: Request):
    """Waffle roster: this engine plus peers that advertise a primary UI.

    Varnish loop-through (federated menu pattern): the RAW roster is
    cached at the *_origin route (ttl+grace = instant serves, background
    refresh); per-request Host rebasing happens HERE, after the cache, so
    one browser's access host never leaks into another's links. Falls
    back to the bespoke TTL/last-good roster when varnish is absent.
    Discovery matches Signals: Status the local PEERS directory in
    parallel, skip a target that is down this round, one-hop PEERS only
    from live engines. LAN IP Host rebases this-host links so the ZT
    name need not resolve.
    """
    from atelier.engine.s2s import (
        collect_peer_surfaces_cached,
        rebase_items_for_request,
    )

    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or ""
    import httpx

    vport = (os.environ.get("VARNISH_PORT") or "6094").strip()
    url = f"http://127.0.0.1:{vport}/api/atelier/v1/federation/surfaces_origin"
    try:
        resp = httpx.get(url, timeout=6.0)
        if resp.status_code == 200:
            payload = resp.json()
            items = payload.get("items") or []
            return {"items": rebase_items_for_request(items, host)}
    except Exception:  # noqa: BLE001 — varnish absent → bespoke fallback
        pass
    try:
        items = collect_peer_surfaces_cached()
    except Exception as exc:  # noqa: BLE001 — waffle degrades to empty
        return {"items": [], "error": str(exc)}
    return {"items": rebase_items_for_request(items, host)}


@app.get("/api/atelier/v1/federation/surfaces_origin")
def federation_surfaces_origin():
    """Raw roster for the varnish cache — the live collector, no rebasing.

    Failures must be real HTTP errors (502), never 200+error, so the VCL
    error discipline holds: a failed background refresh is abandoned and
    the last good object keeps serving.
    """
    from atelier.engine.s2s import collect_peer_surfaces

    try:
        return {"items": collect_peer_surfaces()}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/health")
def health():
    try:
        client = _get_client()
        resp = client.stub.HealthCheck(
            atelier_pb2.HealthCheckRequest(), timeout=5
        )
        return {"status": resp.status, "version": resp.version}
    except Exception as exc:
        _reset_client()
        return _error_envelope(f"gRPC health check failed: {exc}")


# ── PGlite supervisor surfaces ────────────────────────────────────
# bin/pglite-supervisor.sh respawns pglite when it dies and writes
# state to .app/pglite-supervisor.state.  These endpoints expose that
# state to the UI's Status panel + restart button.  When pglite is
# *not* the active database (operator pointed ATELIER_DB_URL at an
# external Postgres), the state file won't exist and we report
# "unmanaged" so the UI hides the chip + button.

_PGLITE_STATE_FILE = "/home/cdsw/.app/pglite-supervisor.state"
_PGLITE_RESTART_SENTINEL = "/home/cdsw/.app/pglite-supervisor.restart"


def _probe_pglite(port: int, timeout: float = 1.0) -> dict:
    """Quick TCP probe.  Returns ``{listening, error}``."""
    import socket
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        s.close()
        return {"listening": True, "error": None}
    except Exception as exc:
        return {"listening": False, "error": str(exc)}


@app.get("/api/pglite/status")
def pglite_status():
    """Supervisor state + live TCP probe for the Status page chip.

    Returns ``managed=False`` when pglite isn't the active backend
    (external Postgres, sqlite tier-0 tests) so the UI knows to hide
    the chip rather than showing a perpetually-unknown state.
    """
    import json
    import os

    if not os.path.exists(_PGLITE_STATE_FILE):
        return {
            "managed": False,
            "reason": "no supervisor state file — pglite is not the active backend or the supervisor hasn't started yet",
        }

    try:
        with open(_PGLITE_STATE_FILE) as f:
            state = json.load(f)
    except Exception as exc:
        return _error_envelope(f"failed to read pglite supervisor state: {exc}")

    port = int(state.get("port") or 5440)
    probe = _probe_pglite(port)
    state["managed"] = True
    state["probe"] = probe
    return state


@app.post("/api/pglite/restart")
def pglite_restart():
    """Request an operator-initiated pglite restart.

    Touches the sentinel the supervisor watches; the supervisor then
    SIGTERMs the current child and respawns immediately (no backoff —
    operator restart counts as healthy intent).  Idempotent — touching
    the sentinel while the supervisor is already restarting is safe
    and short-circuits the next watch-loop iteration.
    """
    import os
    import time

    if not os.path.exists(_PGLITE_STATE_FILE):
        return _error_envelope(
            "pglite is not under supervisor control — restart not available",
            status=409,
        )

    # Touch the sentinel — create-or-update mtime.  The supervisor's
    # watch loop polls mtime every ~2s.
    try:
        with open(_PGLITE_RESTART_SENTINEL, "a"):
            os.utime(_PGLITE_RESTART_SENTINEL, (time.time(), time.time()))
    except Exception as exc:
        return _error_envelope(f"failed to touch restart sentinel: {exc}")

    return {"ok": True, "requested_at": time.time()}


@app.get("/api/agents")
def list_agents():
    try:
        client = _get_client()
        resp = client.stub.ListAgents(
            atelier_pb2.ListAgentsRequest(), timeout=5
        )
    except Exception as exc:
        # Surface as JSON envelope so the Workflows page (which calls
        # .json() on the response) can render the error instead of
        # blowing up with "Unexpected token 'I', 'Internal S'...".
        _reset_client()
        return _error_envelope(f"ListAgents failed: {exc}")
    return {
        "agents": [
            {
                "id": a.id,
                "name": a.name,
                "description": a.description,
                "role": a.role,
                "tool_ids": list(a.tool_ids),
            }
            for a in resp.agents
        ]
    }


@app.get("/api/skills")
def list_skills():
    """Return skill definitions from .claude/commands/ markdown files."""
    commands_dir = _project_root / ".claude" / "commands"
    skills = []
    if commands_dir.is_dir():
        for md_file in sorted(commands_dir.glob("*.md")):
            content = md_file.read_text()
            # First line is the title (# Title)
            lines = content.strip().splitlines()
            title = lines[0].lstrip("# ").strip() if lines else md_file.stem
            # Second non-empty line is the description
            description = ""
            for line in lines[1:]:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    description = stripped
                    break
            skills.append({
                "id": md_file.stem,
                "title": title,
                "description": description,
                "content": content,
            })
    return {"skills": skills}


@app.get("/api/skills/{skill_id}")
def get_skill(skill_id: str):
    """Return a single skill's markdown content."""
    from fastapi.responses import Response

    md_file = _project_root / ".claude" / "commands" / f"{skill_id}.md"
    if not md_file.exists():
        return Response(status_code=404, content="Skill not found")
    content = md_file.read_text()
    lines = content.strip().splitlines()
    title = lines[0].lstrip("# ").strip() if lines else skill_id
    description = ""
    for line in lines[1:]:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            description = stripped
            break
    return {"id": skill_id, "title": title, "description": description, "content": content}


@app.get("/api/data-sources")
def list_data_sources(include_archived: bool = False):
    """Return registered data sources. Excludes archived by default."""
    try:
        client = _get_client()
        resp = client.stub.ListDataSources(
            atelier_pb2.ListDataSourcesRequest(
                include_archived=include_archived,
            ), timeout=5
        )
    except Exception as exc:
        _reset_client()
        return _error_envelope(f"ListDataSources failed: {exc}")
    return {
        "sources": [
            {
                "id": s.id,
                "source_type": s.source_type,
                "source_uri": s.source_uri,
                "display_name": s.display_name,
                "vocabulary_mode": s.vocabulary_mode,
                "vocab_uri": s.vocab_uri,
                "created_at": s.created_at,
                "metadata": s.metadata_json,
                "is_archived": s.is_archived,
            }
            for s in resp.sources
        ]
    }


@app.get("/api/datasets")
def list_datasets(source_id: str | None = None,
                  include_archived: bool = False):
    """Return classification datasets. Excludes archived by default."""
    try:
        client = _get_client()
        req = atelier_pb2.ListDatasetsRequest(
            include_archived=include_archived,
        )
        if source_id:
            req.source_id = source_id
        resp = client.stub.ListDatasets(req, timeout=5)
    except Exception as exc:
        _reset_client()
        return _error_envelope(f"ListDatasets failed: {exc}")
    return {
        "datasets": [
            {
                "id": d.id,
                "name": d.name,
                "description": d.description,
                "parquet_path": d.parquet_path,
                "row_count": d.row_count,
                "source_id": d.source_id,
                "version_number": d.version_number,
                "is_active": d.is_active,
                "summary": d.summary,
                "fsm_run_id": d.fsm_run_id,
                "created_at": d.created_at,
                "is_archived": d.is_archived,
            }
            for d in resp.datasets
        ]
    }


@app.post("/api/datasets/{dataset_id}/activate")
def activate_dataset(dataset_id: str):
    """Set a dataset version as active for its source."""
    try:
        from atelier.db.dao import AtelierDao
        dao = AtelierDao()
        ds = dao.get_dataset(dataset_id)
        if ds is None:
            return _error_envelope("Dataset not found", status=404)
        if not ds.get("source_id"):
            return _error_envelope("Dataset has no source", status=400)
        dao.set_active_version(ds["source_id"], dataset_id)
        return {"ok": True, "dataset_id": dataset_id, "source_id": ds["source_id"]}
    except Exception as exc:
        return _error_envelope(f"activate_dataset failed: {exc}")


# ── ML Artifact Sets ──────────────────────────────────────────────


@app.get("/api/artifact-sets")
def list_artifact_sets(source_id: str | None = None,
                       include_archived: bool = False):
    """Return registered ML artifact sets, newest first.

    Each row indexes the on-disk paths of a CatBoost classifier (always)
    plus optional SVM / UMAP and the training-time metadata an Extend
    Classification run needs (vocab signature, embedding model, etc.).
    """
    try:
        from atelier.db.dao import AtelierDao
        dao = AtelierDao()
        rows = dao.list_artifact_sets(
            source_id=source_id, include_archived=include_archived,
        )
    except Exception as exc:
        return _error_envelope(f"list_artifact_sets failed: {exc}")
    return {"artifact_sets": rows}


@app.get("/api/artifact-sets/{artifact_set_id}")
def get_artifact_set(artifact_set_id: str):
    """Return a single artifact set by id."""
    try:
        from atelier.db.dao import AtelierDao
        row = AtelierDao().get_artifact_set(artifact_set_id)
        if row is None:
            return _error_envelope("Artifact set not found", status=404)
        return row
    except Exception as exc:
        return _error_envelope(f"get_artifact_set failed: {exc}")


@app.post("/api/artifact-sets/{artifact_set_id}/activate")
def activate_artifact_set(artifact_set_id: str):
    """Promote an artifact set to globally active.

    Demotes any currently-active set in the same transaction; the
    Postgres partial unique index on ``(is_active) WHERE is_active``
    enforces the only-one-active invariant.
    """
    try:
        from atelier.db.dao import AtelierDao
        ok = AtelierDao().set_active_artifact_set(artifact_set_id)
        if not ok:
            return _error_envelope("Artifact set not found", status=404)
        return {"ok": True, "artifact_set_id": artifact_set_id, "is_active": True}
    except Exception as exc:
        return _error_envelope(f"activate_artifact_set failed: {exc}")


@app.post("/api/artifact-sets/{artifact_set_id}/archive")
def archive_artifact_set(artifact_set_id: str):
    """Soft-delete an artifact set.  Files on disk are untouched."""
    try:
        from atelier.db.dao import AtelierDao
        ok = AtelierDao().archive_artifact_set(artifact_set_id)
        if not ok:
            return _error_envelope("Artifact set not found", status=404)
        return {"ok": True, "artifact_set_id": artifact_set_id, "is_archived": True}
    except Exception as exc:
        return _error_envelope(f"archive_artifact_set failed: {exc}")


@app.post("/api/artifact-sets/{artifact_set_id}/unarchive")
def unarchive_artifact_set(artifact_set_id: str):
    """Reverse the archive operation.  Does NOT promote to active."""
    try:
        from atelier.db.dao import AtelierDao
        ok = AtelierDao().unarchive_artifact_set(artifact_set_id)
        if not ok:
            return _error_envelope("Artifact set not found", status=404)
        return {"ok": True, "artifact_set_id": artifact_set_id, "is_archived": False}
    except Exception as exc:
        return _error_envelope(f"unarchive_artifact_set failed: {exc}")


@app.get("/api/artifact-sets/{artifact_set_id}/compatibility")
def artifact_set_compatibility(artifact_set_id: str, source_id: str):
    """Pre-check whether an artifact set's vocab is compatible with a source.

    Returns ``{status: ok|superset|partial|disjoint, missing_codes,
    extra_codes}``.  The UI calls this before enabling the Extend
    button so it can warn — never block — the operator.
    """
    try:
        import json
        from atelier.classify.artifact_set import check_compatibility
        from atelier.classify.taxonomy import load_sample_vocabulary
        from atelier.db.dao import AtelierDao

        dao = AtelierDao()
        row = dao.get_artifact_set(artifact_set_id)
        if row is None:
            return _error_envelope("Artifact set not found", status=404)

        artifact_classes = json.loads(row["classes"])

        # Resolve the source's preferred vocab.  For sources without a
        # registered loader (hive — vocab is loaded at run start) we
        # report ok against the artifact's own signature so the UI
        # doesn't surface a spurious warning.
        if source_id == "ootb-sample":
            cs = load_sample_vocabulary(hierarchical=True)
            source_classes = [c.code for c in cs.categories]
        else:
            source_classes = artifact_classes

        report = check_compatibility(artifact_classes, source_classes)
        return {
            "artifact_set_id": artifact_set_id,
            "source_id": source_id,
            "status": report.status,
            "missing_codes": report.missing_codes,
            "extra_codes": report.extra_codes,
            "artifact_signature": report.artifact_signature,
            "candidate_signature": report.candidate_signature,
        }
    except Exception as exc:
        return _error_envelope(f"artifact_set_compatibility failed: {exc}")


@app.get("/api/artifact-sets/{artifact_set_id}/extend-scope")
def artifact_set_extend_scope(artifact_set_id: str, source_id: str):
    """Salient measures for the ML Artifacts panel.

    Given an artifact set + a data source, report:

    - what the artifact set bundles (CatBoost / SVM / UMAP, classes,
      embedding model + dim, vocab signature);
    - the producing dataset's training scope (tables/columns the
      CatBoost was fit on);
    - the source's *current* discovered scope (from its metadata —
      ``table_count`` / ``column_count`` written at seed/discovery time);
    - the delta — ``new_tables`` and ``new_columns`` the operator
      could pick up with an Extend Classification run, without
      re-training.

    Vocab compatibility is included so the panel doesn't need a second
    round-trip to ``/compatibility``.
    """
    try:
        import json as _json
        import re
        from atelier.classify.artifact_set import check_compatibility
        from atelier.classify.taxonomy import load_sample_vocabulary
        from atelier.db.dao import AtelierDao
        from atelier.db.model import Dataset

        dao = AtelierDao()
        artifact = dao.get_artifact_set(artifact_set_id)
        if artifact is None:
            return _error_envelope("Artifact set not found", status=404)

        source = dao.get_data_source(source_id)
        if source is None:
            return _error_envelope("Data source not found", status=404)

        # Source-side counts (from metadata stamped at seed/discovery).
        source_meta: dict = {}
        if source.get("metadata"):
            try:
                source_meta = _json.loads(source["metadata"]) or {}
            except Exception:
                source_meta = {}
        source_table_count = source_meta.get("table_count")
        source_column_count = source_meta.get("column_count")

        # Producing dataset — the classify run whose output IS this
        # artifact set.  Compared against the source's current scope to
        # derive ``new_*``.  Look up by artifact_set_id (lineage column
        # added 20260427) and fall back to fsm_run_id for older rows.
        classified_column_count: int | None = None
        classified_table_count: int | None = None
        producing_dataset_id: str | None = None
        with dao.get_session() as session:
            ds = (
                session.query(Dataset)
                .filter_by(artifact_set_id=artifact_set_id, run_kind="classify")
                .order_by(Dataset.created_at.desc())
                .first()
            )
            if ds is None and artifact.get("fsm_run_id"):
                ds = (
                    session.query(Dataset)
                    .filter_by(fsm_run_id=artifact["fsm_run_id"])
                    .order_by(Dataset.created_at.desc())
                    .first()
                )
            if ds is not None:
                producing_dataset_id = ds.id
                classified_column_count = int(ds.row_count or 0)
                # Summary is a free-form string like "X tables, Y columns".
                # Parse the leading integer out — best-effort.
                if ds.summary:
                    m = re.match(r"\s*(\d+)\s+tables?", ds.summary)
                    if m:
                        classified_table_count = int(m.group(1))

        # Whether the artifact's training source matches the candidate.
        # When False, every entity in the candidate is fair game for
        # Extend (nothing has been classified there yet by this model).
        same_source = artifact.get("source_id") == source_id

        if same_source:
            new_table_count = (
                None
                if source_table_count is None or classified_table_count is None
                else max(0, source_table_count - classified_table_count)
            )
            new_column_count = (
                None
                if source_column_count is None or classified_column_count is None
                else max(0, source_column_count - classified_column_count)
            )
        else:
            new_table_count = source_table_count
            new_column_count = source_column_count

        # Vocab compatibility — same logic as the dedicated endpoint, kept
        # here so the panel only needs one fetch.
        artifact_classes = _json.loads(artifact["classes"])
        if source_id == "ootb-sample":
            cs = load_sample_vocabulary(hierarchical=True)
            source_classes = [c.code for c in cs.categories]
        else:
            source_classes = artifact_classes
        report = check_compatibility(artifact_classes, source_classes)

        return {
            "artifact_set_id": artifact_set_id,
            "source_id": source_id,
            "same_source": same_source,
            "bundle": {
                "catboost": bool(artifact.get("catboost_path")),
                "svm": bool(artifact.get("svm_path")),
                "umap": bool(artifact.get("umap_path")),
            },
            "classes_count": len(artifact_classes),
            "embedding_model": artifact.get("embedding_model"),
            "embedding_dim": artifact.get("embedding_dim"),
            "vocab_signature": artifact.get("vocab_signature"),
            "is_active": bool(artifact.get("is_active")),
            "is_archived": bool(artifact.get("is_archived")),
            "created_at": artifact.get("created_at"),
            "summary": artifact.get("summary"),
            "training_source_id": artifact.get("source_id"),
            "fsm_run_id": artifact.get("fsm_run_id"),
            "producing_dataset_id": producing_dataset_id,
            "source_table_count": source_table_count,
            "source_column_count": source_column_count,
            "classified_table_count": classified_table_count,
            "classified_column_count": classified_column_count,
            "new_table_count": new_table_count,
            "new_column_count": new_column_count,
            "vocab_compatibility": {
                "status": report.status,
                "missing_codes": report.missing_codes,
                "extra_codes": report.extra_codes,
                "artifact_signature": report.artifact_signature,
                "candidate_signature": report.candidate_signature,
            },
        }
    except Exception as exc:
        return _error_envelope(f"artifact_set_extend_scope failed: {exc}")


# ── Archive / unarchive ──────────────────────────────────────────


@app.post("/api/data-sources/{source_id}/archive")
def archive_data_source(source_id: str):
    """Archive a data source and all its datasets."""
    try:
        from atelier.db.dao import AtelierDao
        dao = AtelierDao()
        found = dao.archive_data_source(source_id)
        if not found:
            return _error_envelope("Data source not found", status=404)
        return {"ok": True, "source_id": source_id, "is_archived": True}
    except Exception as exc:
        return _error_envelope(f"archive_data_source failed: {exc}")


@app.post("/api/data-sources/{source_id}/unarchive")
def unarchive_data_source(source_id: str):
    """Unarchive a data source and all its datasets."""
    try:
        from atelier.db.dao import AtelierDao
        dao = AtelierDao()
        found = dao.unarchive_data_source(source_id)
        if not found:
            return _error_envelope("Data source not found", status=404)
        return {"ok": True, "source_id": source_id, "is_archived": False}
    except Exception as exc:
        return _error_envelope(f"unarchive_data_source failed: {exc}")


@app.post("/api/data-sources")
def create_data_source(body: dict):
    """Create a new hive/synth data source with explicit vocab_uri."""
    try:
        from atelier.db.dao import AtelierDao
        dao = AtelierDao()
        source_id = body.get("source_id", "")
        if not source_id:
            return _error_envelope("source_id is required", status=400)
        result = dao.get_or_create_data_source(
            source_id=source_id,
            source_type=body.get("source_type", "hive"),
            display_name=body.get("display_name", source_id),
            source_uri=body.get("source_uri", source_id),
            vocabulary_mode=body.get("vocabulary_mode", "hive"),
            vocab_uri=body.get("vocab_uri", ""),
            metadata=body.get("metadata"),
        )
        return {"ok": True, "source": result}
    except Exception as exc:
        return _error_envelope(f"create_data_source failed: {exc}")


@app.patch("/api/data-sources/{source_id}")
def update_data_source(source_id: str, body: dict):
    """Update mutable fields on a data source (e.g. vocab_uri)."""
    try:
        from atelier.db.dao import AtelierDao
        dao = AtelierDao()
        result = dao.update_data_source(source_id, **body)
        if result is None:
            return _error_envelope("Data source not found", status=404)
        return {"ok": True, "source": result}
    except Exception as exc:
        return _error_envelope(f"update_data_source failed: {exc}")


@app.post("/api/datasets/{dataset_id}/archive")
def archive_dataset(dataset_id: str):
    """Archive a single dataset."""
    try:
        from atelier.db.dao import AtelierDao
        dao = AtelierDao()
        found = dao.archive_dataset(dataset_id)
        if not found:
            return _error_envelope("Dataset not found", status=404)
        return {"ok": True, "dataset_id": dataset_id, "is_archived": True}
    except Exception as exc:
        return _error_envelope(f"archive_dataset failed: {exc}")


@app.post("/api/datasets/{dataset_id}/unarchive")
def unarchive_dataset(dataset_id: str):
    """Unarchive a single dataset."""
    try:
        from atelier.db.dao import AtelierDao
        dao = AtelierDao()
        found = dao.unarchive_dataset(dataset_id)
        if not found:
            return _error_envelope("Dataset not found", status=404)
        return {"ok": True, "dataset_id": dataset_id, "is_archived": False}
    except Exception as exc:
        return _error_envelope(f"unarchive_dataset failed: {exc}")


@app.get("/api/datasets/{dataset_id}/data")
def get_dataset_data(dataset_id: str):
    """Serve a dataset's parquet file for the Embeddings page."""
    from fastapi.responses import Response

    client = _get_client()
    resp = client.stub.ListDatasets(
        atelier_pb2.ListDatasetsRequest(include_archived=True), timeout=5
    )
    dataset = next((d for d in resp.datasets if d.id == dataset_id), None)
    if dataset is None:
        return Response(status_code=404, content="Dataset not found")

    if not dataset.parquet_path:
        return Response(
            status_code=404,
            content="No parquet data yet — run a classification pipeline first",
        )

    parquet_path = Path(dataset.parquet_path)
    if not parquet_path.is_absolute():
        parquet_path = _project_root / parquet_path
    if not parquet_path.exists():
        return Response(status_code=404, content="Parquet file not found")

    return FileResponse(
        str(parquet_path),
        media_type="application/octet-stream",
        filename=f"{dataset_id}.parquet",
    )


# ── Status / health ───────────────────────────────────────────────


@app.get("/api/status")
def status():
    """Aggregated infrastructure health for the operator dashboard.

    Declared as ``def`` (not ``async def``) so FastAPI runs it in a
    threadpool. All three probes below are synchronous blocking calls;
    running them on the event loop would freeze every other request
    when one backend is slow — which on CAI triggers the platform's
    health check to declare the app dead and restart it.
    """
    import time
    import urllib.request

    from atelier.config import load_config

    cfg = load_config()
    checks: dict = {}

    # gRPC server — timeout prevents hang on stale channel
    try:
        t0 = time.monotonic()
        client = _get_client()
        resp = client.stub.HealthCheck(
            atelier_pb2.HealthCheckRequest(), timeout=5
        )
        ms = int((time.monotonic() - t0) * 1000)
        checks["grpc"] = {"ok": True, "version": resp.version, "latency_ms": ms}
    except Exception as e:
        _reset_client()
        checks["grpc"] = {"ok": False, "error": str(e)}

    # PostgreSQL — retry with backoff. PGlite can have transient
    # stalls (pglite-socket@0.0.13 wedge); one timeout shouldn't
    # flip the status card amber. Three attempts with 1s backoff.
    pg_ok = False
    pg_err = None
    for attempt in range(3):
        try:
            t0 = time.monotonic()
            from sqlalchemy import text
            engine = _get_status_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            ms = int((time.monotonic() - t0) * 1000)
            checks["postgres"] = {"ok": True, "latency_ms": ms}
            pg_ok = True
            break
        except Exception as e:
            pg_err = e
            if attempt < 2:
                time.sleep(1)
                _reset_status_engine()

    if not pg_ok:
        _reset_status_engine()
        checks["postgres"] = {"ok": False, "error": str(pg_err)}

    # Qdrant — short timeout
    try:
        t0 = time.monotonic()
        url = f"http://{cfg.qdrant_host}:{cfg.qdrant_http_port}/healthz"
        urllib.request.urlopen(url, timeout=3)
        ms = int((time.monotonic() - t0) * 1000)
        checks["qdrant"] = {"ok": True, "latency_ms": ms}
    except Exception as e:
        checks["qdrant"] = {"ok": False, "error": str(e)}

    # Config state (safe subset — no secrets)
    db_masked = cfg.db_url.split("@")[-1] if "@" in cfg.db_url else cfg.db_url
    model_discovery = None
    try:
        from atelier.agents import check_model_upgrade
        model_discovery = check_model_upgrade(cfg)
    except Exception:
        pass
    checks["config"] = {
        "app_display_name": cfg.app_display_name,
        "has_anthropic": cfg.has_anthropic,
        "has_bedrock": cfg.has_bedrock,
        "has_classify_llm": cfg.has_classify_llm,
        "agent_model": cfg.agent_model,
        "qdrant_host": cfg.qdrant_host,
        "qdrant_http_port": cfg.qdrant_http_port,
        "db_url_masked": db_masked,
        "model_discovery": model_discovery,
        # Pipeline-stage capability flags consumed by the Workflows
        # graph to decide which skill nodes are gated (absent /
        # grayed) vs. attached (idle) vs. live (highlighted).
        # ``has_overwatch`` already factors in the Anthropic-direct
        # gate; the cautious_review and agent flags are read raw and
        # the graph composes them with run state at render time.
        "cautious_review_enabled": getattr(
            cfg, "classify_cautious_review_enabled", False,
        ),
        "overwatch_enabled": getattr(cfg, "has_overwatch", False),
        "precondition_enabled": getattr(
            cfg, "classify_precondition_enabled", True,
        ),
        "classify_agent_enabled": getattr(
            cfg, "classify_agent_enabled", False,
        ),
    }

    # "connected" = gRPC reachable. gRPC is the only strict dealbreaker;
    # the frontend cannot function without it. Postgres and Qdrant can
    # flake transiently (PGlite wedges, Qdrant cold-start), but that
    # shouldn't mark the whole app as disconnected on the Landing page.
    # We separately compute "degraded" so the dashboard can show a more
    # useful state than a binary badge.
    grpc_ok = checks.get("grpc", {}).get("ok", False)
    all_ok = all(
        checks.get(svc, {}).get("ok", False)
        for svc in ("grpc", "postgres", "qdrant")
    )
    connected = grpc_ok
    degraded = grpc_ok and not all_ok

    return {**checks, "connected": connected, "degraded": degraded}


# ── Agent SDK endpoints ────────────────────────────────────────────


@app.post("/api/agents/validate-credentials")
def validate_credentials():
    """Validate all configured LLM provider credentials.

    Synchronous (runs in threadpool) — the validation does blocking
    network calls to Anthropic/Bedrock which must not block the event
    loop.
    """
    try:
        from atelier.agents import validate_credentials as _validate
    except ImportError:
        return {"any_valid": False, "error": "agents extra not installed (pip install atelier[agents])"}

    from atelier.config import load_config
    cfg = load_config()
    try:
        return _validate(cfg)
    except Exception as exc:
        return {"any_valid": False, "providers": {}, "configured": [], "error": str(exc)}


@app.post("/api/agents/smoke-test")
async def agent_smoke_test():
    """Run a minimal Claude Agent SDK query to prove the pipeline works."""
    try:
        from atelier.agents.client import run_smoke_test, run_smoke_test_async, _NeedsAsync
    except ImportError:
        return {"success": False, "error": "agents extra not installed (pip install atelier[agents])"}

    from atelier.config import load_config
    cfg = load_config()
    try:
        try:
            return run_smoke_test(cfg)
        except _NeedsAsync:
            return await run_smoke_test_async(cfg)
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@app.get("/api/agents/model-discovery")
def model_discovery():
    """Check for model upgrades via the Anthropic Models API."""
    try:
        from atelier.agents import check_model_upgrade
    except ImportError:
        return {"upgrade_available": False, "reason": "agents extra not installed"}

    from atelier.config import load_config
    cfg = load_config()
    try:
        return check_model_upgrade(cfg)
    except Exception as exc:
        return {"upgrade_available": False, "reason": "error", "error": str(exc)}


# ── Overwatch ──────────────────────────────────────────────────────


@app.get("/api/overwatch/status")
def overwatch_status():
    """Report overwatch configuration, readiness, and GitHub App health.

    ``model`` reflects the WTA's currently selected model_ref —
    Overwatch follows that selection so reviewers/operators see a
    single provider+model knob rather than an Overwatch-specific one.
    """
    try:
        from atelier.config import load_config
        from atelier.terminal_selection import active_model_ref, get_active
        cfg = load_config()
        active_entry = get_active(cfg)
        result: dict = {
            "enabled": cfg.overwatch_enabled,
            "has_overwatch": cfg.has_overwatch,
            "has_anthropic": cfg.has_anthropic,
            "has_bedrock": cfg.has_bedrock,
            "autonomy": cfg.overwatch_autonomy,
            "model": active_model_ref(cfg),
            "model_source": "wta",
            "wta_entry_id": active_entry.id if active_entry else None,
            "github_app": {"configured": False},
        }
        if cfg.overwatch_github_app_id:
            result["github_app"]["configured"] = True
            result["github_app"]["repo"] = cfg.overwatch_github_repo
            # Probe GitHub App connectivity if credentials available
            if cfg.overwatch_github_private_key_path:
                try:
                    from atelier.overwatch.github_app import GitHubApp
                    app = GitHubApp.from_config(cfg)
                    if app:
                        result["github_app"].update(app.ping())
                except Exception as e:
                    result["github_app"]["error"] = str(e)
        return result
    except Exception as exc:
        return _error_envelope(f"overwatch_status failed: {exc}")


@app.get("/api/overwatch/report/{run_id}")
def overwatch_report(run_id: str):
    """Return the overwatch analysis report for a pipeline run."""
    try:
        from pathlib import Path
        report_path = Path("build/results") / run_id / "overwatch.md"
        if not report_path.exists():
            return _error_envelope(
                f"No overwatch report for run {run_id}", status=404
            )
        return {"ok": True, "run_id": run_id, "report": report_path.read_text()}
    except Exception as exc:
        return _error_envelope(f"overwatch_report failed: {exc}")


# ── CDP Control Plane Discovery ────────────────────────────────────


@app.post("/api/cdp/discover")
def cdp_discover():
    """Discover Atlas/Ranger service URLs from the CDP control plane.

    Uses cdpcurl (Ed25519 signed requests) to query the CDP API for
    datalake endpoints.  Requires CDP API access key configured in
    ``~/.cdp/credentials`` or ``CDP_ACCESS_KEY_ID`` / ``CDP_PRIVATE_KEY``
    environment variables.
    """
    try:
        import requests as _req
        from cdpcurl.requests_auth import auth_v1

        cdp_api = "https://api.us-west-1.cdp.cloudera.com"
        result: dict = {"ok": True, "environments": [], "datalakes": [],
                        "discovered_services": {}}

        # List environments
        try:
            r = _req.post(f"{cdp_api}/api/v1/environments2/listEnvironments",
                         json={}, auth=auth_v1(), timeout=15)
            if r.ok:
                envs = r.json().get("environments", [])
                result["environments"] = [
                    {"name": e.get("environmentName", ""), "crn": e.get("crn", "")}
                    for e in envs
                ]
        except Exception as e:
            result["environment_error"] = str(e)

        # List datalakes and extract service endpoints
        try:
            r = _req.post(f"{cdp_api}/api/v1/datalake/listDatalakes",
                         json={}, auth=auth_v1(), timeout=15)
            if r.ok:
                dls = r.json().get("datalakes", [])
                for dl in dls:
                    dl_info = {
                        "name": dl.get("datalakeName", ""),
                        "status": dl.get("status", ""),
                        "services": {},
                    }
                    endpoints = dl.get("endpoints", {}).get("endpoints", [])
                    for ep in endpoints:
                        svc = ep.get("serviceName", "")
                        if svc in ("ATLAS_SERVER", "RANGER_ADMIN", "HIVESERVER2",
                                   "NIFI", "NIFI_REGISTRY", "SOLR"):
                            dl_info["services"][svc] = ep.get("serviceUrl", "")
                            result["discovered_services"][svc] = ep.get("serviceUrl", "")
                    result["datalakes"].append(dl_info)
        except Exception as e:
            result["datalake_error"] = str(e)

        return result

    except ImportError:
        return _error_envelope(
            "cdpcurl not installed — pip install cdpcurl. "
            "Required for CDP control plane discovery."
        )
    except Exception as exc:
        return _error_envelope(f"CDP discover failed: {exc}")


# ── Governance (Atlas + Ranger) ────────────────────────────────────


@app.get("/api/governance/status")
def governance_status():
    """Probe Atlas and Ranger connectivity. Returns per-service health."""
    try:
        from atelier.config import load_config
        from atelier.governance.client import GovernanceClient
        cfg = load_config()
        result: dict = {
            "atlas": {"configured": cfg.has_atlas, "ok": False},
            "ranger": {"configured": cfg.has_ranger, "ok": False},
            "cluster_name": cfg.governance_cluster_name,
            "auto_sync": cfg.governance_auto_sync,
            "dry_run": cfg.governance_dry_run,
        }
        gc = GovernanceClient.from_atelier_config(cfg)
        if cfg.has_atlas:
            result["atlas"].update(gc.atlas.ping())
        if cfg.has_ranger:
            result["ranger"].update(gc.ranger.ping())
        return result
    except Exception as exc:
        return _error_envelope(f"governance_status failed: {exc}")


@app.post("/api/governance/sync-taxonomy")
def governance_sync_taxonomy(source_id: str | None = None, dry_run: bool | None = None):
    """Push the active vocabulary to Atlas as classification types with hierarchy."""
    try:
        from atelier.config import load_config
        from atelier.governance.client import GovernanceClient
        from atelier.governance.sync import TaxonomyNode, sync_taxonomy_to_atlas
        from atelier.classify.taxonomy import load_sample_vocabulary

        cfg = load_config()
        if not cfg.has_atlas:
            return _error_envelope("Atlas not configured — set ATELIER_ATLAS_URL")

        gc = GovernanceClient.from_atelier_config(cfg)
        use_dry_run = dry_run if dry_run is not None else cfg.governance_dry_run

        # Load vocabulary for the source
        vocab = load_sample_vocabulary(hierarchical=True)
        nodes = []
        for cat in vocab.all_categories:
            nodes.append(TaxonomyNode(
                code=cat.code,
                label=cat.label,
                notation=getattr(cat, "notation", ""),
                parent_code=getattr(cat, "parent_code", "") or "",
            ))

        # Sort parents-first for superType resolution
        code_set = {n.code for n in nodes}
        sorted_nodes = sorted(nodes, key=lambda n: n.code.count("."))

        report = sync_taxonomy_to_atlas(gc.atlas, sorted_nodes, dry_run=use_dry_run)
        return {
            "ok": True,
            "dry_run": report.dry_run,
            "total": report.total,
            "created": len(report.created),
            "skipped": len(report.skipped),
            "failed": len(report.failed),
            "failed_types": report.failed[:10],
        }
    except Exception as exc:
        return _error_envelope(f"sync_taxonomy failed: {exc}")


@app.post("/api/governance/tag-results")
def governance_tag_results(
    source_id: str | None = None,
    run_id: str | None = None,
    dry_run: bool | None = None,
):
    """Apply classification results from a pipeline run to Atlas entities."""
    try:
        import json as _json
        from pathlib import Path
        from atelier.config import load_config
        from atelier.governance.client import GovernanceClient
        from atelier.governance.sync import ColumnClassification, sync_classifications_to_atlas

        cfg = load_config()
        if not cfg.has_atlas:
            return _error_envelope("Atlas not configured — set ATELIER_ATLAS_URL")

        gc = GovernanceClient.from_atelier_config(cfg)
        use_dry_run = dry_run if dry_run is not None else cfg.governance_dry_run

        # Find the latest run results
        results_base = Path("build/results")
        if run_id:
            results_path = results_base / run_id / "classifications.json"
        else:
            # Find most recent
            runs = sorted(results_base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if not runs:
                return _error_envelope("No classification results found in build/results/")
            results_path = runs[0] / "classifications.json"

        if not results_path.exists():
            return _error_envelope(f"Results file not found: {results_path}")

        with open(results_path) as f:
            raw = _json.load(f)

        # Group by table, build ColumnClassification objects
        by_table: dict[str, list] = {}
        for c in raw:
            table = c.get("table_name", "unknown")
            by_table.setdefault(table, []).append(
                ColumnClassification(
                    column_name=c["column_name"],
                    tags=[c.get("predicted_code", "")],
                    confidence=float(c.get("confidence", 0) or 0),
                    reason=c.get("evidence", ""),
                )
            )

        total_results = []
        for table_name, cols in by_table.items():
            table_results = sync_classifications_to_atlas(
                gc.atlas, table_name, cols,
                cluster_name=cfg.governance_cluster_name,
                dry_run=use_dry_run,
            )
            total_results.extend(table_results)

        success = sum(1 for r in total_results if r.status == "success")
        skipped = sum(1 for r in total_results if r.status == "skipped")
        errors = sum(1 for r in total_results if r.status == "error")
        dry = sum(1 for r in total_results if r.status == "dry_run")

        return {
            "ok": True,
            "dry_run": use_dry_run,
            "total": len(total_results),
            "success": success,
            "skipped": skipped,
            "errors": errors,
            "dry_run_count": dry,
            "tables": list(by_table.keys()),
        }
    except Exception as exc:
        return _error_envelope(f"tag_results failed: {exc}")


# ── CAI Data Platform ──────────────────────────────────────────────


@app.get("/api/data-connections")
def list_data_connections():
    """Return the HOCON-configured CAI Data Platform connection names."""
    try:
        from atelier.config import load_config
        from atelier.data.connections import list_connections
        cfg = load_config()
        return {"connections": list_connections(cfg)}
    except Exception as exc:
        return _error_envelope(f"list_data_connections failed: {exc}")


@app.get("/api/data-platforms")
def list_data_platforms():
    """Unified CAI Data Platform list — Hive connections + filesystem mounts.

    The Status page renders both kinds in a single dropdown, discriminated
    by ``kind`` ("hive" or "filesystem").  Hive entries come from the
    HOCON ``cml_data_connection_names`` list; filesystem entries come
    from the ``data_sources`` table where ``source_type="filesystem"``
    (seeded at startup from local mounts — ootb-sample, synthetic,
    meta-tagging, plus any future entries).

    Future schemes (``s3://``, ``jdbc://``) plug in here without a
    client-side change: the dropdown already knows how to branch on
    ``kind``.
    """
    try:
        from atelier.config import load_config
        from atelier.data.connections import list_connections
        from atelier.db.dao import AtelierDao
        from atelier.db.model import DataSource
        import json as _json

        cfg = load_config()
        platforms: list[dict] = []

        # ── Hive connections (HOCON config list) ──
        # These are connection *names* from ATELIER_DATA_CONNECTIONS — they
        # may or may not have a corresponding DB row yet.
        hocon_hive_names: set[str] = set()
        for name in list_connections(cfg):
            hocon_hive_names.add(name)
            platforms.append({
                "id": name,
                "kind": "hive",
                "label": f"Hive: {name}",
                "source_uri": f"hive://{name}",
                "vocab_uri": "",
                "mount": None,
                "table_count": None,
                "column_count": None,
            })

        # ── DB-registered sources (non-archived) ──
        # Includes *both* filesystem mounts and hive sources seeded from
        # ATELIER_CLASSIFY_CONNECTION or auto-discovered at startup.
        # De-duplicate hive entries that already appeared via the HOCON
        # config list above (keyed on source_id).
        # Retry-on-disconnect — the Status page polls this endpoint and
        # a single transient PGlite blip would otherwise surface as a
        # hard 500 in the UI.
        dao = AtelierDao()
        # Materialise to dicts *inside* the session so we never touch
        # detached ORM instances after run_with_retry closes the session.
        rows = dao.run_with_retry(
            lambda session: [
                dao._source_to_dict(r)
                for r in (
                    session.query(DataSource)
                    .filter_by(is_archived=False)
                    .order_by(DataSource.id)
                    .all()
                )
            ]
        )
        for r in rows:
            meta = {}
            if r.get("metadata"):
                try:
                    meta = _json.loads(r["metadata"]) or {}
                except Exception:
                    meta = {}

            if r["source_type"] == "hive":
                # Skip if already surfaced via the HOCON config list.
                # HOCON entries use the bare connection name as id;
                # DB-seeded entries use "connection/database".  Check
                # both the full id and the connection component.
                conn = meta.get("connection", "")
                if r["id"] in hocon_hive_names or conn in hocon_hive_names:
                    continue
                platforms.append({
                    "id": r["id"],
                    "kind": "hive",
                    "label": r["display_name"] or f"Hive: {r['id']}",
                    "source_uri": r["source_uri"] or "",
                    "vocab_uri": r.get("vocab_uri") or "",
                    "mount": None,
                    "table_count": meta.get("table_count"),
                    "column_count": meta.get("column_count"),
                })
            else:
                # Filesystem source
                mount = None
                source_uri = r["source_uri"] or ""
                if source_uri.startswith("file://"):
                    mount = source_uri[len("file://"):]
                platforms.append({
                    "id": r["id"],
                    "kind": "filesystem",
                    "label": f"Filesystem: {r['display_name']}",
                    "source_uri": source_uri,
                    "vocab_uri": r.get("vocab_uri") or "",
                    "mount": mount,
                    "table_count": meta.get("table_count"),
                    "column_count": meta.get("column_count"),
                })

        return {"platforms": platforms}
    except Exception as exc:
        return _error_envelope(f"list_data_platforms failed: {exc}")


@app.get("/api/filesystem-sources/{source_id}/stats")
def get_filesystem_source_stats(source_id: str):
    """Return mount-side stats for a filesystem-backed source.

    Used by the Data Platform card to render the row body for a
    selected Filesystem entry: table_count, column_count, and
    annotation_count (from ``annotations.csv`` if present).
    """
    try:
        import json as _json
        from atelier.db.dao import AtelierDao
        from atelier.db.model import DataSource

        dao = AtelierDao()
        with dao.get_session() as session:
            r = session.query(DataSource).filter_by(
                id=source_id, source_type="filesystem",
            ).first()
            if r is None:
                return _error_envelope(
                    f"Filesystem source {source_id!r} not found",
                )
            meta = {}
            if r.source_metadata:
                try:
                    meta = _json.loads(r.source_metadata) or {}
                except Exception:
                    meta = {}
            mount = None
            if r.source_uri and r.source_uri.startswith("file://"):
                mount = r.source_uri[len("file://"):]

            # Annotation count: count rows in annotations.csv if vocab_uri
            # points at one.  Fall back to None on any error so the UI
            # degrades gracefully (empty cell rather than red).
            annotation_count = None
            if r.vocab_uri and r.vocab_uri.startswith("file://"):
                import csv as _csv
                vocab_path = Path(r.vocab_uri[len("file://"):])
                if vocab_path.is_file():
                    try:
                        with open(vocab_path, newline="") as f:
                            annotation_count = sum(
                                1 for _ in _csv.DictReader(f)
                            )
                    except Exception:
                        annotation_count = None

            return {
                "ok": True,
                "source_id": r.id,
                "display_name": r.display_name,
                "mount": mount,
                "vocab_uri": r.vocab_uri or "",
                "table_count": meta.get("table_count"),
                "column_count": meta.get("column_count"),
                "annotation_count": annotation_count,
            }
    except Exception as exc:
        return _error_envelope(f"filesystem_source_stats failed: {exc}")


@app.post("/api/data-connections/{name}/test")
def test_data_connection(name: str):
    """Run ``show databases`` against the named CAI connection via cml.data_v1."""
    try:
        from atelier.config import load_config
        from atelier.data.connections import test_connection
        return test_connection(load_config(), name)
    except Exception as exc:
        return _error_envelope(f"test_data_connection failed: {exc}")


@app.post("/api/data-connections/{name}/refresh")
def refresh_data_connection(name: str):
    """Probe connection: list databases with table counts and annotations status."""
    try:
        from atelier.config import load_config
        from atelier.data.connections import refresh_connection
        return refresh_connection(load_config(), name)
    except Exception as exc:
        return _error_envelope(f"refresh_data_connection failed: {exc}")


# ── Vocabulary ────────────────────────────────────────────────────


@app.get("/api/vocabulary/stats")
def vocabulary_stats(source_id: str | None = None):
    """Return vocabulary term count, source-aware.

    When source_id is 'ootb-sample', returns the expanded ontology count.
    Otherwise falls back to cache → hive → universal.
    """
    try:
        from atelier.config import load_config
        from atelier.classify.taxonomy import (
            load_annotations_from_json,
            load_annotations_from_hive as _hive_vocab,
            load_sample_vocabulary,
            load_universal_vocabulary,
            compose_vocabularies,
        )

        cfg = load_config()

        # OOTB sample and local Synthetic both use the expanded
        # ontology — their reference codes share the 316-category
        # ICE vocabulary.
        if source_id == "ootb-sample":
            try:
                sample_vocab = load_sample_vocabulary(hierarchical=True)
                return {"terms": len(sample_vocab.categories), "source": source_id}
            except FileNotFoundError:
                pass

        project_root = Path(__file__).resolve().parent.parent.parent
        cache_path = project_root / "build" / "data" / "annotations" / "annotations.json"

        # Universal base (always available)
        universal = load_universal_vocabulary(hierarchical=True)

        # Try cache first (domain extensions — validated, reject empty)
        if cache_path.exists():
            cs = load_annotations_from_json(cache_path, hierarchical=True)
            if len(cs.categories) > 0:
                composed = compose_vocabularies(universal, cs)
                return {"terms": len(composed.categories), "source": "cache"}

        # Try hive domain extensions
        try:
            cs = _hive_vocab(cfg)
            if len(cs.categories) > 0:
                composed = compose_vocabularies(universal, cs)
                return {"terms": len(composed.categories), "source": "hive"}
        except Exception:
            pass

        # Universal only (no domain extensions)
        return {"terms": len(universal.categories), "source": "universal"}
    except Exception as exc:
        return _error_envelope(f"vocabulary_stats failed: {exc}")


# ── Settings (runtime config overlay) ──────────────────────────────
#
# Session-level tuning of the DST classification pipeline.  Changes
# apply to the next pipeline run and reset on gateway restart.  For
# persistent changes, operators edit config/base.conf or env vars.


@app.get("/api/settings")
def get_settings():
    """Return settings metadata + current effective values.

    Effective value = overlay (if set) else HOCON default.  The
    metadata dict feeds the React Settings page (label, range,
    description, captions).
    """
    try:
        from atelier.config import load_config
        from atelier.config_overlay import SETTINGS_METADATA, get_overlay

        cfg = load_config()
        overlay = get_overlay()
        values: dict = {}
        for key, meta in SETTINGS_METADATA.items():
            if key in overlay:
                values[key] = overlay[key]
            else:
                values[key] = getattr(cfg, key, meta.get("default"))
        return {
            "metadata": SETTINGS_METADATA,
            "values": values,
            "overlay_keys": sorted(overlay.keys()),
        }
    except Exception as exc:
        return _error_envelope(f"get_settings failed: {exc}")


@app.patch("/api/settings")
def patch_settings(body: dict):
    """Validate and apply settings updates to the runtime overlay.

    Body is a flat {key: value} map.  Returns the updated values on
    success, or a 400 envelope on validation failure.
    """
    try:
        from atelier.config_overlay import set_overlay
        overlay = set_overlay(body or {})
        return {"ok": True, "overlay": overlay}
    except ValueError as exc:
        return _error_envelope(str(exc), status=400)
    except Exception as exc:
        return _error_envelope(f"patch_settings failed: {exc}")


@app.get("/api/acceleration")
def get_acceleration():
    """Return current GPU detection + resolved acceleration methods.

    Read-only probe — drives the "Acceleration" card in Settings.  Never
    blocks or fails; a CPU-only host simply reports ``available: false``.
    Uses the subprocess-isolated probe: the in-process one would pin a
    CUDA context per device on the gateway for its whole life, which the
    zero-share GPU policy counts as occupancy against co-tenant engines.
    """
    try:
        from atelier.classify.gpu import preflight_gpu_isolated
        from atelier.config import load_config
        cfg = load_config()
        info = preflight_gpu_isolated().to_dict()
        gpu_on = info["available"] and cfg.classify_gpu_enabled != "false"
        info["methods"] = {
            # sage_enabled = explicit flag OR auto-enabled on GPU
            "sage": gpu_on or cfg.classify_sage_enabled,
            "sage_gpu": gpu_on,
            "shap_gpu": gpu_on,
            "catboost_gpu": gpu_on,
            "embedding_sharded": (
                gpu_on and info["device_count"] > 1
                and cfg.classify_gpu_shard_threshold < 1_000_000
            ),
        }
        info["config"] = {
            "gpu_enabled": cfg.classify_gpu_enabled,
            "shard_threshold": cfg.classify_gpu_shard_threshold,
            "sage_chunk": cfg.classify_gpu_sage_chunk,
        }
        return info
    except Exception as exc:
        return _error_envelope(f"acceleration probe failed: {exc}")


@app.post("/api/settings/reset")
def reset_settings():
    """Clear the overlay — revert to HOCON/env defaults."""
    try:
        from atelier.config_overlay import clear_overlay
        clear_overlay()
        return {"ok": True}
    except Exception as exc:
        return _error_envelope(f"reset_settings failed: {exc}")


@app.get("/api/settings/focus")
def get_settings_focus(run_id: str | None = None, source_id: str | None = None):
    """Return the adaptive focus list for the Settings page.

    Resolution order:
      1. Explicit ``run_id`` query param → read that run's focus_settings.json.
      2. ``source_id`` → look up its active dataset's fsm_run_id and use that run.
      3. Neither provided → fall back to the curated starter-focus set.

    Response shape::

        {
          "run_id": "a22f1f10" | None,
          "source": "hybrid" | "rules" | "overwatch" | "starter",
          "focus_keys": [...],
          "deterministic": [...],
          "from_overwatch": [...],
          "historical": { key: value_at_run_time, ... },
          "current": { key: resolved_value_now, ... },
          "computed_at": iso8601 | None,
        }

    Missing snapshot / focus files degrade gracefully — caller always
    gets a usable structure.
    """
    try:
        import json
        from pathlib import Path
        from atelier.classify.pipeline import _PROJECT_ROOT
        from atelier.config import load_config
        from atelier.config_overlay import (
            SETTINGS_METADATA,
            STARTER_FOCUS_KEYS,
            apply_to_config,
        )

        cfg = apply_to_config(load_config())

        # Resolve run_id from source_id when necessary.
        if run_id is None and source_id:
            try:
                from atelier.db.dao import AtelierDao
                active = AtelierDao().get_active_version(source_id)
                if active and active.get("fsm_run_id"):
                    run_id = active["fsm_run_id"]
            except Exception as exc:
                log.debug("focus: source_id lookup failed: %s", exc)

        results_dir: Path | None = None
        if run_id:
            results_dir = _PROJECT_ROOT / "build" / "results" / run_id
            if not results_dir.exists():
                results_dir = None

        focus_keys: list[str] = []
        deterministic: list[str] = []
        from_overwatch: list[str] = []
        source = "starter"
        computed_at: str | None = None
        historical: dict = {}

        if results_dir:
            focus_path = results_dir / "focus_settings.json"
            if focus_path.exists():
                try:
                    payload = json.loads(focus_path.read_text())
                    focus_keys = list(payload.get("focus_keys") or [])
                    deterministic = list(payload.get("deterministic") or [])
                    from_overwatch = list(payload.get("from_overwatch") or [])
                    source = str(payload.get("source") or "starter")
                    computed_at = payload.get("computed_at")
                except Exception as exc:
                    log.warning("focus_settings.json unreadable: %s", exc)

            snap_path = results_dir / "settings_snapshot.json"
            if snap_path.exists():
                try:
                    snap = json.loads(snap_path.read_text())
                    historical = dict(snap.get("resolved_values") or {})
                except Exception as exc:
                    log.warning("settings_snapshot.json unreadable: %s", exc)

        # Starter fallback — never leave the UI empty-handed.
        if not focus_keys:
            focus_keys = list(STARTER_FOCUS_KEYS)
            source = "starter"

        # Always emit current values for the focus keys so the UI can
        # show historical vs current side-by-side.
        current: dict = {}
        for key in focus_keys:
            meta = SETTINGS_METADATA.get(key) or {}
            current[key] = getattr(cfg, key, meta.get("default"))

        return {
            "run_id": run_id,
            "source": source,
            "focus_keys": focus_keys,
            "deterministic": deterministic,
            "from_overwatch": from_overwatch,
            "historical": historical,
            "current": current,
            "computed_at": computed_at,
        }
    except Exception as exc:
        return _error_envelope(f"get_settings_focus failed: {exc}")


# ── Terminal WebSocket ─────────────────────────────────────────────


@app.websocket("/ws/terminal/{session_id}")
async def terminal_ws(websocket: WebSocket, session_id: str):
    """Persistent terminal session backed by the Claude Agent SDK.

    Sessions survive WebSocket disconnects.  On reconnect, the output
    ring buffer is replayed so the user sees everything that happened
    while they were away.  SDK queries continue running in the background
    even when no client is connected.

    The read loop is kept concurrent with any in-flight SDK query by
    routing output through an async ``send`` callback registered on
    the session.
    """
    await websocket.accept()

    from atelier.terminal import get_or_create_session

    session, is_new = get_or_create_session(session_id)

    # Serialize websocket writes across the read loop and the
    # background SDK-query task. ``WebSocket.send_json`` is not
    # reentrant — concurrent sends corrupt the frame stream.
    send_lock = asyncio.Lock()

    async def send(frame: dict) -> None:
        async with send_lock:
            await websocket.send_json(frame)

    if is_new:
        session.set_emit(send)
        for frame in session.welcome():
            await send(frame)
    else:
        # Reconnecting to existing session — replay buffered output.
        replay_frames = session.attach(send)
        for frame in replay_frames:
            await send(frame)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if frame.get("type") == "input":
                await session.handle_input(frame.get("data", ""))
    except WebSocketDisconnect:
        pass
    finally:
        # Detach but do NOT shutdown — session stays alive for reconnection.
        session.detach()


@app.get("/api/terminal/sessions")
def terminal_sessions():
    """List active terminal sessions (debugging / operator use)."""
    from atelier.terminal import list_sessions
    return {"sessions": list_sessions()}


# ── Web Terminal Agent model catalog + selection ──────────────────


@app.get("/api/terminal/models")
def terminal_models():
    """Return the frontier-model catalog with per-row availability,
    rolling-stats summaries, and the currently active selection.

    Populates the Web Terminal Agent card on the Status page.
    """
    try:
        from atelier.config import load_config
        from atelier.terminal_catalog import available_models
        from atelier.terminal_selection import get_active, has_override
        from atelier.terminal_stats import summary

        cfg = load_config()
        catalog = available_models(cfg)
        active = get_active(cfg)

        rows = []
        for entry in catalog:
            stats = summary(entry.provider, entry.id)
            d = entry.to_dict()
            d["stats"] = stats
            rows.append(d)

        return {
            "models": rows,
            "active": active.to_dict() if active else None,
            "override_set": has_override(),
        }
    except Exception as exc:
        return _error_envelope(f"terminal model catalog failed: {exc}")


@app.post("/api/terminal/models/active")
def terminal_models_set_active(payload: dict):
    """Set the active terminal model. Body: ``{"id": "<entry_id>"}``.

    Rejects unknown or unavailable ids (e.g. Bedrock entries without
    AWS credentials, or Anthropic entries without ANTHROPIC_API_KEY).
    """
    try:
        entry_id = (payload or {}).get("id", "")
        if not isinstance(entry_id, str) or not entry_id.strip():
            return _error_envelope("missing 'id' in request body")
        from atelier.config import load_config
        from atelier.terminal_selection import set_active
        cfg = load_config()
        entry = set_active(entry_id.strip(), cfg)
        return {"active": entry.to_dict(), "override_set": True}
    except ValueError as exc:
        return _error_envelope(str(exc))
    except Exception as exc:
        return _error_envelope(f"terminal model set failed: {exc}")


@app.delete("/api/terminal/models/active")
def terminal_models_clear_active():
    """Clear the override — next query falls back to ``cfg.agent_model``."""
    try:
        from atelier.config import load_config
        from atelier.terminal_selection import clear_active, get_active
        clear_active()
        cfg = load_config()
        active = get_active(cfg)
        return {"active": active.to_dict() if active else None, "override_set": False}
    except Exception as exc:
        return _error_envelope(f"terminal model clear failed: {exc}")


# ── Classification FSM ─────────────────────────────────────────────


@app.get("/api/fsm/status")
def fsm_status():
    """Return current classification pipeline FSM state."""
    try:
        from atelier.classify import get_fsm_status
        status = get_fsm_status()
        if status is None:
            return {"state": "IDLE", "progress": {}}
        return status
    except Exception as exc:
        return _error_envelope(f"FSM status failed: {exc}")


@app.post("/api/fsm/start")
def fsm_start(source_id: str | None = None):
    """Start a classification pipeline run.

    The pipeline requires an LLM backend.  When ANTHROPIC_API_KEY is
    available (always true with Claude Code), the pipeline auto-defaults
    to AnthropicStructuredBackend + Haiku 4.5.  An explicit classify LLM
    (ATELIER_LLM_API_KEY / ATELIER_LLM_BASE_URL) overrides the default.

    For dev/CI testing without real API calls, inject ``samples=`` and
    ``llm_backend=`` via the Python API.

    Args:
        source_id: Data source to classify. When "ootb-sample", the
            pipeline auto-loads sample CSVs and the expanded vocabulary.
    """
    import threading
    try:
        from atelier.classify import get_fsm
        from atelier.classify.pipeline import run_classification_pipeline
        from atelier.config import load_config
        from atelier.db.dao import AtelierDao

        cfg = load_config()
        fsm = get_fsm(dao=AtelierDao())

        # Check if already running
        current = fsm.get_status()
        if current and current.state.value not in ("IDLE", "CONVERGED", "ERROR"):
            return {"run_id": current.id, "started": False,
                    "error": f"Already running: {current.state.value}"}

        if not cfg.has_classify_llm:
            return {"started": False,
                    "error": "No classification LLM configured. "
                    "Set ANTHROPIC_SUBAGENT_MODEL or ATELIER_LLM_API_KEY."}

        # Gate: serialize on the task queue before dispatching the pipeline.
        # Tasks enqueued from the Session pod or via the Web Terminal Agent
        # (apply forward transforms, verify, render change-management guide,
        # etc.) must complete before a classification run so the pipeline
        # reads the post-apply Qdrant collection rather than racing it.
        # Refuses the start when any task fails — the operator must resolve
        # via `python -m atelier.task_queue {retry|cancel}` before retrying.
        try:
            from atelier import task_handlers  # noqa: F401 — registers handlers
            from atelier.task_queue import drain_then_check
            is_clean, queue_err, queue_state = drain_then_check()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Task queue gate failed to evaluate (%s) — proceeding with "
                "pipeline start; investigate task_queue logs", exc,
            )
        else:
            if not is_clean:
                return {"started": False,
                        "error": f"Task queue not clean: {queue_err}",
                        "queue_state": queue_state}
            if queue_state.get("drain", {}).get("ran"):
                logger.info(
                    "Pre-pipeline drain ran %d task(s); queue now clean, "
                    "proceeding with pipeline start.",
                    queue_state["drain"]["ran"],
                )

        # Resolve source metadata: connection, database, vocab_uri.
        # Precedence ladder (lowest → highest):
        #   base.conf < .env.cai.enc < ATELIER_CLASSIFY_* operator env
        #     < data_sources DAO row (UI-saved) < fsm_start source_id
        # Start from the env-var defaults so a deploy with CONNECTION +
        # DATABASE set reaches a usable Classification Pipeline without
        # the operator having to click through the Data Platform panel.
        # Any UI-saved source row for `source_id` overrides per-run.
        vocab_uri = None
        connection_name = getattr(cfg, "classify_connection_name", "") or None
        database = getattr(cfg, "classify_database", "") or "default"
        # OOTB sample and local Synthetic skip the DAO lookup — the
        # pipeline handles their auto-resolution internally.
        if source_id and source_id != "ootb-sample":
            try:
                from atelier.db.dao import AtelierDao
                src = AtelierDao().get_data_source(source_id)
            except Exception as exc:
                logger.warning(
                    "DAO lookup for source %r failed: %s — falling back to "
                    "env-var defaults (connection=%r, database=%r)",
                    source_id, exc, connection_name, database,
                )
                src = None
            if src:
                vocab_uri = src.get("vocab_uri") or vocab_uri
                # source_uri format: "{connection}/{database}"
                uri = src.get("source_uri", "")
                if uri.startswith("file://"):
                    # Filesystem mount — the pipeline loads it via
                    # load_filesystem_source; no Hive connection to
                    # derive here.
                    pass
                elif "/" in uri:
                    connection_name, database = uri.split("/", 1)
                elif uri:
                    connection_name = uri
            elif source_id:
                # Operator explicitly selected a source that doesn't
                # exist (or the DAO lookup failed).  Proceeding with
                # env-var defaults would silently classify the wrong
                # database — fail fast instead.
                return {"started": False,
                        "error": f"Data source {source_id!r} not found. "
                        "Select a valid source or remove the source_id parameter."}

        nautilus_watcher = None

        def _background():
            nonlocal nautilus_watcher
            # Pipeline owns run creation via fsm.start_run() — don't
            # create a run here (avoids double-run bug).
            #
            # Outer BaseException catch (not just Exception) so even
            # thread-level escape hatches — KeyboardInterrupt, a signal
            # handler that raises, SystemExit from a misbehaving lib —
            # still end with an FSM.ERROR transition rather than the
            # observed silent-death failure mode where the thread
            # vanishes and the FSM stays pinned to LLM_SWEEP forever.
            result = None
            try:
                result = run_classification_pipeline(
                    cfg, fsm, source_id=source_id, vocab_uri=vocab_uri,
                    connection_name=connection_name, database=database,
                )
            except BaseException as exc:
                logger.exception("Pipeline background thread died: %s", exc)
                try:
                    from atelier.classify.fsm import FSMState
                    current = fsm.get_status()
                    if current and current.state.value not in (
                        "IDLE", "CONVERGED", "ERROR",
                    ):
                        fsm.advance(
                            current.id, FSMState.ERROR,
                            error=(
                                f"thread died: {type(exc).__name__}: {exc}"
                                if not isinstance(exc, SystemExit)
                                else "thread exited (SystemExit)"
                            ),
                        )
                except Exception:
                    logger.debug("FSM error-transition failed", exc_info=True)
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
            finally:
                # Start the nautilus watcher as soon as the pipeline
                # has a run_id attached to the FSM.  We do this here
                # (after pipeline returns) only when a watcher wasn't
                # attached during the run — the inline hook below is
                # the normal path; this is a best-effort cleanup.
                if nautilus_watcher is not None:
                    try:
                        nautilus_watcher.stop()
                    except Exception:
                        logger.debug("nautilus stop failed", exc_info=True)
                    from atelier.overwatch.nautilus import clear_active_watcher
                    run_id = getattr(nautilus_watcher, "run_id", None)
                    if run_id:
                        clear_active_watcher(run_id)
            return result

        t = threading.Thread(target=_background, daemon=True)
        t.start()

        # Persist the user's expressed classification intent so an
        # AMP/app restart's auto-start can honor it without depending
        # on DAO availability or PGlite ephemerality.  See
        # ``_last_user_selected_source_id`` for the read path.
        _persist_active_source_id(source_id)

        # Attach nautilus once the pipeline has claimed its run_id.
        # The pipeline calls fsm.start_run() very early; poll briefly
        # so the watcher covers the LLM sweep (where the interesting
        # failures live) rather than waiting for the run to end.
        try:
            from atelier.overwatch.nautilus import (
                NautilusWatcher, nautilus_config_from_cfg, set_active_watcher,
            )
            ncfg = nautilus_config_from_cfg(cfg)
            if ncfg.enabled:
                def _auto_cancel_on_stall(rec):
                    """Request cooperative cancel when nautilus detects a stall.
                    Each trigger type only fires once per watcher (fired_triggers
                    dedup), so this doesn't storm. ``_dispatch`` hands the
                    decision to ``_request_cancel`` which flips
                    ``state.cancelled`` and lets ``fsm_cancel`` follow up.
                    """
                    return {
                        "decision": "cancelled",
                        "reason": f"auto-cancel on {rec.trigger}: {rec.trigger_detail}",
                    }
                for _ in range(20):  # up to ~1s — pipeline registers quickly
                    curr = fsm.get_status()
                    if curr and curr.id:
                        nautilus_watcher = NautilusWatcher(
                            run_id=curr.id, fsm=fsm, cfg=ncfg,
                            intervene_callback=_auto_cancel_on_stall,
                        )
                        set_active_watcher(curr.id, nautilus_watcher)
                        nautilus_watcher.start()
                        break
                    import time as _time
                    _time.sleep(0.05)
        except Exception:
            logger.exception("nautilus attach failed (non-fatal)")

        return {"started": True, "source_id": source_id}
    except Exception as exc:
        return _error_envelope(f"FSM start failed: {exc}")


@app.post("/api/fsm/start-bootstrap")
def fsm_start_bootstrap():
    """Deprecated — redirects to /api/fsm/start.

    The convergence loop is now built into the single pipeline entry point.
    """
    return fsm_start()


@app.post("/api/fsm/extend")
def fsm_extend(body: dict):
    """Start an Extend Classification run.

    Body: ``{"source_id": "ootb-sample", "artifact_set_id": "abcd1234",
    "parent_dataset_id": "wxyz5678"}`` (parent_dataset_id is optional).

    Mirrors :func:`fsm_start`'s background-thread plumbing so the
    existing /api/fsm/status polling carries the run through to the UI
    without any new client-side wiring.  Rejects 409 when an FSM run
    is already in flight.
    """
    import threading
    try:
        from atelier.classify import get_fsm
        from atelier.classify.extend_pipeline import run_extend_classification
        from atelier.config import load_config

        source_id = body.get("source_id")
        artifact_set_id = body.get("artifact_set_id")
        parent_dataset_id = body.get("parent_dataset_id")

        if not source_id:
            return _error_envelope("source_id is required", status=400)
        if not artifact_set_id:
            return _error_envelope("artifact_set_id is required", status=400)

        # Pre-check the artifact set exists at the gateway layer so a
        # nonexistent id returns 404 synchronously instead of leaving
        # the operator polling the FSM for a run that errored in a
        # background thread.
        from atelier.db.dao import AtelierDao
        dao = AtelierDao()
        if dao.get_artifact_set(artifact_set_id) is None:
            return _error_envelope(
                f"Artifact set {artifact_set_id!r} not found",
                status=404,
            )

        cfg = load_config()
        fsm = get_fsm(dao=dao)

        # Resolve source metadata: connection + database for Hive sources.
        # Mirrors fsm_start's resolution so an Extend run targeting a
        # hive source_id like ``"hive-poc/reference_corpus"`` reaches
        # ``discover_tables`` with both halves populated.  Without this,
        # connection_name/database stay None and Hive receives a
        # malformed ``SHOW TABLES IN None`` query.
        connection_name = getattr(cfg, "classify_connection_name", "") or None
        database = getattr(cfg, "classify_database", "") or "default"
        if source_id not in ("ootb-sample", "meta-tagging"):
            src = dao.get_data_source(source_id)
            if src:
                uri = src.get("source_uri", "")
                if uri.startswith("file://"):
                    # Filesystem mount — the pipeline loads it via
                    # load_filesystem_source; no Hive connection to
                    # derive here.
                    pass
                elif "/" in uri:
                    connection_name, database = uri.split("/", 1)
                elif uri:
                    connection_name = uri
            else:
                return _error_envelope(
                    f"Data source {source_id!r} not found", status=404,
                )

        # Refuse to spawn while another run is in flight — the FSM
        # singleton can only carry one run at a time.  Status codes
        # match the pattern used by the bootstrap classify start above.
        current = fsm.get_status()
        if current and current.state.value not in ("IDLE", "CONVERGED", "ERROR"):
            return _error_envelope(
                f"FSM busy: {current.state.value}", status=409,
            )

        def _background():
            try:
                run_extend_classification(
                    cfg, fsm,
                    source_id=source_id,
                    artifact_set_id=artifact_set_id,
                    parent_dataset_id=parent_dataset_id,
                    connection_name=connection_name,
                    database=database,
                )
            except BaseException as exc:
                _log.exception("Extend pipeline thread died: %s", exc)
                try:
                    from atelier.classify.fsm import FSMState
                    cur = fsm.get_status()
                    if cur and cur.state.value not in ("IDLE", "CONVERGED", "ERROR"):
                        fsm.advance(
                            cur.id, FSMState.ERROR,
                            error=f"thread died: {type(exc).__name__}: {exc}",
                        )
                except Exception:
                    _log.debug("FSM error-transition failed", exc_info=True)
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise

        threading.Thread(target=_background, daemon=True).start()
        return {
            "started": True,
            "run_kind": "extend",
            "source_id": source_id,
            "artifact_set_id": artifact_set_id,
            "parent_dataset_id": parent_dataset_id,
        }
    except Exception as exc:
        return _error_envelope(f"FSM extend failed: {exc}")


@app.get("/api/fsm/runs")
def fsm_runs():
    """List past classification pipeline runs."""
    try:
        from atelier.classify import get_fsm
        fsm = get_fsm()
        runs = fsm.list_runs()
        return {"runs": [r.to_dict() for r in runs]}
    except Exception as exc:
        return _error_envelope(f"FSM runs failed: {exc}")


@app.post("/api/fsm/cancel")
def fsm_cancel(payload: dict | None = None):
    """Request cancellation of the in-flight classification run.

    Two layers of effect, both applied on every call:

    1. **Cooperative cancel** — sets ``state.cancelled = True`` on the
       live BootstrapState via the nautilus registry, so the pipeline
       polls the flag between LLM batches and exits cleanly after the
       current batch finishes.

    2. **FSM force-reset** — transitions the FSM to ``ERROR`` with a
       clear cancellation reason, unconditionally.  This recovers the
       operator when the pipeline thread died silently (native
       segfault, hung TCP connection with no timeout, etc.) and the
       nautilus state was never registered or the cooperative flag
       has no reader.  Without this, a stuck run would require a
       gateway restart to clear.

    Operator-initiated only — the autonomy gating on
    ``atelier.overwatch.kill_run`` protects supervisor-initiated
    cancels, this endpoint is the UI's direct path.
    """
    try:
        from atelier.classify import get_fsm
        from atelier.classify.fsm import FSMState
        from atelier.overwatch.nautilus import get_state, get_active_watcher

        fsm = get_fsm()
        current = fsm.get_status()
        if current is None:
            return _error_envelope("no run registered")
        if current.state.value in ("IDLE", "CONVERGED", "ERROR"):
            return {
                "cancelled": False,
                "run_id": current.id,
                "state": current.state.value,
                "error": f"run is not in-flight (state={current.state.value})",
            }

        reason = "operator cancelled via Status panel"
        if isinstance(payload, dict):
            reason = str(payload.get("reason") or reason)[:500]

        # Layer 1 — cooperative cancel via nautilus (best-effort).
        state = get_state(current.id)
        cooperative = False
        if state is not None:
            state.cancelled = True
            state.cancellation_reason = reason
            cooperative = True

        # Stop any nautilus watcher regardless of whether the state was
        # registered — watchers can outlive their state registration
        # window if the pipeline thread has already died silently.
        watcher = get_active_watcher(current.id)
        watcher_stopped = False
        if watcher is not None:
            try:
                watcher.stop()
                watcher_stopped = True
            except Exception:
                pass

        # Layer 2 — unconditional FSM force-reset.  Even when the
        # cooperative path succeeded we also set ERROR so the UI can
        # redraw the Start button immediately; the next ``fsm_start``
        # will observe a clean terminal state rather than waiting for
        # the in-flight batch to finish + exit the sweep loop.  If the
        # thread is still alive and processing, its own finally block
        # will no-op on the already-ERROR state via the ValueError
        # guard in advance().
        try:
            fsm.advance(
                current.id, FSMState.ERROR,
                error=f"cancelled by operator: {reason}",
            )
        except ValueError:
            pass  # state already terminal

        return {
            "cancelled": True,
            "run_id": current.id,
            "state": "ERROR",
            "reason": reason,
            "cooperative_cancel_registered": cooperative,
            "watcher_stopped": watcher_stopped,
            "note": (
                None if cooperative else
                "Cooperative cancel could not be registered "
                "(no live nautilus state — thread likely died silently "
                "or hadn't registered yet).  FSM was force-reset to "
                "ERROR; if an in-flight LLM call is hung it will be "
                "abandoned when the gateway next restarts."
            ),
        }
    except Exception as exc:
        return _error_envelope(f"FSM cancel failed: {exc}")


# ── Nautilus (mid-run overwatch watcher) ──────────────────────────


@app.get("/api/overwatch/nautilus/{run_id}")
def overwatch_nautilus_status(run_id: str):
    """Return the nautilus watcher status for a specific run.

    Populated while the run is executing; after the run terminates the
    watcher stops but its intervention history remains queryable until
    the gateway restarts.
    """
    try:
        from atelier.overwatch.nautilus import get_active_watcher
        watcher = get_active_watcher(run_id)
        if watcher is None:
            return {"run_id": run_id, "running": False, "interventions": []}
        return watcher.status()
    except Exception as exc:
        return _error_envelope(f"nautilus status failed: {exc}")


@app.get("/api/overwatch/nautilus")
def overwatch_nautilus_list():
    """List all active nautilus watchers (at most one per concurrent run)."""
    try:
        from atelier.overwatch.nautilus import list_active_watchers
        return {"watchers": [w.status() for w in list_active_watchers()]}
    except Exception as exc:
        return _error_envelope(f"nautilus list failed: {exc}")


# ── Orchestration WebSocket ────────────────────────────────────────

_orchestration_clients: set[WebSocket] = set()


async def broadcast_orchestration_event(event: dict):
    """Push an event to all connected orchestration WebSocket clients."""
    for ws in list(_orchestration_clients):
        try:
            await ws.send_json(event)
        except Exception:
            _orchestration_clients.discard(ws)


@app.websocket("/ws/orchestration")
async def orchestration_ws(websocket: WebSocket):
    """Live orchestration events for the XYFlow canvas.

    Streams agent_spawned, agent_reasoning, agent_tool_call,
    agent_completed events from the classification agent loop.
    """
    await websocket.accept()
    _orchestration_clients.add(websocket)
    await websocket.send_json({
        "type": "topology_reset",
        "message": "Orchestration channel connected.",
    })
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _orchestration_clients.discard(websocket)


# ── Serve React build (production) ───────────────────────────────

_ui_dist = Path(__file__).resolve().parent.parent.parent / "ui" / "dist"

# Serve ghostty-web WASM assets (production — in dev, Vite serves from public/)
_ghostty_dir = _project_root / "ui" / "public" / "ghostty"
if _ghostty_dir.exists():
    app.mount("/ghostty", StaticFiles(directory=str(_ghostty_dir)), name="ghostty")

_bundle_path = _project_root / "build" / "atelier-state-bundle.tgz"


@app.get("/api/bundle/download")
def bundle_download():
    """Stream the pre-built state bundle for exfiltration."""
    if not _bundle_path.exists():
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=404,
            content={"detail": "No bundle found. Build it first."},
        )
    return FileResponse(
        str(_bundle_path),
        media_type="application/gzip",
        filename="atelier-state-bundle.tgz",
    )


@app.get("/api/bundle/info")
def bundle_info():
    """Return bundle metadata (size, sha256) without downloading."""
    if not _bundle_path.exists():
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=404, content={"detail": "No bundle found."}
        )
    import hashlib

    size = _bundle_path.stat().st_size
    h = hashlib.sha256()
    with open(_bundle_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"size_bytes": size, "sha256": h.hexdigest(), "filename": _bundle_path.name}


if _ui_dist.exists():
    # Mount static assets (JS/CSS bundles)
    _assets_dir = _ui_dist / "assets"
    if _assets_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(_assets_dir)),
            name="assets",
        )

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the React SPA for all non-API routes."""
        if full_path.startswith("api/"):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        return FileResponse(str(_ui_dist / "index.html"))
