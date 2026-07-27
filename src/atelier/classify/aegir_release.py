"""Load Ægir DDL-release corpora as Atelier classification input.

Ægir (``~/local/src/zndx/aegir``) generates a deterministic ontology → SQL
DDL spine and emits an *Atelier-facing release*: a blind-classification
dataset where each column is presented by its sample **values** only, with
the true ontology code held back as a separate scoring key.  This is the
"DDL artifacts as classification input" bridge — the maturing DDL-generation
procedures over in Ægir become inputs to classification here.

Release layout (produced by ``aegir/scripts/build_atelier_release.py``)::

    <release_dir>/
      corpus_columns.parquet   table_id, column_id, column_name,
                               [register, name_provenance,]           (v2)
                               n_rows, sample_values[],
                               [fk_to_table_id]             ← RELEASE (blind)
      reference.parquet        table_id, column_id, reference_code,
                               template_id, ...             ← held-back key
      release_stats.json       scale stats + generation_manifest (v2)

**Release v2 (natural register)** ships an *always-named* surface: every
column carries a DBA-register ``column_name`` — realistic, cooperative
naming by an author independent of the vocabulary's (the shared-author
semantic register never ships un-degraded).  Names are legitimate
evidence; the ``name_provenance`` ladder (``engine-derived`` >
``composed`` > ``semantic-passthrough`` > ``degraded-mechanical``) grades
trust and is carried on each :class:`ColumnSample` for slicing results by
rung — never for gating features.  v0.3 releases (no register columns)
load identically with those fields ``None``.

The answer key is a physically separate file, mirroring the project
invariant that reference columns are excluded from inputs.  Scored runs
are additionally guarded by :func:`check_release_pin`, which refuses a
release whose ``generation_manifest`` does not match the pinned
sdg-corpora checkout (``cfg.aegir_corpora_dir``).

The ``reference_code`` is **not** loaded onto the samples by default —
:func:`load_aegir_release_samples` returns truly blind ``TableSample``\\ s.
Use :func:`load_aegir_reference` to get the scoring key for evaluation, and
:func:`attach_reference` (or ``with_reference=True``) only when you want the
validation channel populated.

The release ``source_table`` / ``template_id`` join cleanly to the Ægir
Atlas projection (``footprint.<source_table>@aegir`` is an ``rdbms_table``);
see :mod:`atelier.governance.atlas` for reading that structural surface and
writing classifications back onto the same entities.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from atelier.classify.sampler import ColumnSample, TableSample

logger = logging.getLogger(__name__)

CORPUS_FILE = "corpus_columns.parquet"
REFERENCE_FILE = "reference.parquet"
STATS_FILE = "release_stats.json"

# Embedding text keeps a small head of values (parity with the other loaders);
# the full reservoir rides along in ``all_values`` for the sampler / ML stack.
_EMBED_HEAD = 5


def _infer_type(values: list[str]) -> str:
    """Best-effort SQL-ish type from sample values (parity with real_data_loader)."""
    if not values:
        return "string"
    non_null = [v for v in values if v not in ("", None)]
    if not non_null:
        return "string"
    if all(_is_int(v) for v in non_null):
        return "bigint"
    if all(_is_float(v) for v in non_null):
        return "double"
    return "string"


def _is_int(v: str) -> bool:
    try:
        int(str(v).strip())
        return True
    except (ValueError, TypeError):
        return False


def _is_float(v: str) -> bool:
    try:
        float(str(v).strip())
        return True
    except (ValueError, TypeError):
        return False


def _read_parquet_rows(path: Path) -> list[dict]:
    """Read a parquet file to a list of row dicts (pyarrow is already a dep)."""
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def resolve_reference_path(release_dir: str | Path) -> Path:
    """Locate the held-back key for a release.

    Since 2026-07-03 (accepted ask, replies note §3.1) the reference ships in
    a physically separate ``<release>.key/`` sibling so the blind surface can
    be handed to filesystem-capable agents without the key sitting beside it.
    Legacy releases (v0.3, pre-recut previews) keep it in the release dir.
    """
    release_dir = Path(release_dir)
    key_sibling = release_dir.parent / f"{release_dir.name}.key" / REFERENCE_FILE
    if key_sibling.exists():
        return key_sibling
    legacy = release_dir / REFERENCE_FILE
    if legacy.exists():
        return legacy
    raise FileNotFoundError(
        f"reference key not found: neither {key_sibling} nor {legacy}"
    )


def load_aegir_reference(release_dir: str | Path) -> dict[tuple[str, str], str]:
    """Load the held-back scoring key.

    Returns a map ``(table_id, column_id) -> reference_code``.  This file is
    the answer key — never feed it to a classifier; use it only to score
    predictions after a blind run.
    """
    ref_path = resolve_reference_path(release_dir)

    rows = _read_parquet_rows(ref_path)
    key: dict[tuple[str, str], str] = {}
    for r in rows:
        code = r.get("reference_code")
        if code:
            key[(r["table_id"], r["column_id"])] = code
    logger.info("Loaded %d reference codes from %s", len(key), ref_path)
    return key


def load_aegir_release_samples(
    release_dir: str | Path,
    *,
    sample_size: int = 50,
    with_reference: bool = False,
    max_tables: int | None = None,
) -> list[TableSample]:
    """Load an Ægir DDL release as blind ``TableSample`` objects.

    Each ``table_id`` becomes one :class:`TableSample`; each release column
    becomes a :class:`ColumnSample` carrying its ``sample_values``.  Column
    names are the release's (obfuscated) names — the signal is in the values.

    Args:
        release_dir: Directory holding ``corpus_columns.parquet`` (and, if
            ``with_reference``, ``reference.parquet``).
        sample_size: Cap on values retained per column (full reservoir minus
            the head still rides in ``all_values``).
        with_reference: When True, populate ``ColumnSample.reference_code``
            from the held-back key for the validation channel.  Leave False
            (default) for a truly blind classification input.
        max_tables: Optional cap on number of tables loaded (smoke runs).

    Returns:
        ``list[TableSample]`` ready to inject via
        ``run_classification_pipeline(samples=...)``.
    """
    release_dir = Path(release_dir)
    corpus_path = release_dir / CORPUS_FILE
    if not corpus_path.exists():
        raise FileNotFoundError(f"release corpus not found: {corpus_path}")

    rows = _read_parquet_rows(corpus_path)

    ref_key: dict[tuple[str, str], str] = {}
    if with_reference:
        ref_key = load_aegir_reference(release_dir)

    # Group columns by table_id, preserving first-seen order.
    by_table: dict[str, list[dict]] = {}
    for r in rows:
        by_table.setdefault(r["table_id"], []).append(r)

    table_ids = list(by_table)
    if max_tables is not None:
        table_ids = table_ids[:max_tables]

    samples: list[TableSample] = []
    for table_id in table_ids:
        col_rows = by_table[table_id]
        sibling_names = [c["column_name"] for c in col_rows]
        columns: list[ColumnSample] = []
        for c in col_rows:
            values = [str(v) for v in (c.get("sample_values") or [])][:sample_size]
            columns.append(
                ColumnSample(
                    name=c["column_name"],
                    column_type=_infer_type(values),
                    values=values[:_EMBED_HEAD],
                    all_values=values,
                    total_count=int(c.get("n_rows") or len(values)),
                    null_count=0,
                    table_name=table_id,
                    database="aegir_release",
                    siblings=sibling_names,
                    reference_code=ref_key.get((table_id, c["column_id"])),
                    # v2 provenance — None on v0.3 releases.  column_id is
                    # the predictions join key; register/name_provenance
                    # slice results by rung (never gate features).
                    column_id=c.get("column_id"),
                    register=c.get("register"),
                    name_provenance=c.get("name_provenance"),
                    fk_to_table_id=c.get("fk_to_table_id"),
                )
            )
        samples.append(
            TableSample(name=table_id, database="aegir_release", columns=columns)
        )

    total_cols = sum(len(t.columns) for t in samples)
    logger.info(
        "Loaded %d columns across %d tables from %s%s",
        total_cols,
        len(samples),
        corpus_path,
        " (with reference)" if with_reference else " (blind)",
    )
    return samples


def load_aegir_reference_by_name(
    release_dir: str | Path,
) -> dict[tuple[str, str], str]:
    """Scoring key joined to the names that survive into ``TableSample``.

    Blind predictions come back keyed by ``(table_name, column_name)`` (the
    sample's :attr:`ColumnSample.qualified_name` parts), but the held-back
    key is keyed by ``column_id``.  This joins ``corpus_columns`` (which
    carries both) to ``reference`` on ``column_id`` and returns
    ``(table_id, column_name) -> reference_code`` — directly usable to score
    a blind run without populating the samples themselves.
    """
    release_dir = Path(release_dir)
    code_by_id = load_aegir_reference(release_dir)
    corpus_rows = _read_parquet_rows(release_dir / CORPUS_FILE)

    by_name: dict[tuple[str, str], str] = {}
    for r in corpus_rows:
        code = code_by_id.get((r["table_id"], r["column_id"]))
        if code:
            by_name[(r["table_id"], r["column_name"])] = code
    return by_name


def load_release_stats(release_dir: str | Path) -> dict:
    """Load ``release_stats.json`` (dataset-card scale stats), or ``{}``."""
    stats_path = Path(release_dir) / STATS_FILE
    if not stats_path.exists():
        return {}
    return json.loads(stats_path.read_text())


# ── SDG vocabulary (sdg-corpora submodule) ───────────────────────────

# Layout inside the pinned sdg-corpora checkout (cfg.aegir_corpora_dir).
ANNOTATIONS_PARQUET = Path("vocabulary") / "annotations.parquet"


def load_sdg_vocabulary(corpora_dir: str | Path, *, hierarchical: bool = True):
    """Load sdg-corpora's ``vocabulary/annotations.parquet`` as the vocabulary.

    This is the classification target set for the Ægir efficacy gate: the
    released, versioned SKOS annotation table (944 codes at corpora
    ``a6ab350``; content-derived and *moving* — never hardcode the count).
    The schema is the Atelier ReferenceCategory shape (``code, label,
    abbrev, notation, parent_code, taxonomy, description, common_names,
    example_values``); ``example_values`` (pipe-separated column-name hints
    on the ``domain_hypernym`` rows) feeds ``embedding_text`` via the
    builder's ``specifics`` channel.  Hierarchy comes from ``parent_code``
    (``HierarchicalCategorySet``), which is also what hierarchical scoring
    walks on the Ægir side.

    Local TTL snapshots (``atelier-vocab.ttl``) are pre-migration artifacts
    and are NOT authoritative for this path.
    """
    # Shared builder — the same normalization every other vocabulary
    # source (hive, JSON cache, fixtures) goes through, so SDG codes get
    # identical embedding_text / tree semantics.
    from atelier.classify.taxonomy import _build_category_set_from_records

    path = Path(corpora_dir) / ANNOTATIONS_PARQUET
    if not path.exists():
        raise FileNotFoundError(
            f"SDG vocabulary not found: {path} — is the sdg-corpora "
            f"submodule initialized? (git submodule update --init "
            f"external/sdg-corpora)"
        )

    rows = _read_parquet_rows(path)
    records: list[dict] = []
    for r in rows:
        rec = dict(r)
        example_values = rec.pop("example_values", None)
        if example_values:
            rec["specifics"] = str(example_values)
        records.append(rec)

    category_set = _build_category_set_from_records(
        records, hierarchical=hierarchical
    )
    logger.info(
        "Loaded %d SDG vocabulary codes from %s", len(category_set.categories), path
    )
    return category_set


# ── Generation-manifest pin guard ────────────────────────────────────


class ReleasePinError(RuntimeError):
    """A run was attempted against a release that fails the generation pin.

    Raised by :func:`check_release_pin` — fail-fast by design (Defaults
    Philosophy): a scored efficacy run against a mismatched or unpinned
    release silently measures a corpus that no longer exists.
    """


def load_generation_manifest(release_dir: str | Path) -> dict:
    """The v2 ``generation_manifest`` from ``release_stats.json``, or ``{}``.

    Fields: ``ontology_sha`` (sdg-corpora commit the vocabulary/ontology
    were cut from), ``vocab_generation`` (annotation-row count at emission),
    ``ddl_run_id``, ``corpus_run_id`` (may be null pre-P5), ``naming``.
    """
    return load_release_stats(release_dir).get("generation_manifest") or {}


def _resolve_corpora_sha(corpora_dir: Path) -> str | None:
    """Short HEAD sha of the pinned sdg-corpora checkout, or None."""
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "-C", str(corpora_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _sha_matches(a: str, b: str) -> bool:
    """Prefix-tolerant sha comparison (``--short`` length may drift)."""
    a, b = a.strip().lower(), b.strip().lower()
    if not a or not b:
        return False
    if min(len(a), len(b)) < 7:
        return a == b
    return a.startswith(b) or b.startswith(a)


def check_release_pin(
    release_dir: str | Path,
    corpora_dir: str | Path,
    *,
    expected_ontology_sha: str | None = None,
    scored: bool = False,
) -> dict:
    """Refuse a run when the release's generation doesn't match the pin.

    The fail-fast guard from the 2026-07-03 sync contract: a release is
    only classifiable when its ``generation_manifest`` agrees with the
    pinned sdg-corpora checkout — otherwise the vocabulary being predicted
    into and the corpus being classified are from different generations
    (that mismatch is exactly what the corpora-HEAD version skew looks
    like, measured).

    Checks:

    1. The manifest exists (pre-v2 emissions have none → refuse).
    2. ``ontology_sha`` matches the pinned corpora HEAD (or the explicit
       ``expected_ontology_sha`` override when git isn't available).
    3. ``vocab_generation`` equals the pinned ``annotations.parquet`` row
       count — the vocabulary the run will actually load.

    4. ``holdout_partition`` is present (all emissions since 2026-07-03
       carry it — the RWKV-era eval-design field).  With ``scored=True``
       (efficacy-gate runs) a ``preview*`` partition is refused: scoring a
       release whose corpus overlaps the small model's future training mix
       measures memorization, not capability.

    ``corpus_run_id`` may be null pre-P5 (tolerated); ``ddl_run_id`` is
    recorded in the log for run provenance, with nothing local to compare
    against yet.

    Returns the manifest on success; raises :class:`ReleasePinError`.
    """
    release_dir = Path(release_dir)
    corpora_dir = Path(corpora_dir)

    manifest = load_generation_manifest(release_dir)
    if not manifest:
        raise ReleasePinError(
            f"release {release_dir} carries no generation_manifest in "
            f"{STATS_FILE} — a pre-v2 emission.  Scored runs require a v2+ "
            f"release (re-emit via aegir/scripts/build_atelier_release.py)."
        )

    expected = expected_ontology_sha or _resolve_corpora_sha(corpora_dir)
    if not expected:
        raise ReleasePinError(
            f"cannot resolve the pinned sdg-corpora generation at "
            f"{corpora_dir} (not a git checkout?) and no "
            f"expected_ontology_sha was supplied — refusing to run "
            f"unpinned."
        )
    actual = str(manifest.get("ontology_sha") or "")
    if not _sha_matches(actual, expected):
        raise ReleasePinError(
            f"generation pin mismatch: release {release_dir} was cut from "
            f"sdg-corpora {actual or '<missing>'} but the pinned checkout "
            f"at {corpora_dir} is {expected}.  Re-pin the submodule or "
            f"point at a matching release."
        )

    annotations_path = corpora_dir / ANNOTATIONS_PARQUET
    if not annotations_path.exists():
        raise ReleasePinError(
            f"pinned vocabulary not found: {annotations_path} — is the "
            f"sdg-corpora submodule initialized?"
        )
    import pyarrow.parquet as pq

    vocab_rows = pq.ParquetFile(annotations_path).metadata.num_rows
    declared = manifest.get("vocab_generation")
    if declared is not None and int(declared) != int(vocab_rows):
        raise ReleasePinError(
            f"vocabulary generation mismatch: release declares "
            f"vocab_generation={declared} but the pinned "
            f"annotations.parquet holds {vocab_rows} codes."
        )

    holdout = manifest.get("holdout_partition")
    if not holdout:
        raise ReleasePinError(
            f"release {release_dir} declares no holdout_partition — "
            f"emissions since 2026-07-03 carry it; re-emit the release."
        )
    if scored and str(holdout).startswith("preview"):
        raise ReleasePinError(
            f"scored run refused: holdout_partition={holdout!r} is a preview "
            f"cut (no train/eval separation). Efficacy-gate runs pin at a "
            f"real holdout partition (P4+)."
        )

    logger.info(
        "Release pin OK: ontology_sha=%s vocab_generation=%s ddl_run_id=%s "
        "corpus_run_id=%s naming=%s",
        actual,
        declared,
        manifest.get("ddl_run_id"),
        manifest.get("corpus_run_id"),
        manifest.get("naming"),
    )
    return manifest
