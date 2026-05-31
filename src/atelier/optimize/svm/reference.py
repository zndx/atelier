"""Reference + synth corpus loading + encoding cache for the SVM channel.

Public API:
    - ``load_synth_rows(corpus_dir)``  → list[Row]
    - ``corpus_hash(corpus_dir)``       → 12-char SHA1 hex
    - ``reference_hash(weights_path)``  → 12-char SHA1 hex (audit-aware)
    - ``load_reference_rows_with_weights(weights_path, database, refresh_cache)``
    - ``encode_with_cache(texts, cache_key, refresh, batch_size)``
    - ``DEFAULT_CORPUS_DIR``, ``EMBED_CACHE`` (path constants)

All consumed by the training dispatcher (atelier.optimize.svm.train),
the Gate B head trainer (atelier.optimize.svm.gate), the post-
integration diagnostic (svm_target_health), the corpus metrology, and
the refinement loop.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np

from atelier.optimize.svm.reflect import (
    Row,
    build_texts_and_labels,
    load_rows,
)

log = logging.getLogger("atelier.optimize.svm.reference")

# Cache + corpus paths.  Match the historical script's locations so
# existing embedding cache files at build/reflect_nhsvm_eval_shap_v2/
# remain valid post-migration.
REPORT_DIR = Path("build/reflect_nhsvm_eval_shap_v2")
EMBED_CACHE = REPORT_DIR / "embeddings_cache.npz"
DEFAULT_CORPUS_DIR = Path("build/data/svm_training/corpus_v2")


# ──────────────────────────────────────────────────────────────────────
# Load synth corpus
# ──────────────────────────────────────────────────────────────────────

def load_synth_rows(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> list[Row]:
    """Load synth_rows.jsonl from a corpus directory."""
    synth_path = corpus_dir / "synth_rows.jsonl"
    if not synth_path.exists():
        raise FileNotFoundError(
            f"Synth corpus missing at {synth_path}. "
            f"Run scripts/generate_corpus_v2.py first."
        )
    rows: list[Row] = []
    with synth_path.open() as f:
        for line in f:
            d = json.loads(line)
            rows.append(Row(
                table=d["table"],
                column=d["column"],
                column_type=d["column_type"],
                sample_values=d["sample_values"],
                siblings_full=d["siblings_full"],
                mnemonic=d["mnemonic"],
                code=d["code"],
            ))
    log.info("  loaded %d synth rows from %s", len(rows), synth_path)
    return rows


def corpus_hash(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> str:
    """Hash of the synth corpus file for cache invalidation."""
    synth_path = corpus_dir / "synth_rows.jsonl"
    if not synth_path.exists():
        return "none"
    h = hashlib.sha1()
    with synth_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def reference_hash(weights_path: Path) -> str:
    """Hash tied to the audit-derived weights file for cache invalidation
    of reference-primary embeddings.  Uses (mtime, size, summary block) so
    the key changes if the audit re-runs even if weights values land
    identically."""
    if not weights_path.exists():
        return "noaudit"
    st = weights_path.stat()
    h = hashlib.sha1()
    h.update(f"{st.st_mtime_ns}:{st.st_size}".encode())
    try:
        blob = json.loads(weights_path.read_text())
        summary = json.dumps(blob.get("summary", {}), sort_keys=True)
        h.update(summary.encode())
    except Exception:
        pass
    return h.hexdigest()[:12]


# ──────────────────────────────────────────────────────────────────────
# Load reference rows + join with audit-derived training weights
# ──────────────────────────────────────────────────────────────────────

def load_reference_rows_with_weights(
    *,
    weights_path: Path = Path("build/data/agent_mediated/training_weights.json"),
    database: str | None = None,
    refresh_cache: bool = False,
) -> tuple[list[Row], list[str], np.ndarray, dict, dict]:
    """Load the agent-mediated reference and join it with per-row training
    weights from the audit policy file.

    Rows excluded from training (confidence=low/unsure → weight 0) are
    dropped here so the caller never has to worry about them.  Singleton
    classes are also dropped (NHSVM constraint).  Rows present in the
    reference but absent from the audit default to weight=1.0 — keeps the
    function safe to call when audit.json hasn't been re-run after the
    reference grew.

    If ``weights_path`` is absent entirely, falls back to weight=1.0
    everywhere with no exclusions (other than singletons).  Same default
    as if you ran ``scripts/audit_reference_quality.py`` with all weights
    set to 1.0.

    Returns
    -------
    kept_rows : list[Row]
        Rows after exclusions, parallel to kept_labels and kept_weights.
    kept_labels : list[str]
        Code labels per kept row.
    kept_weights : np.ndarray  (float32, shape (N,))
        Per-row training weight, parallel to kept_rows.
    coverage_table : dict[str, dict]
        Per-code coverage stats from the audit (n_total, n_kept,
        effective_weight_sum, needs_synth_augmentation).  Empty if no
        audit file.
    audit_summary : dict
        Policy + summary + audit_source + timestamp metadata block,
        for surfacing in the results JSON.  Empty if no audit file.
    """
    if weights_path.exists():
        weights_blob = json.loads(weights_path.read_text())
        per_row_weights = weights_blob.get("per_row", {})
        coverage_table = weights_blob.get("per_code_coverage", {})
        audit_summary = {
            "policy": weights_blob.get("policy", {}),
            "summary": weights_blob.get("summary", {}),
            "audit_source": weights_blob.get("audit_source"),
            "timestamp": weights_blob.get("timestamp"),
        }
        log.info(
            "  loaded training weights from %s (%s rows, %s codes in audit)",
            weights_path,
            audit_summary["summary"].get("n_rows_total", "?"),
            audit_summary["summary"].get("n_codes", "?"),
        )
    else:
        per_row_weights = {}
        coverage_table = {}
        audit_summary = {}
        log.info(
            "  training_weights.json not found at %s; defaulting weight=1.0 everywhere",
            weights_path,
        )

    # Resolve the reference database from config (fail-closed). No customer
    # database is baked in: callers either pass database= explicitly or set
    # ATELIER_REFERENCE_DATABASE (classify.reference_database). The legacy
    # Hive reference path is deprecated in favor of the public
    # GitTables/SOTAB fixture path, which needs no Hive database at all.
    if database is None:
        from atelier.config import load_config
        database = (load_config().classify_reference_database or "").strip()
    if not database:
        raise ValueError(
            "load_reference_rows_with_weights: no reference database "
            "configured. This legacy Hive reference path is deprecated — "
            "prefer the public GitTables/SOTAB fixture path. To use the Hive "
            "path anyway, set ATELIER_REFERENCE_DATABASE (or pass database=). "
            "No customer database is defaulted."
        )

    log.info("Loading reference rows from %s...", database)
    real_rows = load_rows(refresh_cache=refresh_cache, database=database)
    _, real_labels = build_texts_and_labels(real_rows)

    # First pass: apply audit-derived exclusions + collect per-row weights.
    indexed: list[tuple[Row, str, float]] = []
    n_excluded_weight = 0
    n_missing_audit = 0
    for r, lbl in zip(real_rows, real_labels):
        key = f"{r.table}.{r.column}"
        entry = per_row_weights.get(key)
        if entry is None:
            n_missing_audit += 1
            weight = 1.0  # default for rows not in audit
        else:
            if entry.get("exclude", False) or float(entry.get("weight", 1.0)) <= 0.0:
                n_excluded_weight += 1
                continue
            weight = float(entry["weight"])
        indexed.append((r, lbl, weight))

    # Second pass: drop singleton classes (NHSVM constraint).
    code_counts = Counter(lbl for _, lbl, _ in indexed)
    singletons = {c for c, n in code_counts.items() if n < 2}
    kept_rows: list[Row] = []
    kept_labels: list[str] = []
    kept_weights: list[float] = []
    n_singletons_dropped = 0
    for r, lbl, w in indexed:
        if lbl in singletons:
            n_singletons_dropped += 1
            continue
        kept_rows.append(r)
        kept_labels.append(lbl)
        kept_weights.append(w)

    log.info(
        "  reference: %d rows kept  (%d excluded by audit policy, %d singleton-class rows dropped, %d not in audit→default weight 1.0)",
        len(kept_rows), n_excluded_weight, n_singletons_dropped, n_missing_audit,
    )
    return (
        kept_rows,
        kept_labels,
        np.asarray(kept_weights, dtype=np.float32),
        coverage_table,
        audit_summary,
    )


# ──────────────────────────────────────────────────────────────────────
# Encoding with cache (keyed on text hash)
# ──────────────────────────────────────────────────────────────────────

def encode_with_cache(
    texts: list[str], cache_key: str,
    *, refresh: bool = False, batch_size: int = 64,
) -> np.ndarray:
    """Encode texts with ModernBERT, cache by (text_hash, key)."""
    from atelier.optimize.svm.encoder import encode_modernbert

    text_hash = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    full_key = f"{cache_key}_{text_hash}"

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if EMBED_CACHE.exists() and not refresh:
        cache = np.load(EMBED_CACHE, allow_pickle=False)
        if full_key in cache.files:
            log.info("  cache HIT %s  shape=%s", full_key, cache[full_key].shape)
            return cache[full_key]

    log.info("  cache MISS %s — encoding %d texts...", full_key, len(texts))
    t0 = time.time()
    embeddings = encode_modernbert(texts, batch_size=batch_size, pooling="mean")
    elapsed = time.time() - t0
    log.info("  encoded %d in %.1fs (%.0f rows/s)", len(texts), elapsed,
             len(texts) / max(elapsed, 1e-6))

    existing: dict = {}
    if EMBED_CACHE.exists():
        cache = np.load(EMBED_CACHE, allow_pickle=False)
        existing = {k: cache[k] for k in cache.files}
    existing[full_key] = embeddings
    np.savez_compressed(EMBED_CACHE, **existing)
    return embeddings
