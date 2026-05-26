"""Foundational data-loading helpers for the SVM channel's optimize cycle.

This module contains the shared utilities consumed across `just optimize
svm` stages (Phase D training, Gate B, refinement loop, audit).  The
diagnostic CLI that originally lived at ``scripts/reflect_nhsvm.py``
imports its public API from here too — the script retains its own
diagnostic main() (fit_and_predict, measure, render_report, knob sweep)
because those are tied to the diagnostic, not the library.

Public API:
    - ``Row`` dataclass
    - ``load_rows(refresh_cache, database)``
    - ``build_texts_and_labels(rows)``
    - ``build_category_set()``
    - ``characterize_failures(rows, predicted, labels, category_set)``
    - ``AGENT_MEDIATED`` (default agent-mediated.json path)

All other scripts in the optimize cycle should import from
``atelier.optimize.svm.reflect`` directly.
"""
from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("atelier.optimize.svm.reflect")

# Module-level paths.  REPORT_DIR / HIVE_CACHE preserve the legacy
# behavior expected by `scripts/reflect_nhsvm.py`'s diagnostic; this
# module uses HIVE_CACHE for load_rows() caching.
REPORT_DIR = Path("build/reflect_nhsvm")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
HIVE_CACHE = REPORT_DIR / "hive_cache.json"
AGENT_MEDIATED = Path("build/data/agent_mediated/agent_mediated.json")


# ──────────────────────────────────────────────────────────────────────
# Row + Hive pull
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Row:
    table: str
    column: str
    column_type: str
    sample_values: list[str]
    siblings_full: list[str]   # full ordered column list (for centered window)
    mnemonic: str
    code: str

    @property
    def key(self) -> str:
        return f"{self.table}.{self.column}"


def _pull_hive_table(conn, database: str, table: str) -> dict:
    """Pull (column_name, type, top-5 distinct samples) per column in one table."""
    desc = conn.get_pandas_dataframe(f"describe {database}.{table}")
    cols = []
    types = {}
    for _, row in desc.iterrows():
        c = (row["col_name"] or "").strip()
        if not c or c.startswith("#"):
            continue
        cols.append(c)
        types[c] = (row.get("data_type") or "").strip().lower()
    if not cols:
        return {"columns": [], "types": {}, "samples": {}}

    # Pull 100 rows; sample 5 non-null distinct values per column
    sdf = conn.get_pandas_dataframe(f"select * from {database}.{table} limit 100")
    # Hive returns qualified col names: strip 'table.' prefix
    sdf.columns = [c.split(".", 1)[1] if "." in c else c for c in sdf.columns]
    samples = {}
    for c in cols:
        if c not in sdf.columns:
            samples[c] = []
            continue
        vals = sdf[c].dropna().astype(str).tolist()
        # dedupe-preserve-order, take top 5
        seen, deduped = set(), []
        for v in vals:
            v = v.strip()
            if not v or v in seen:
                continue
            seen.add(v)
            deduped.append(v)
            if len(deduped) >= 5:
                break
        samples[c] = deduped
    return {"columns": cols, "types": types, "samples": samples}


def load_rows(refresh_cache: bool, database: str) -> list[Row]:
    """Load reference rows with fresh Hive samples (cached)."""
    am = json.loads(AGENT_MEDIATED.read_text())
    # Tables to query: those that appear in the reference
    tables_needed = sorted({k.split(".", 1)[0] for k in am})

    if HIVE_CACHE.exists() and not refresh_cache:
        cache = json.loads(HIVE_CACHE.read_text())
        if set(cache.keys()) >= set(tables_needed):
            log.info("Loaded Hive samples from cache: %s", HIVE_CACHE)
            data = cache
        else:
            cache = None
    else:
        cache = None

    if cache is None:
        import cml.data_v1 as cmldata  # type: ignore[import-not-found]
        conn = cmldata.get_connection("hive-poc")
        data = {}
        t0 = time.time()
        for i, t in enumerate(tables_needed, 1):
            log.info("[%d/%d] pulling %s", i, len(tables_needed), t)
            try:
                data[t] = _pull_hive_table(conn, database, t)
            except Exception as exc:
                log.warning("  %s: %s", t, exc)
                data[t] = {"columns": [], "types": {}, "samples": {}}
        log.info("Hive pull complete in %.1fs", time.time() - t0)
        HIVE_CACHE.write_text(json.dumps(data, indent=2))

    rows: list[Row] = []
    missing_in_hive = 0
    for key, entry in am.items():
        if not isinstance(entry, dict):
            continue  # skip non-dual-format stragglers (should be zero post-migration)
        table, column = key.split(".", 1)
        t_data = data.get(table)
        if not t_data or column not in t_data["types"]:
            missing_in_hive += 1
            continue
        rows.append(Row(
            table=table, column=column,
            column_type=t_data["types"].get(column, ""),
            sample_values=t_data["samples"].get(column, []),
            siblings_full=t_data["columns"],
            mnemonic=entry["mnemonic"],
            code=entry["code"],
        ))
    log.info("Loaded %d rows (%d reference entries missing in Hive)",
             len(rows), missing_in_hive)
    return rows


# ──────────────────────────────────────────────────────────────────────
# Feature text — uses build_svm_text shape augmented with centered-window siblings
# ──────────────────────────────────────────────────────────────────────

def build_texts_and_labels(rows: list[Row]) -> tuple[list[str], list[str]]:
    """Build (text, label) per row using build_svm_text — the canonical
    NHSVM input shape — augmented with the centered-window sibling list.
    """
    from atelier.classify.svm_classifier import build_svm_text
    from atelier.classify.features import _closest_siblings

    texts: list[str] = []
    labels: list[str] = []
    for r in rows:
        # NHSVM training text is build_svm_text shape — we extend it
        # with the centered-window siblings so the SVM sees the same
        # local neighborhood the cosine embedding sees.
        siblings = _closest_siblings(r.column, r.siblings_full, k=4)
        sibling_part = ", ".join(s.replace("_", " ") for s in siblings)
        base = build_svm_text(r.column, r.column_type, r.sample_values)
        text = f"{base} | siblings: {sibling_part}" if sibling_part else base
        texts.append(text)
        labels.append(r.code)
    return texts, labels


# ──────────────────────────────────────────────────────────────────────
# Category set — full vocabulary from default.annotations
# ──────────────────────────────────────────────────────────────────────

def build_category_set():
    """Build the full hierarchical CategorySet from default.annotations."""
    import cml.data_v1 as cmldata  # type: ignore[import-not-found]
    from atelier.classify.taxonomy import _build_category_set_from_records
    conn = cmldata.get_connection("hive-poc")
    vdf = conn.get_pandas_dataframe(
        "select id, annotation, definition from default.annotations "
        "where deprecated != 'yes'"
    )
    records = []
    for _, v in vdf.iterrows():
        records.append({
            "id": str(v["id"]),
            "ontology": str(v["id"]),
            "annotation": str(v["annotation"]),
            "label": str(v.get("annotation", "")),
        })
    cs = _build_category_set_from_records(records, hierarchical=True)
    log.info("Built category_set: %d codes (%d categories)",
             len(records),
             len(getattr(cs, "all_categories", cs.categories)))
    return cs


# ──────────────────────────────────────────────────────────────────────
# Forensics — characterize residual failures
# ──────────────────────────────────────────────────────────────────────

def characterize_failures(rows: list[Row], predicted: list[str | None],
                          labels: list[str], category_set) -> dict:
    """For each misclassified row, classify the failure mode."""
    failure_kinds: Counter = Counter()
    fail_examples: list[dict] = []
    for r, pred, true_code in zip(rows, predicted, labels):
        if pred is None:
            failure_kinds["singleton_dropped"] += 1
            fail_examples.append({
                "key": r.key, "true": r.code, "pred": None,
                "kind": "singleton_dropped",
            })
            continue
        if pred == true_code:
            continue
        # Classify the kind of confusion
        true_ancestors = set(category_set.ancestors(true_code)) | {true_code}
        pred_ancestors = set(category_set.ancestors(pred)) | {pred}
        if pred in true_ancestors:
            kind = "predicted_ancestor"
        elif true_code in pred_ancestors:
            kind = "predicted_descendant"
        elif true_ancestors & pred_ancestors:
            kind = "predicted_sibling_subtree"
        else:
            kind = "predicted_cross_subtree"
        failure_kinds[kind] += 1
        fail_examples.append({
            "key": r.key, "true": true_code, "pred": pred, "kind": kind,
            "mnemonic_true": r.mnemonic,
        })
    return {
        "failure_kinds": dict(failure_kinds),
        "examples_top": fail_examples[:30],
        "total_failures": len(fail_examples),
    }
