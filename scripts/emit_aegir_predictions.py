#!/usr/bin/env python3
"""Emit Atelier predictions for an Ægir release — the efficacy-gate handoff.

Writes the predictions parquet in the schema settled in the 2026-07-03 sync
advice, scored Ægir-side by ``aegir/scripts/score_atelier_predictions.py``
(``just score-atelier <predictions.parquet>``)::

    table_id, column_id, predicted_code      required
    belief, plausibility                     optional (Brier calibration)

Predictions may sit at any vocabulary depth — the scorer walks
``parent_code`` with ``1/(1+d)`` credit in both directions, so calibrated
coarseness (``cautious_code(tau)``) is rewarded, never cliffed.

Modes:

``--golden``
    predicted_code == the release's own held-back reference at full belief.
    A plumbing check for the schema / (table_id, column_id) join / scorer
    loop — expect ``hierarchical_score == 1.0`` and coverage 1.0.  NOT a
    classification result; the artifact is stamped ``mode=golden`` in the
    log and should never be quoted as accuracy.

``--from-run RUN_ID``
    Map ``build/results/<run_id>/classifications.json`` (rows keyed
    ``table_name``/``column_name``) back to release ids via
    ``corpus_columns.parquet`` — v2 guarantees ``column_name`` unique
    within its table, so the name-keyed join is sound.

The generation-manifest pin guard (``check_release_pin``) runs first and
refuses mismatched or pre-v2 releases; ``--allow-unpinned`` downgrades that
to a loud warning for plumbing work against legacy emissions (v0.3).

Usage::

    uv run python scripts/emit_aegir_predictions.py --golden \\
        --release-dir ~/local/src/zndx/aegir/build/atelier_release_preview \\
        --out build/aegir_predictions_golden.parquet
    (cd ../aegir && just score-atelier \\
        $(pwd)/../atelier/build/aegir_predictions_golden.parquet)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from atelier.classify.aegir_release import (  # noqa: E402
    CORPUS_FILE,
    ReleasePinError,
    check_release_pin,
    load_generation_manifest,
    resolve_reference_path,
)


def _read_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _resolve_dir(value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else REPO / p


def golden_predictions(release_dir: Path) -> list[dict]:
    """predicted_code == reference_code at full belief (plumbing check).

    Set-valued reference codes (``|``-separated, P5+) collapse to their
    first member — the scorer credits any member of the set at 1.0.
    """
    rows = _read_rows(resolve_reference_path(release_dir))
    preds: list[dict] = []
    for r in rows:
        code = str(r.get("reference_code") or "")
        if not code:
            continue
        preds.append(
            {
                "table_id": r["table_id"],
                "column_id": r["column_id"],
                "predicted_code": code.split("|")[0],
                "belief": 1.0,
                "plausibility": 1.0,
            }
        )
    return preds


def agent_mediated_predictions(am_dir: Path) -> tuple[list[dict], int]:
    """The agent-mediated referee's decisions as a predictions parquet.

    Keyed directly by the working set's ``column_id`` (single ingress —
    no release join needed). ``belief``/``plausibility`` both carry the
    referee's stated confidence (a singleton posterior, no interval).
    Unresolved decisions are skipped and counted.
    """
    ws = json.loads((am_dir / "working_set.json").read_text())
    decisions = json.loads((am_dir / "agent_mediated.json").read_text())
    preds: list[dict] = []
    skipped = 0
    for qname, decision in decisions.items():
        code = decision.get("code")
        col = ws["columns"].get(qname)
        if not code or not col or not col.get("column_id"):
            skipped += 1
            continue
        conf = float(decision.get("confidence", 0.0))
        preds.append({
            "table_id": col["table_id"],
            "column_id": col["column_id"],
            "predicted_code": code,
            "belief": conf,
            "plausibility": conf,
        })
    return preds, skipped


def run_predictions(release_dir: Path, run_dir: Path) -> tuple[list[dict], int]:
    """Join a pipeline run's classifications back to release ids.

    ``classifications.json`` rows carry ``table_name`` (== release
    ``table_id`` for aegir_release loads) and ``column_name``;
    ``corpus_columns.parquet`` recovers ``column_id``.  Returns
    ``(predictions, n_unjoined)``.
    """
    results_path = run_dir / "classifications.json"
    if not results_path.exists():
        raise FileNotFoundError(f"no classifications.json in {run_dir}")
    classifications = json.loads(results_path.read_text())

    id_by_name: dict[tuple[str, str], str] = {}
    for r in _read_rows(release_dir / CORPUS_FILE):
        id_by_name[(r["table_id"], r["column_name"])] = r["column_id"]

    preds: list[dict] = []
    unjoined = 0
    for row in classifications:
        key = (str(row.get("table_name") or ""), str(row.get("column_name") or ""))
        column_id = id_by_name.get(key)
        if column_id is None:
            unjoined += 1
            continue
        pred: dict = {
            "table_id": key[0],
            "column_id": column_id,
            "predicted_code": str(row.get("predicted_code") or ""),
        }
        if row.get("belief") is not None:
            pred["belief"] = float(row["belief"])
        if row.get("plausibility") is not None:
            pred["plausibility"] = float(row["plausibility"])
        preds.append(pred)
    return preds, unjoined


def write_predictions(preds: list[dict], out_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not preds:
        raise SystemExit("refusing to write an empty predictions parquet")
    has_belief = any("belief" in p for p in preds)
    cols: dict[str, list] = {
        "table_id": [p["table_id"] for p in preds],
        "column_id": [p["column_id"] for p in preds],
        "predicted_code": [p["predicted_code"] for p in preds],
    }
    if has_belief:
        cols["belief"] = [p.get("belief") for p in preds]
        cols["plausibility"] = [p.get("plausibility") for p in preds]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), out_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--release-dir", default=None,
                    help="Ægir release dir (default: cfg.aegir_release_dir)")
    ap.add_argument("--corpora-dir", default=None,
                    help="pinned sdg-corpora checkout (default: cfg.aegir_corpora_dir)")
    ap.add_argument("--out", default="build/aegir_predictions.parquet")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--golden", action="store_true",
                      help="predictions == reference (plumbing check, expect 1.0)")
    mode.add_argument("--from-run", metavar="RUN_ID",
                      help="map build/results/<RUN_ID>/classifications.json to release ids")
    mode.add_argument("--from-agent-mediated", metavar="TAXONOMY_ID",
                      help="emit the agent-mediated referee's decisions "
                           "(build/data/agent_mediated/<id>/)")
    ap.add_argument("--results-root", default="build/results",
                    help="root holding <RUN_ID>/ dirs (with --from-run)")
    ap.add_argument("--allow-unpinned", action="store_true",
                    help="downgrade a failed generation pin to a warning (plumbing only)")
    a = ap.parse_args(argv)

    # Config supplies defaults per the HOCON directive; CLI overrides.
    from atelier.config import load_config

    cfg = load_config()
    release_dir = _resolve_dir(
        a.release_dir or cfg.aegir_release_dir
        or ap.error("--release-dir not given and classify.aegir.release_dir is empty")
    )
    corpora_dir = _resolve_dir(a.corpora_dir or cfg.aegir_corpora_dir)

    try:
        manifest = check_release_pin(release_dir, corpora_dir)
        print(f"pin OK: ontology_sha={manifest.get('ontology_sha')} "
              f"vocab_generation={manifest.get('vocab_generation')}")
    except ReleasePinError as exc:
        if not a.allow_unpinned:
            raise SystemExit(f"REFUSED: {exc}")
        print(f"WARNING (unpinned run — plumbing only, never score): {exc}",
              file=sys.stderr)
        manifest = load_generation_manifest(release_dir)

    if a.golden:
        preds = golden_predictions(release_dir)
        print(f"golden predictions: {len(preds)} rows "
              f"(predicted_code == reference; NOT a classification result)")
    elif a.from_agent_mediated:
        am_dir = REPO / "build/data/agent_mediated" / a.from_agent_mediated
        preds, skipped = agent_mediated_predictions(am_dir)
        print(f"agent-mediated predictions: {len(preds)} rows from {am_dir} "
              f"({skipped} unresolved skipped)")
    else:
        run_dir = _resolve_dir(a.results_root) / a.from_run
        preds, unjoined = run_predictions(release_dir, run_dir)
        print(f"run predictions: {len(preds)} rows from {run_dir} "
              f"({unjoined} unjoinable rows skipped)")
        if not preds:
            raise SystemExit("no prediction rows joined to the release — "
                             "was the run over this release?")

    out_path = _resolve_dir(a.out)
    write_predictions(preds, out_path)
    print(f"wrote {out_path}")
    print("score with: (cd <aegir> && just score-atelier "
          f"{out_path} release=<release_dir>)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
