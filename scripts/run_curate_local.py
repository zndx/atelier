#!/usr/bin/env python3
"""Agent-mediated blind curation on the local ``referee`` capability.

The `just agent` phase for the SDG corpus (Arm T stage 2): walks the working
set assembled by ``build_agent_mediated_sdg.py`` and produces the
agent-mediated reference that seeds `just optimize maxsim` / `svm`.

Division of labor (membrane-oracle, per the curate skill's principles):
the HARNESS owns the working set, the vocabulary, all deterministic checks,
and the audit trail; the model only proposes. Two stages per table:

  A. shortlist — one schema-enforced call over the whole table against the
     compact vocabulary index (code + label), yielding candidate codes per
     column;
  B. decide — one schema-enforced call per column with the candidates'
     FULL metadata (definitions, common names, parents) + sibling context.
     Harness validates the code exists (one retry with the error fed back);
     hierarchical integrity is the model's to use (parent codes are
     first-class targets — mass belongs at the deepest SUPPORTED level).

Resume-safe: decisions land in ``agent_mediated.json`` per table;
``review_state.json`` tracks completed tables. Audit trail in ``audit.json``
records candidates, rationale, retained reasoning traces, latencies, model.

The referee capability is served by the Atelier engine (:50251) — start it
first (`just engine-serve`); this script fails fast when unreachable.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

# All strings are grammar-BOUNDED (maxLength): constrained decoding can
# pathologically flood whitespace/padding inside unbounded strings until the
# token budget truncates the JSON mid-structure (observed live on the first
# smoke: 2,589-line 6KB "rationale"). A bounded grammar cannot flood.
SHORTLIST_SCHEMA = {
    "type": "object",
    "properties": {
        "columns": {
            "type": "array",
            "maxItems": 60,
            "items": {
                "type": "object",
                "properties": {
                    "column_name": {"type": "string", "maxLength": 80},
                    "candidates": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 60},
                        "minItems": 1, "maxItems": 6,
                    },
                },
                "required": ["column_name", "candidates"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["columns"],
    "additionalProperties": False,
}

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "column_name": {"type": "string", "maxLength": 80},
        "code": {"type": "string", "maxLength": 60},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string", "maxLength": 400},
    },
    "required": ["column_name", "code", "confidence", "rationale"],
    "additionalProperties": False,
}


class CurationResponseError(RuntimeError):
    """The referee's response was unusable (truncated / unparseable)."""

_SYSTEM = (
    "You are the agent-mediated referee for a blind column-classification "
    "corpus. You see column names, sample values, and sibling columns — "
    "never any answer key. Classify into the SDG vocabulary. Hierarchical "
    "integrity: every code, parent or leaf, is a first-class target; when "
    "the values genuinely span several children, the parent IS the correct "
    "answer — never force a leaf. Confidence is your honest posterior."
)


def _vocab_index(vocabulary: dict) -> str:
    lines = []
    for code, v in sorted(vocabulary.items()):
        hint = ""
        if v.get("hints"):
            hint = f"  [cols like: {', '.join(v['hints'][:6])}]"
        lines.append(f"{code}: {v['label']}{hint}")
    return "\n".join(lines)


def _column_block(qname: str, col: dict) -> str:
    vals = " | ".join(str(v) for v in col["values"][:10])
    return (f"- {col['column_name']} (type {col['column_type']}, "
            f"n_rows {col['n_rows']}): {vals}")


def _shortlist_prompt(table_id: str, cols: list[tuple[str, dict]], index: str) -> str:
    blocks = "\n".join(_column_block(q, c) for q, c in cols)
    return (f"Table {table_id} columns:\n{blocks}\n\n"
            f"Vocabulary (code: label):\n{index}\n\n"
            f"For EACH column, list 1-6 candidate codes from the vocabulary "
            f"(most plausible first). Include a parent code when the values "
            f"might span its children. JSON only.")


def _decision_prompt(qname: str, col: dict, candidates: list[str],
                     vocabulary: dict) -> str:
    cand_lines = []
    for code in candidates:
        v = vocabulary.get(code)
        if not v:
            continue
        cand_lines.append(
            f"{code}: {v['label']} (parent {v['parent_code']}) — "
            f"{v['description']}"
            + (f" [common: {v['common_names']}]" if v["common_names"] else "")
        )
    sibs = ", ".join(col["siblings"])
    vals = " | ".join(str(v) for v in col["values"])
    return (f"Column: {col['column_name']} (type {col['column_type']}, "
            f"n_rows {col['n_rows']})\nValues: {vals}\n"
            f"Sibling columns in the same table: {sibs}\n\n"
            f"Candidate codes (you may also choose any ANCESTOR of a "
            f"candidate if the values span its children):\n"
            + "\n".join(cand_lines)
            + "\n\nDecide the single best code. JSON only.")


class Curator:
    def __init__(self, ws: dict, out_dir: Path, *, workers: int = 3,
                 complete_fn=None) -> None:
        import re
        self.vocabulary = ws["vocabulary"]
        self.columns = ws["columns"]
        self.out_dir = out_dir
        self.workers = workers
        self.index = _vocab_index(self.vocabulary)
        # Deterministic cross-check (curate-skill principle 3, harness >
        # model): identifier-shaped names (_id/_ref/_key/…) always get the
        # vocabulary's identifier-family codes appended to their candidates —
        # the LLM shortlist reliably under-surfaces them.
        self._id_name_re = re.compile(
            r"(?:^|_)(?:id|ref|key|uuid|guid|code)s?$", re.IGNORECASE)
        self._identifier_codes = [
            code for code, v in sorted(self.vocabulary.items())
            if any(w in v["label"].lower()
                   for w in ("identifier", "reference", "key", "designat"))
        ][:6]
        if complete_fn is None:
            from atelier.engine.client import complete_detailed
            complete_fn = complete_detailed
        self._complete = complete_fn
        self._io_lock = threading.Lock()

        self.decisions = self._load(out_dir / "agent_mediated.json")
        self.audit = self._load(out_dir / "audit.json")
        self.state = self._load(out_dir / "review_state.json")
        self.state.setdefault("done_tables", [])

    @staticmethod
    def _load(path: Path) -> dict:
        return json.loads(path.read_text()) if path.exists() else {}

    def _persist(self) -> None:
        with self._io_lock:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "agent_mediated.json").write_text(
                json.dumps(self.decisions, indent=1) + "\n")
            (self.out_dir / "audit.json").write_text(
                json.dumps(self.audit, indent=1) + "\n")
            (self.out_dir / "review_state.json").write_text(
                json.dumps(self.state, indent=1) + "\n")

    def _ask(self, prompt: str, schema: dict, max_tokens: int = 4096) -> dict:
        r = self._complete(prompt, capability="referee", system_prompt=_SYSTEM,
                           max_tokens=max_tokens, temperature=0.0,
                           json_schema=json.dumps(schema))
        if r.get("finish_reason") == "length":
            raise CurationResponseError(
                f"truncated at max_tokens={max_tokens} "
                f"({r.get('completion_tokens')} completion tokens)")
        try:
            return {"parsed": json.loads(r["text"]), "raw": r}
        except json.JSONDecodeError as exc:
            raise CurationResponseError(
                f"unparseable JSON ({len(r['text'])} chars): {exc}") from exc

    # ── Stage A ──────────────────────────────────────────────────────

    def shortlist(self, table_id: str, cols: list[tuple[str, dict]]) -> dict[str, list[str]]:
        out = self._ask(_shortlist_prompt(table_id, cols, self.index),
                        SHORTLIST_SCHEMA, max_tokens=8192)
        by_name: dict[str, list[str]] = {}
        for entry in out["parsed"]["columns"]:
            cands = [c for c in entry["candidates"] if c in self.vocabulary]
            if cands:
                by_name[entry["column_name"]] = cands
        return by_name

    # ── Stage B ──────────────────────────────────────────────────────

    def decide(self, qname: str, col: dict, candidates: list[str]) -> None:
        t0 = time.time()
        try:
            attempt = self._ask(
                _decision_prompt(qname, col, candidates, self.vocabulary),
                DECISION_SCHEMA)
        except (CurationResponseError, Exception) as exc:  # noqa: BLE001
            # One clean retry for transient response pathologies, then
            # record unresolved — a stuck column must not kill the table.
            try:
                attempt = self._ask(
                    _decision_prompt(qname, col, candidates, self.vocabulary),
                    DECISION_SCHEMA)
            except Exception as exc2:  # noqa: BLE001
                with self._io_lock:
                    self.decisions[qname] = {"code": None, "confidence": 0.0,
                                             "unresolved": True}
                    self.audit[qname] = {
                        "candidates": candidates, "code": None,
                        "error": f"{exc} / retry: {exc2}"[:300],
                        "elapsed_s": round(time.time() - t0, 2),
                        "register": col.get("register"),
                        "name_provenance": col.get("name_provenance"),
                    }
                return
        code = attempt["parsed"]["code"]
        retried = False
        if code not in self.vocabulary:
            retried = True
            retry_prompt = (
                _decision_prompt(qname, col, candidates, self.vocabulary)
                + f"\n\nYour previous answer {code!r} is NOT in the "
                f"vocabulary. Choose an exact code from the candidates or "
                f"their ancestors."
            )
            try:
                attempt = self._ask(retry_prompt, DECISION_SCHEMA)
                code = attempt["parsed"]["code"]
            except Exception:  # noqa: BLE001 — keep the first (invalid) answer path
                pass

        parsed = attempt["parsed"]
        if code in self.vocabulary:
            decision = {
                "code": code,
                "confidence": float(parsed.get("confidence", 0.0)),
            }
        else:
            decision = {"code": None, "confidence": 0.0, "unresolved": True}
        audit_row = {
            "candidates": candidates,
            "code": code if code in self.vocabulary else None,
            "raw_code": parsed["code"],
            "confidence": parsed.get("confidence"),
            "rationale": parsed.get("rationale", ""),
            "reasoning_head": (attempt["raw"].get("reasoning_content") or "")[:400],
            "model": attempt["raw"].get("model", ""),
            "retried": retried,
            "elapsed_s": round(time.time() - t0, 2),
            "register": col.get("register"),
            "name_provenance": col.get("name_provenance"),
        }
        # Mutations share _io_lock with _persist: table-parallel workers
        # must not resize these dicts mid-serialization (observed live:
        # "dictionary changed size during iteration" at table 1072/1074).
        with self._io_lock:
            self.decisions[qname] = decision
            self.audit[qname] = audit_row

    # ── Loop ─────────────────────────────────────────────────────────

    def _curate_table(self, table_id: str, cols: list[tuple[str, dict]],
                      tag: str) -> None:
        try:
            by_name = self.shortlist(table_id, cols)
        except Exception as exc:  # noqa: BLE001 — skip table, keep going
            print(f"{tag} {table_id}: shortlist FAILED: {exc}", file=sys.stderr)
            return
        fallback = ["SDG.ICE", "SDG.GENERIC", "SDG.PROCESS",
                    "SDG.INDEPENDENT_CONTINUANT"]
        for qname, col in cols:
            candidates = by_name.get(
                col["column_name"],
                [c for c in fallback if c in self.vocabulary])
            if self._id_name_re.search(col["column_name"]):
                candidates = candidates + [c for c in self._identifier_codes
                                           if c not in candidates]
            self.decide(qname, col, candidates)
        with self._io_lock:
            self.state["done_tables"].append(table_id)
        self._persist()
        print(f"{tag} {table_id}: {len(cols)} cols done", flush=True)

    def run(self, *, max_tables: int | None = None) -> dict:
        by_table: dict[str, list[tuple[str, dict]]] = {}
        for qname, col in self.columns.items():
            by_table.setdefault(col["table_id"], []).append((qname, col))
        todo = [t for t in by_table if t not in self.state["done_tables"]]
        if max_tables is not None:
            todo = todo[:max_tables]

        # Table-level parallelism: tables are tiny (avg ~2 cols in the
        # preview) so the serial shortlist dominates — run whole tables
        # concurrently up to the referee's seq slots (max_num_seqs=4).
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [
                pool.submit(self._curate_table, table_id, by_table[table_id],
                            f"[{i}/{len(todo)}]")
                for i, table_id in enumerate(todo, 1)
            ]
            for f in futures:
                f.result()

        n_resolved = sum(1 for d in self.decisions.values() if d.get("code"))
        summary = {"tables_done": len(self.state["done_tables"]),
                   "decisions": len(self.decisions),
                   "resolved": n_resolved}
        print(json.dumps(summary))
        return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--taxonomy-id", default="sdg")
    ap.add_argument("--max-tables", type=int, default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out-root", default="build/data/agent_mediated")
    a = ap.parse_args(argv)

    out_dir = REPO / a.out_root / a.taxonomy_id
    ws_path = out_dir / "working_set.json"
    if not ws_path.exists():
        raise SystemExit(f"no working set at {ws_path} — run "
                         f"scripts/build_agent_mediated_sdg.py first")
    ws = json.loads(ws_path.read_text())

    from atelier.engine.client import engine_status
    try:
        engine_status(timeout=5.0)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"engine unreachable (start with `just engine-serve`): {exc}")

    curator = Curator(ws, out_dir, workers=a.workers)
    curator.run(max_tables=a.max_tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
