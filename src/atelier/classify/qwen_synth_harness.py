"""ACP workpack that asks a Qwen thinking model to author synthetic tables.

The model sees an ontology slice (code, label, abbrev, parent, description)
inside an isolated directory, plus a skill and an agent profile. Upstream
column labels are a scoring input after the process exits. They are not
written into the workpack and this module's pack writer does not read them.

``local`` is the Gaius thinking endpoint registered as ``qwen-thinking``.
``cerebras`` is the budgeted hosted route (``qwen-thinking-cerebras``) and
is selected only when the caller passes that provider.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SKILL_REL = Path(".grok") / "skills" / "synthetic-relational-corpus" / "SKILL.md"
_AGENT_REL = Path(".grok") / "agents" / "relational-synth.md"

ONTOLOGY_FIELDS = ("code", "label", "abbrev", "parent_code", "description")
PROVIDERS = {
    "local": "qwen-thinking",
    "cerebras": "qwen-thinking-cerebras",
}
DEFAULT_PROVIDER = "local"
TABLE_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
_TOKEN = re.compile(r"[A-Za-z]+|\d+")

Runner = Callable[[list[str], dict[str, str], float | None], subprocess.CompletedProcess]


class QwenSynthError(RuntimeError):
    """The workpack could not be built or the manifest could not be read."""


@dataclass(frozen=True)
class Workpack:
    workdir: Path
    codes: tuple[str, ...]
    rows_per_table: int


@dataclass(frozen=True)
class Assignment:
    table: str
    column: str
    code: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class SynthScore:
    requested: tuple[str, ...]
    emitted_codes: tuple[str, ...]
    coverage: float
    parse_ok: bool
    name_jaccard: float | None
    value_jaccard: float | None
    problems: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "requested": list(self.requested),
            "emitted_codes": list(self.emitted_codes),
            "coverage": self.coverage,
            "parse_ok": self.parse_ok,
            "name_jaccard": self.name_jaccard,
            "value_jaccard": self.value_jaccard,
            "problems": list(self.problems),
        }


def write_workpack(
    dest: Path,
    annotations_csv: Path,
    codes: Sequence[str],
    *,
    rows_per_table: int = 8,
) -> Workpack:
    """Write an isolated directory whose only catalogue is an ontology slice."""
    dest = Path(dest)
    wanted = tuple(dict.fromkeys(str(c).strip() for c in codes if str(c).strip()))
    if not wanted:
        raise QwenSynthError("at least one ontology code is required")
    if rows_per_table < 1:
        raise QwenSynthError("rows_per_table must be positive")
    if dest.exists() and any(dest.iterdir()):
        raise QwenSynthError(f"workpack directory is not empty: {dest}")
    sliced = slice_ontology(annotations_csv, wanted)
    dest.mkdir(parents=True, exist_ok=True)
    _write_ontology(dest / "ontology.csv", sliced)
    (dest / "TASK.md").write_text(_task_markdown(wanted, rows_per_table), encoding="utf-8")
    (dest / "request.json").write_text(
        json.dumps({"codes": list(wanted), "rows_per_table": rows_per_table}, indent=2) + "\n",
        encoding="utf-8",
    )
    _copy_repo_file(_SKILL_REL, dest / _SKILL_REL)
    _copy_repo_file(_AGENT_REL, dest / _AGENT_REL)
    (dest / "out" / "tables").mkdir(parents=True)
    return Workpack(workdir=dest, codes=wanted, rows_per_table=rows_per_table)


def slice_ontology(annotations_csv: Path, codes: Sequence[str]) -> list[dict[str, str]]:
    """Ontology rows for ``codes`` plus ancestors. Catalogue columns only."""
    rows = _read_ontology(annotations_csv)
    by_code = {row["code"]: row for row in rows if row["code"]}
    missing = [code for code in codes if code not in by_code]
    if missing:
        raise QwenSynthError(
            f"ontology has no row for {', '.join(missing)}"
        )
    wanted = set(codes)
    pending = list(codes)
    while pending:
        code = pending.pop()
        parent = by_code[code]["parent_code"]
        if parent and parent in by_code and parent not in wanted:
            wanted.add(parent)
            pending.append(parent)
    return [row for row in rows if row["code"] in wanted]


def pack_inputs(workdir: Path) -> list[Path]:
    """Files the generator is given. Excludes anything under ``out/``."""
    workdir = Path(workdir)
    kept: list[Path] = []
    for path in sorted(workdir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(workdir)
        if rel.parts and rel.parts[0] == "out":
            continue
        kept.append(path)
    return kept


def grok_argv(
    workdir: Path,
    *,
    provider: str = DEFAULT_PROVIDER,
    max_turns: int = 8,
) -> list[str]:
    """Headless grok invocation. The model id comes from ``provider``."""
    if provider not in PROVIDERS:
        raise QwenSynthError(
            f"unknown provider {provider!r}; expected one of {', '.join(PROVIDERS)}"
        )
    if max_turns < 1:
        raise QwenSynthError("max_turns must be positive")
    workdir = Path(workdir)
    prompt = (
        "Follow the synthetic-relational-corpus skill. "
        "Requested codes and the row count are in TASK.md. "
        "Write the tables and the manifest under out/."
    )
    argv = [
        "grok",
        "-p", prompt,
        "--verbatim",
        "-m", PROVIDERS[provider],
        "--agent", str(workdir / _AGENT_REL),
        "--cwd", str(workdir),
        "--trust",
        "--sandbox", "strict",
        "--max-turns", str(max_turns),
        "--output-format", "json",
        "--always-approve",
        "--no-subagents",
        "--disable-web-search",
        "--tools", "read_file,write,list_dir",
        "--disallowed-tools",
        "search_tool,use_tool,run_terminal_cmd,web_search,web_fetch",
    ]
    # Kernel-enforced. ``grok_env`` supplies bubblewrap when the system
    # image does not; without it, grok refuses this list rather than
    # starting with the paths readable.
    for rule in (
        f"Read({_REPO_ROOT}/**)",
        f"Read({Path.home()}/.grok/sessions/**)",
    ):
        argv.extend(("--deny", rule))
    return argv


def grok_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for the grok child. Vendor skill trees stay out of the pack."""
    env = dict(os.environ if base is None else base)
    env["GROK_CLAUDE_SKILLS_ENABLED"] = "false"
    env["GROK_CURSOR_SKILLS_ENABLED"] = "false"
    if shutil.which("bwrap", path=env.get("PATH")) is None:
        bindir = _bwrap_bindir()
        if bindir is not None:
            env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    return env


def _bwrap_bindir() -> str | None:
    """Directory of a runnable bubblewrap, including a nix-store copy."""
    store = Path("/nix/store")
    if not store.is_dir():
        return None
    candidates = sorted(
        (
            path for path in store.glob("*-bubblewrap-*/bin/bwrap")
            if path.is_file() and "-dev" not in path.parent.parent.name
        ),
        reverse=True,
    )
    for path in candidates:
        try:
            probe = subprocess.run(
                [str(path), "--version"],
                capture_output=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0:
            return str(path.parent)
    return None


def run_generator(
    workdir: Path,
    *,
    provider: str = DEFAULT_PROVIDER,
    max_turns: int = 8,
    timeout_s: float | None = None,
    runner: Runner | None = None,
) -> subprocess.CompletedProcess:
    """Run grok in ``workdir``. A ``runner`` replaces the process in tests."""
    workdir = Path(workdir)
    argv = grok_argv(workdir, provider=provider, max_turns=max_turns)
    env = grok_env()
    call = runner or _subprocess_runner
    started = time.monotonic()
    completed = call(argv, env, timeout_s)
    log_dir = workdir / "harness"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "argv.json").write_text(json.dumps(argv, indent=2) + "\n", encoding="utf-8")
    (log_dir / "grok.stdout").write_text(completed.stdout or "", encoding="utf-8")
    (log_dir / "grok.stderr").write_text(completed.stderr or "", encoding="utf-8")
    (log_dir / "timing.json").write_text(
        json.dumps({
            "provider": provider,
            "model": PROVIDERS[provider],
            "seconds": round(time.monotonic() - started, 3),
            "returncode": completed.returncode,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return completed


def score_workdir(workdir: Path, gold: Sequence[object] | None = None) -> SynthScore:
    """Score ``out/`` against optional labeled columns.

    ``gold`` items expose ``table``, ``column``, ``code``, and ``values``.
    Passing None reports coverage only and leaves both Jaccard scores empty.
    """
    workdir = Path(workdir)
    request = _read_request(workdir)
    requested = tuple(request["codes"])
    allowed = _ontology_codes(workdir)
    assignments, problems = _load_assignments(workdir, allowed)
    emitted = tuple(dict.fromkeys(item.code for item in assignments if item.code))
    covered = [code for code in requested if code in set(emitted)]
    coverage = (len(covered) / len(requested)) if requested else 0.0
    name_j, value_j = _jaccard_against_gold(requested, assignments, gold)
    return SynthScore(
        requested=requested,
        emitted_codes=emitted,
        coverage=coverage,
        parse_ok=not problems,
        name_jaccard=name_j,
        value_jaccard=value_j,
        problems=tuple(problems),
    )


def score_with_corpora(workdir: Path, corpora: str | Path | None = None) -> SynthScore:
    """Score with sdg-corpora labels. Called after generation, never while packing."""
    from atelier.classify.ontology_corpus import load_labeled_columns

    return score_workdir(workdir, load_labeled_columns(corpora))


def _subprocess_runner(
    argv: list[str],
    env: dict[str, str],
    timeout_s: float | None,
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            argv,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        stderr = f"{stderr}\nQwenSynthError: grok exceeded {timeout_s}s\n"
        return subprocess.CompletedProcess(argv, 124, stdout=stdout, stderr=stderr)


def _read_ontology(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise QwenSynthError(f"ontology csv is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for raw in reader:
            rows.append({
                field: str(raw.get(field) or "").strip()
                for field in ONTOLOGY_FIELDS
            })
    if not rows:
        raise QwenSynthError(f"ontology csv has no rows: {path}")
    return rows


def _write_ontology(path: Path, rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ONTOLOGY_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in ONTOLOGY_FIELDS})


def _task_markdown(codes: Sequence[str], rows_per_table: int) -> str:
    lines = [
        "# Request",
        "",
        "Follow the synthetic-relational-corpus skill.",
        "",
        "Requested codes:",
        "",
    ]
    lines.extend(f"- {code}" for code in codes)
    lines.extend(["", f"Data rows per table: {rows_per_table}", ""])
    return "\n".join(lines)


def _copy_repo_file(relative: Path, dest: Path) -> None:
    source = _REPO_ROOT / relative
    if not source.is_file():
        raise QwenSynthError(f"harness file is missing: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)


def _read_request(workdir: Path) -> dict:
    path = workdir / "request.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QwenSynthError(f"request.json is not readable: {exc}") from exc
    codes = payload.get("codes")
    if not isinstance(codes, list) or not all(isinstance(c, str) and c for c in codes):
        raise QwenSynthError("request.json codes must be a list of strings")
    return payload


def _ontology_codes(workdir: Path) -> set[str]:
    path = workdir / "ontology.csv"
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            str(row.get("code") or "").strip()
            for row in csv.DictReader(handle)
            if str(row.get("code") or "").strip()
        }


def _load_assignments(
    workdir: Path,
    allowed_codes: set[str],
) -> tuple[list[Assignment], list[str]]:
    problems: list[str] = []
    path = workdir / "out" / "manifest.json"
    if not path.is_file():
        return [], ["manifest.json is missing"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], [f"manifest.json is not JSON: {exc.msg}"]
    if not isinstance(payload, list):
        return [], ["manifest.json must be a JSON list"]
    assignments: list[Assignment] = []
    seen_files: dict[str, dict[str, list[str]]] = {}
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            problems.append(f"manifest[{index}] is not an object")
            continue
        table = str(item.get("table") or "").strip()
        column = str(item.get("column") or "").strip()
        code = str(item.get("code") or "").strip()
        if not table or not column:
            problems.append(f"manifest[{index}] needs table and column")
            continue
        if not TABLE_NAME.fullmatch(table):
            problems.append(f"manifest[{index}] table {table!r} is not a safe name")
            continue
        if code and allowed_codes and code not in allowed_codes:
            problems.append(f"manifest[{index}] code {code!r} is not in ontology.csv")
            continue
        header, rows = seen_files.get(table) or _read_table(workdir, table, problems)
        seen_files.setdefault(table, (header, rows))  # type: ignore[arg-type]
        if header is None:
            continue
        if column not in header:
            problems.append(f"{table}.csv has no column {column!r}")
            continue
        values = tuple(row[column] for row in rows if row.get(column, "").strip())
        if not values:
            problems.append(f"{table}.{column} has no cell values")
            continue
        assignments.append(Assignment(table=table, column=column, code=code, values=values))
    return assignments, problems


def _read_table(
    workdir: Path,
    table: str,
    problems: list[str],
) -> tuple[dict[str, int] | None, list[dict[str, str]]]:
    path = workdir / "out" / "tables" / f"{table}.csv"
    if not path.is_file():
        problems.append(f"out/tables/{table}.csv is missing")
        return None, []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = {name.strip(): name for name in (reader.fieldnames or []) if name and name.strip()}
        rows = []
        for raw in reader:
            rows.append({key.strip(): str(value or "").strip() for key, value in raw.items() if key})
    if not header:
        problems.append(f"out/tables/{table}.csv has no header")
        return None, []
    return header, rows


def _jaccard_against_gold(
    requested: Sequence[str],
    assignments: Sequence[Assignment],
    gold: Sequence[object] | None,
) -> tuple[float | None, float | None]:
    if gold is None:
        return None, None
    by_code: dict[str, list[object]] = {}
    for item in gold:
        code = str(getattr(item, "code", "") or "")
        if code:
            by_code.setdefault(code, []).append(item)
    name_scores: list[float] = []
    value_scores: list[float] = []
    generated: dict[str, list[Assignment]] = {}
    for item in assignments:
        if item.code:
            generated.setdefault(item.code, []).append(item)
    for code in requested:
        gold_cols = by_code.get(code) or []
        if not gold_cols:
            continue
        gen_cols = generated.get(code) or []
        name_scores.append(_pair_jaccard(
            _tokens_from(item.column for item in gen_cols),
            _tokens_from(str(getattr(item, "column", "") or "") for item in gold_cols),
        ))
        gen_values: list[str] = []
        gold_values: list[str] = []
        for item in gen_cols:
            gen_values.extend(item.values)
        for item in gold_cols:
            gold_values.extend(str(v) for v in (getattr(item, "values", []) or []))
        value_score = jaccard(_tokens_from(gen_values), _tokens_from(gold_values))
        if value_score is not None:
            value_scores.append(value_score)
    return _macro(name_scores), _macro(value_scores)


def _pair_jaccard(left: set[str], right: set[str]) -> float:
    score = jaccard(left, right)
    return 0.0 if score is None else score


def jaccard(left: set[str], right: set[str]) -> float | None:
    if not left and not right:
        return None
    union = left | right
    return len(left & right) / len(union)


def _tokens_from(texts: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for text in texts:
        for token in _TOKEN.findall(str(text)):
            token = token.lower()
            if len(token) > 1:
                out.add(token)
    return out


def _macro(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _default_annotations() -> Path:
    return _REPO_ROOT / "external" / "sdg-corpora" / "vocabulary" / "annotations.csv"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qwen thinking synthetic-corpus harness")
    parser.add_argument("--codes", required=True, help="Comma-separated ontology codes")
    parser.add_argument("--annotations", type=Path, default=_default_annotations())
    parser.add_argument("--provider", choices=tuple(PROVIDERS), default=DEFAULT_PROVIDER)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    codes = [part.strip() for part in str(args.codes).split(",") if part.strip()]
    workdir = args.workdir
    if workdir is None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        workdir = Path(os.environ.get("TMPDIR", "/tmp")) / f"qwen-synth-{stamp}"
    pack = write_workpack(workdir, args.annotations, codes, rows_per_table=args.rows)
    command = grok_argv(pack.workdir, provider=args.provider, max_turns=args.max_turns)
    if args.dry_run:
        print(json.dumps({"workdir": str(pack.workdir), "argv": command}, indent=2))
        return 0
    completed = run_generator(
        pack.workdir,
        provider=args.provider,
        max_turns=args.max_turns,
        timeout_s=args.timeout,
    )
    if args.score:
        score = score_with_corpora(pack.workdir)
        (pack.workdir / "harness" / "score.json").write_text(
            json.dumps(score.as_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(score.as_dict(), indent=2))
    return int(completed.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
