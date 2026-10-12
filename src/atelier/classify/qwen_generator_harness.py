"""Ask Qwen, through grok, to extend the procedural value generator.

The workpack is an ontology slice of terms the keyword matcher does not
shape, plus the IRIs already accepted. The model writes
``out/extend_generators.py``. Accepted entries are installed into
``insitu_extensions.py`` and ``prepare_insitu`` uses them when the
entity IRI matches. Upstream column labels are not part of the pack.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from atelier.classify.extension_patterns import (
    accept_generators,
    code_iri,
    extensions_path,
    install_generators,
    load_generators,
)
from atelier.classify.qwen_synth_harness import (
    DEFAULT_PROVIDER,
    ONTOLOGY_FIELDS,
    PROVIDERS,
    QwenSynthError,
    grok_argv,
    run_generator,
)
from atelier.classify.synth_registry import _infer_generator

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SKILL_REL = Path(".grok") / "skills" / "extend-procedural-generator" / "SKILL.md"
_AGENT_REL = Path(".grok") / "agents" / "generator-extender.md"
_PROMPT = (
    "Follow the extend-procedural-generator skill. "
    "Requested codes are in TASK.md. "
    "Write out/extend_generators.py."
)


def write_generator_pack(
    dest: Path,
    annotations_csv: Path,
    *,
    limit: int = 24,
    installed: Path | None = None,
) -> tuple[Path, tuple[str, ...]]:
    """Write an isolated pack of uncovered codes. Returns the code tuple."""
    from atelier.classify.qwen_synth_harness import _copy_repo_file

    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise QwenSynthError(f"workpack directory is not empty: {dest}")
    installed = installed or extensions_path()
    installed_source = installed.read_text(encoding="utf-8") if installed.is_file() else ""
    batch = select_uncovered(annotations_csv, limit=limit, installed_source=installed_source)
    if not batch:
        raise QwenSynthError("every ontology code already has a value generator")
    dest.mkdir(parents=True, exist_ok=True)
    _write_ontology(dest / "ontology.csv", batch)
    codes = tuple(row["code"] for row in batch)
    (dest / "TASK.md").write_text(_task(codes), encoding="utf-8")
    (dest / "installed.py").write_text(
        installed_source or "PATTERNS = [\n]\n",
        encoding="utf-8",
    )
    (dest / "request.json").write_text(
        json.dumps({"codes": list(codes)}, indent=2) + "\n",
        encoding="utf-8",
    )
    _copy_repo_file(_SKILL_REL, dest / _SKILL_REL)
    _copy_repo_file(_AGENT_REL, dest / _AGENT_REL)
    (dest / "out").mkdir()
    return dest, codes


def select_uncovered(
    annotations_csv: Path,
    *,
    limit: int,
    installed_source: str = "",
) -> list[dict[str, str]]:
    """Ontology rows the keyword matcher and the installed patterns both miss."""
    if limit < 1:
        raise QwenSynthError("limit must be positive")
    installed = load_generators(installed_source) if installed_source.strip() else {}
    picked: list[dict[str, str]] = []
    for row in _read_rows(annotations_csv):
        cat = SimpleNamespace(
            label=row["label"],
            description=row["description"],
            common_names=row["common_names"],
        )
        if _infer_generator(cat) is not None:
            continue
        iri = code_iri(row["code"])
        if iri and iri in installed:
            continue
        item = {field: row[field] for field in ONTOLOGY_FIELDS}
        item["iri"] = iri or ""
        picked.append(item)
        if len(picked) >= limit:
            break
    return picked


def generator_argv(workdir: Path, *, provider: str = DEFAULT_PROVIDER, max_turns: int = 8) -> list[str]:
    return grok_argv(
        workdir,
        provider=provider,
        max_turns=max_turns,
        prompt=_PROMPT,
        agent=Path(workdir) / _AGENT_REL,
    )


def incorporate(
    workdir: Path,
    annotations_csv: Path,
    *,
    installed: Path | None = None,
) -> dict:
    """Validate ``out/extend_generators.py`` and install the IRIs that pass."""
    source_path = Path(workdir) / "out" / "extend_generators.py"
    if not source_path.is_file():
        return {"accepted": 0, "codes": [], "problems": ["extend_generators.py is missing"]}
    source = _python_source(source_path.read_text(encoding="utf-8"))
    rows = _read_rows(annotations_csv)
    request = json.loads((Path(workdir) / "request.json").read_text(encoding="utf-8"))
    wanted = set(request["codes"])
    batch = [row for row in rows if row["code"] in wanted]
    accepted, problems = accept_generators(source, batch)
    installed = installed or extensions_path()
    if accepted:
        install_generators([item.source for item in accepted], installed)
    codes = sorted(item.code for item in accepted)
    return {"accepted": len(accepted), "codes": codes, "problems": problems}


def _read_rows(path: Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise QwenSynthError(f"ontology csv is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            item = {field: str(raw.get(field) or "").strip() for field in ONTOLOGY_FIELDS}
            item["common_names"] = str(raw.get("common_names") or "").strip()
            rows.append(item)
    return rows


def _write_ontology(path: Path, rows: list[dict[str, str]]) -> None:
    fields = (*ONTOLOGY_FIELDS, "iri")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _task(codes: tuple[str, ...]) -> str:
    lines = [
        "# Request",
        "",
        "Follow the extend-procedural-generator skill.",
        "",
        "Codes the procedural generator does not cover:",
        "",
    ]
    lines.extend(f"- {code}" for code in codes)
    lines.append("")
    return "\n".join(lines)


def _python_source(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines) + "\n"


def _default_annotations() -> Path:
    return _REPO_ROOT / "external" / "sdg-corpora" / "vocabulary" / "annotations.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extend the procedural generator with Qwen")
    parser.add_argument("--annotations", type=Path, default=_default_annotations())
    parser.add_argument("--provider", choices=tuple(PROVIDERS), default=DEFAULT_PROVIDER)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=480.0)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    summaries = []
    failures = 0
    for index in range(args.batches):
        workdir = args.workdir
        if workdir is None or args.batches > 1:
            workdir = Path(f"/tmp/qwen-extend-{index:02d}")
        if workdir.exists():
            import shutil
            shutil.rmtree(workdir)
        try:
            pack, codes = write_generator_pack(
                workdir, args.annotations, limit=args.limit,
            )
        except QwenSynthError as exc:
            summaries.append({"batch": index, "error": str(exc)})
            break
        command = generator_argv(pack, provider=args.provider, max_turns=args.max_turns)
        if args.dry_run:
            print(json.dumps({"workdir": str(pack), "codes": list(codes), "argv": command}, indent=2))
            return 0
        completed = run_generator(
            pack,
            provider=args.provider,
            max_turns=args.max_turns,
            timeout_s=args.timeout,
            prompt=_PROMPT,
            agent=pack / _AGENT_REL,
        )
        report = incorporate(pack, args.annotations)
        report["batch"] = index
        report["returncode"] = completed.returncode
        summaries.append(report)
        print(json.dumps(report, indent=2), flush=True)
        if completed.returncode != 0 or not report["accepted"]:
            failures += 1
            if failures >= 2:
                break
            continue
        failures = 0
    accepted = sum(int(item.get("accepted") or 0) for item in summaries)
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
