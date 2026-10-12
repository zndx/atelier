"""Load and check relational generators bound to published SKOS IRIs.

The coding harness writes ``GENERATORS``. Each key is one concept IRI
from ``vocabulary/vocabulary.ttl``, ``https://signals.zndx.org/sdg#...``.
The callable returns one synthetic table. Exactly one column of that
table carries the same IRI. A generator applies only when a projected
column is assigned that IRI.
"""

from __future__ import annotations

import ast
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_IRI_PREFIX = "https://signals.zndx.org/sdg#"
_FRAGMENT = re.compile(r"[A-Z0-9_]+")
_IRI = re.compile(r"https://signals\.zndx\.org/sdg#[A-Z0-9_]+")
_CALLS = frozenset({
    "str", "int", "float", "len", "range", "chr", "ord", "hex",
    "round", "abs", "min", "max", "sum",
})
_RNG_ATTRS = frozenset({
    "choice", "randint", "randrange", "uniform", "random", "getrandbits",
})
_HEADER = '''\
"""Relational generators for published SKOS IRIs.

The coding harness writes GENERATORS. Each callable returns one table.
insitu_corpus applies an entry only when a projected column is assigned
that IRI.
"""

import random

GENERATORS = {
'''
_CONCEPT_IRI = re.compile(
    r"<(https://signals\.zndx\.org/sdg#[A-Za-z0-9_]+)> a skos:Concept\b"
)
_concepts: set[str] | None = None


class ExtensionError(RuntimeError):
    """A generator source broke the IRI contract."""


@dataclass(frozen=True)
class AcceptedGenerator:
    source: str
    iri: str
    generate: Callable[[random.Random], str]
    code: str


def extensions_path() -> Path:
    return Path(__file__).resolve().parent / "insitu_extensions.py"


def published_concept_iris() -> set[str]:
    """SKOS concept IRIs entailed in the published vocabulary."""
    global _concepts
    if _concepts is None:
        path = (
            Path(__file__).resolve().parents[3]
            / "external" / "sdg-corpora" / "vocabulary" / "vocabulary.ttl"
        )
        if not path.is_file():
            raise ExtensionError(f"published SKOS vocabulary is missing at {path}")
        found = set(_CONCEPT_IRI.findall(path.read_text(encoding="utf-8")))
        if not found:
            raise ExtensionError(f"published SKOS vocabulary at {path} has no concepts")
        _concepts = found
    return _concepts


def code_iri(code: str) -> str | None:
    """Published SKOS IRI for an ``SDG.*`` code.

    The fragment drops the ``SDG.`` prefix and turns dots into
    underscores. The IRI must be a ``skos:Concept`` in the published
    vocabulary. An unpublished string is not a business term.
    """
    text = (code or "").strip()
    if not text.startswith("SDG."):
        return None
    fragment = text[4:].replace(".", "_")
    if not _FRAGMENT.fullmatch(fragment):
        return None
    iri = _IRI_PREFIX + fragment
    if iri not in published_concept_iris():
        return None
    return iri


def load_generators(source: str) -> dict[str, Callable]:
    """Exec a gated extension and return ``IRI -> rng -> str``."""
    module = _execute(_parse(source))
    generators = module.get("GENERATORS")
    if not isinstance(generators, dict):
        raise ExtensionError("GENERATORS must be a dict")
    out: dict[str, Callable] = {}
    for key, generate in generators.items():
        if not isinstance(key, str) or _IRI.fullmatch(key) is None:
            raise ExtensionError(f"{key!r} is not an SDG vocabulary IRI")
        if not callable(generate):
            raise ExtensionError(f"{key} is not a callable")
        out[key] = generate
    return out


def accept_generators(
    source: str,
    batch: list[dict],
) -> tuple[list[AcceptedGenerator], list[str]]:
    """Keep entries whose key is the exact IRI of a requested term."""
    problems: list[str] = []
    try:
        tree = _parse(source)
        module = _execute(tree)
        snippets = _entry_sources(source, tree)
    except ExtensionError as exc:
        return [], [str(exc)]
    generators = module.get("GENERATORS")
    if not isinstance(generators, dict) or len(generators) != len(snippets):
        return [], ["GENERATORS did not match the source entries"]
    by_iri = {}
    for row in batch:
        iri = code_iri(str(row.get("code") or ""))
        if iri:
            by_iri[iri] = row
    accepted: list[AcceptedGenerator] = []
    seen: set[str] = set()
    for (iri, generate), snippet in zip(generators.items(), snippets):
        if iri in seen:
            problems.append(f"{iri} is repeated")
            continue
        seen.add(iri)
        row = by_iri.get(iri)
        if row is None:
            problems.append(f"{iri} is not a requested IRI")
            continue
        reason = _reject_reason(generate, row, iri)
        if reason:
            problems.append(f"{iri} {reason}")
            continue
        accepted.append(AcceptedGenerator(
            source=snippet.strip(),
            iri=iri,
            generate=generate,
            code=str(row["code"]),
        ))
    return accepted, problems


def render_module(snippets: list[str]) -> str:
    body = ",\n".join(f"    {snippet.strip()}" for snippet in snippets)
    if body:
        body += ",\n"
    return _HEADER + body + "}\n"


def install_generators(snippets: list[str], path: Path | None = None) -> int:
    """Rewrite the installed module with ``snippets`` plus what is already there."""
    path = path or extensions_path()
    existing: list[str] = []
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        if text.strip():
            existing = _entry_sources(text, _parse(text))
    seen: set[str] = set()
    kept: list[str] = []
    for snippet in existing + list(snippets):
        key = _iri_key(snippet)
        if key in seen:
            continue
        seen.add(key)
        kept.append(snippet.strip())
    path.write_text(render_module(kept), encoding="utf-8")
    load_generators(path.read_text(encoding="utf-8"))
    return len(kept)


def _parse(source: str) -> ast.AST:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ExtensionError(f"extension is not Python: {exc.msg}") from exc
    _audit(tree)
    return tree


def _audit(tree: ast.AST) -> None:
    if not isinstance(tree, ast.Module):
        raise ExtensionError("extension must be a module")
    assigns = 0
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            if names != ["random"]:
                raise ExtensionError(f"import of {', '.join(names)} is not allowed")
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.Assign):
            assigns += 1
            targets = node.targets
            if (
                len(targets) != 1
                or not isinstance(targets[0], ast.Name)
                or targets[0].id != "GENERATORS"
                or not isinstance(node.value, ast.Dict)
            ):
                raise ExtensionError("the only assignment is GENERATORS = {...}")
            _audit_dict(node.value)
            continue
        raise ExtensionError(
            f"statement {type(node).__name__} is not allowed in an extension"
        )
    if assigns != 1:
        raise ExtensionError("extension must assign GENERATORS once")


def _audit_dict(node: ast.Dict) -> None:
    for key, value in zip(node.keys, node.values):
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise ExtensionError("each key must be an IRI string")
        if _IRI.fullmatch(key.value) is None:
            raise ExtensionError(f"{key.value!r} is not an SDG vocabulary IRI")
        if not isinstance(value, ast.Lambda) or len(value.args.args) != 1 or value.args.args[0].arg != "rng":
            raise ExtensionError("each value must be lambda rng: ...")
        if not _uses_rng(value.body):
            raise ExtensionError("the lambda must call rng")
        _audit_calls(value)


def _uses_rng(node: ast.AST) -> bool:
    found = False
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == "rng" and isinstance(child.ctx, ast.Load):
            found = True
        if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
            if child.value.id == "rng":
                found = True
    return found


def _audit_calls(fn: ast.Lambda) -> None:
    for child in ast.walk(fn):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            if child.id not in _CALLS and child.id != "rng":
                raise ExtensionError(f"name {child.id} is not available in a lambda")
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name):
            if func.id not in _CALLS:
                raise ExtensionError(f"call {func.id} is not allowed")
            continue
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "rng" and func.attr in _RNG_ATTRS:
                continue
            raise ExtensionError(f"call {func.value.id}.{func.attr} is not allowed")
        raise ExtensionError("indirect calls are not allowed")


def _execute(tree: ast.AST) -> dict:
    def _import(name, _globals=None, _locals=None, fromlist=(), level=0):
        if name == "random":
            return random
        raise ImportError(name)

    builtins = {
        "range": range, "str": str, "int": int, "float": float, "len": len,
        "list": list, "dict": dict, "tuple": tuple, "chr": chr, "ord": ord,
        "hex": hex, "round": round, "abs": abs, "min": min, "max": max,
        "sum": sum, "True": True, "False": False, "None": None,
        "__import__": _import,
    }
    module: dict[str, Any] = {"__builtins__": builtins}
    try:
        exec(compile(tree, "<extension>", "exec"), module)
    except ExtensionError:
        raise
    except Exception as exc:
        raise ExtensionError(f"extension failed: {type(exc).__name__}: {exc}") from exc
    return module


def _entry_sources(source: str, tree: ast.AST) -> list[str]:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "GENERATORS" and isinstance(node.value, ast.Dict):
            snippets = []
            for key, value in zip(node.value.keys, node.value.values):
                key_src = ast.get_source_segment(source, key)
                value_src = ast.get_source_segment(source, value)
                if not key_src or not value_src:
                    raise ExtensionError("a generator entry has no source")
                snippets.append(f"{key_src}: {value_src}")
            return snippets
    raise ExtensionError("GENERATORS assignment is missing")


def _iri_key(snippet: str) -> str:
    match = re.search(r"""(['\"])(https://signals\.zndx\.org/sdg#[A-Z0-9_]+)\1""", snippet)
    return match.group(2) if match else snippet


def relational_table_problem(
    value,
    iri: str,
    *,
    label: str = "",
    description: str = "",
) -> str | None:
    """Why ``value`` is not one synthetic table for ``iri``."""
    if not isinstance(value, dict):
        return "generator must return a table"
    table = value.get("table")
    columns = value.get("columns")
    if not isinstance(table, str) or not table.strip() or len(table) > 80:
        return "table name is empty or oversized"
    if not isinstance(columns, list) or len(columns) < 2:
        return "table needs the labeled column and a sibling"
    labeled = 0
    seen_names: set[str] = set()
    for column in columns:
        if not isinstance(column, dict):
            return "column is not a record"
        name = column.get("name")
        cells = column.get("values")
        if not isinstance(name, str) or not name.strip() or name in seen_names:
            return "column name is empty or repeated"
        seen_names.add(name)
        if not isinstance(cells, list) or not cells:
            return "column has no values"
        if any(
            not isinstance(item, str) or not item.strip() or len(item) > 80
            for item in cells
        ):
            return "column value is empty or oversized"
        column_iri = str(column.get("iri") or "")
        if column_iri == iri:
            labeled += 1
        elif column_iri:
            return "a sibling column carries a different IRI"
        if label and all(item.startswith(label) for item in cells):
            return "generator repeats the label"
        if description and any(item in description and len(item) > 24 for item in cells):
            return "generator copies the description"
    if labeled != 1:
        return "exactly one column carries the IRI"
    return None


def _labeled_values(value: dict, iri: str) -> list[str]:
    values: list[str] = []
    for column in value["columns"]:
        if str(column.get("iri") or "") == iri:
            values.extend(column["values"])
    return values


def _reject_reason(generate, row: dict, iri: str) -> str | None:
    label = str(row.get("label") or "")
    description = str(row.get("description") or "")
    seen: set[str] = set()
    try:
        for seed in range(12):
            value = generate(random.Random(seed))
            problem = relational_table_problem(
                value, iri, label=label, description=description,
            )
            if problem:
                return problem
            seen.update(_labeled_values(value, iri))
    except Exception as exc:
        return f"generator raised {type(exc).__name__}"
    if len(seen) < 2:
        return "generator returned one constant"
    return None
