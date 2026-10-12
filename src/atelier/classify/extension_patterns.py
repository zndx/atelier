"""Load and check exact-IRI value generators.

The coding harness writes ``GENERATORS``. Each key is one vocabulary
IRI, ``https://signals.zndx.org/sdg#...``. A generator applies only
when the entity's IRI is that key. There is no label search and no
stand-in value for a term that has no generator.
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
"""Exact IRI value generators for ontology terms.

The coding harness writes GENERATORS. insitu_corpus applies an entry
only when the entity IRI equals the key.
"""

import random

GENERATORS = {
'''


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


def code_iri(code: str) -> str | None:
    """Vocabulary IRI for an ``SDG.*`` code. Exact; not a search."""
    text = (code or "").strip()
    if not text.startswith("SDG."):
        return None
    fragment = text[4:].replace(".", "_")
    if not _FRAGMENT.fullmatch(fragment):
        return None
    return _IRI_PREFIX + fragment


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
        reason = _reject_reason(generate, row)
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


def _reject_reason(generate, row: dict) -> str | None:
    try:
        values = {str(generate(random.Random(seed))) for seed in range(12)}
    except Exception as exc:
        return f"generator raised {type(exc).__name__}"
    if len(values) < 2:
        return "generator returned one constant"
    if any(not value.strip() or len(value) > 80 for value in values):
        return "generator returned an empty or oversized value"
    label = str(row.get("label") or "")
    description = str(row.get("description") or "")
    if label and all(value.startswith(label) for value in values):
        return "generator repeats the label"
    if description and any(value in description and len(value) > 24 for value in values):
        return "generator copies the description"
    return None
