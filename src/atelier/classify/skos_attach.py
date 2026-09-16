"""Attach SKOS reference_code to sample columns by name/label tokens."""

from __future__ import annotations

import re


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def attach_skos_reference(samples, category_set) -> dict[str, str]:
    """Map columns to SKOS codes by label / abbrev / common_names tokens."""
    by = getattr(category_set, "all_by_code", None)
    if callable(by):
        by = by()
    cats = list(by.values()) if isinstance(by, dict) else list(getattr(category_set, "categories", []) or [])
    index: list[tuple[str, set[str]]] = []
    for c in cats:
        tokens = {_norm(c.code), _norm(c.label), _norm(getattr(c, "abbrev", "") or "")}
        extra = getattr(c, "common_names", "") or ""
        for part in re.split(r"[|,;/]", extra):
            n = _norm(part)
            if n:
                tokens.add(n)
        tokens.discard("")
        index.append((c.code, tokens))
    ref: dict[str, str] = {}
    for table in samples:
        for col in table.columns:
            name = _norm(col.name)
            if not name:
                continue
            hit = None
            for code, tokens in index:
                if name in tokens or any(t and (t in name or name in t) for t in tokens if len(t) >= 3):
                    hit = code
                    break
            if hit:
                col.reference_code = hit
                ref[col.name] = hit
                qn = getattr(col, "qualified_name", "") or f"{table.name}.{col.name}"
                ref[qn] = hit
    return ref
