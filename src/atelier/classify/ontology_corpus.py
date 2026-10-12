"""NHSVM rows and MaxSim prototypes from an Aegir release's ontology labels.

The held-back ``reference_code`` is the ontology label on the projected
relational column (Aegir ``ColumnSpec.slot_ref``). It trains the NHSVM
and supplies the prototype values MaxSim embeds. Blind classification
does not receive that key: callers that classify pass
``with_reference=False``.
"""

from __future__ import annotations

from typing import Any

from atelier.enrichment.llm_generator import EnrichmentGenerator, GenerationResult


class OntologyCorpusError(RuntimeError):
    """A configured Aegir release did not yield taxonomy-aligned labels."""


def release_dir(cfg: Any) -> str:
    """Configured release directory, or an empty string when unset."""
    return str(getattr(cfg, "aegir_release_dir", "") or "").strip()


def rows_from_labeled_columns(samples, category_set=None) -> list:
    """One NHSVM ``Row`` per column that carries a taxonomy ``reference_code``."""
    from atelier.classify.ml_train import _infer_column_type
    from atelier.optimize.svm.reflect import Row

    valid = _valid_codes(category_set)
    rows = []
    for table in samples:
        names = [c.name for c in getattr(table, "columns", []) or []]
        for col in table.columns:
            code = getattr(col, "reference_code", None)
            if not code or (valid is not None and code not in valid):
                continue
            values = _column_values(col)
            rows.append(Row(
                table=getattr(table, "name", "") or "",
                column=col.name,
                column_type=_infer_column_type(values[:10]) or "object",
                sample_values=values[:8],
                siblings_full=[n for n in names if n != col.name],
                mnemonic="",
                code=str(code),
            ))
    return rows


def labeled_rows_from_release(release_dir_path: str, category_set) -> list:
    """Supervised rows from the release. The reference key stays off the blind path."""
    from atelier.classify.aegir_release import load_aegir_release_samples

    samples = load_aegir_release_samples(release_dir_path, with_reference=True)
    return rows_from_labeled_columns(samples, category_set)


def exemplars_by_code(samples, category_set=None, *, limit: int = 10) -> dict[str, list[str]]:
    """Real column values grouped by ontology code, for MaxSim prototypes."""
    valid = _valid_codes(category_set)
    out: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for table in samples:
        for col in getattr(table, "columns", []) or []:
            code = getattr(col, "reference_code", None)
            if not code or (valid is not None and str(code) not in valid):
                continue
            code = str(code)
            bucket = out.setdefault(code, [])
            have = seen.setdefault(code, set())
            for text in _column_values(col):
                if text in have:
                    continue
                have.add(text)
                bucket.append(text)
                if len(bucket) >= limit:
                    break
    return out


def ontology_generator(cfg: Any, category_set):
    """MaxSim generator fed by the release, or None when no release is configured.

    An empty directory leaves the LLM enrichment path in place. A directory
    that is set and then fails to load, or that shares no codes with the
    taxonomy, fails closed.
    """
    path = release_dir(cfg)
    if not path:
        return None
    from atelier.classify.aegir_release import load_aegir_release_samples

    samples = load_aegir_release_samples(path, with_reference=True)
    exemplars = exemplars_by_code(samples, category_set)
    if not exemplars:
        raise OntologyCorpusError(
            f"Aegir release {path!r} has no reference_code values in the taxonomy"
        )
    return OntologyLabelGenerator(exemplars)


class OntologyLabelGenerator(EnrichmentGenerator):
    """Prototype text is the projected column's values, padded with the label."""

    def __init__(self, exemplars: dict[str, list[str]]) -> None:
        self._exemplars = {str(k): list(v) for k, v in exemplars.items()}

    @property
    def name(self) -> str:
        return "ontology:aegir-release"

    def generate(
        self,
        source_row: dict,
        *,
        taxonomy_context: dict,
        prior_attempt: GenerationResult | None = None,
        verifier_feedback: dict | None = None,
    ) -> GenerationResult:
        code = str(source_row.get("code") or "")
        label = str(source_row.get("label") or code)
        values = list(self._exemplars.get(code) or [])
        while len(values) < 3:
            values.append(label)
        expected = taxonomy_context.get("expected_parent_path")
        parent_path = list(expected) if isinstance(expected, list) else [label]
        hints: list[str] = []
        for raw in (source_row.get("mnemonic"), label):
            hint = str(raw or "").strip().lower().replace(" ", "_")
            if hint and hint not in hints:
                hints.append(hint)
        enrichment = {
            "prototype_values": values[:10],
            "value_patterns": [{"kind": "regex", "expr": ".+"}],
            "name_hints": hints or [label.lower() or code.lower() or "ontology"],
            "anti_examples": [],
            "parent_path": parent_path,
        }
        return GenerationResult(
            enrichment=enrichment,
            attempts=1,
            notes="ontology reference_code",
        )


def _valid_codes(category_set) -> set[str] | None:
    if category_set is None:
        return None
    cats = getattr(category_set, "all_categories", None)
    if cats is None:
        cats = getattr(category_set, "categories", []) or []
    return {c.code for c in cats}


def _column_values(col) -> list[str]:
    raw = getattr(col, "all_values", None) or getattr(col, "values", None) or []
    return [str(v).strip() for v in raw if str(v).strip()]
