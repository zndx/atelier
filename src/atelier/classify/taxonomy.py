"""Taxonomy management for column type annotation.

Manages the controlled vocabulary (annotations) as a hierarchical category set.
Supports loading from:
- Hive tables (default.annotations via cml.data_v1)
- Cached JSON (build/data/annotations/annotations.json)
- Mock fixtures (for devenv/CI testing)

Ported from signals/src/sigint/category_set.py — adapted for atelier's
hive-based annotations with dot-notation hierarchical codes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path


@dataclass(frozen=True)
class ReferenceCategory:
    """A single category in the controlled vocabulary."""

    code: str           # Hierarchical dot-notation: "0", "0.1", "1.1.1.1.1.1.1"
    label: str          # Human-readable name: "Payment Card Number" (from ontology)
    embedding_text: str  # Pre-built text for the embedding model
    abbrev: str = ""    # Formal code / mnemonic: "PAN" (from annotation)
    description: str = ""
    common_names: str = ""  # Pipe-separated aliases: "Credit Card|CC|DPAN"
    taxonomy: str = "annotations"
    parent_code: str | None = None
    # Sensitivity ratings per data subject role
    sensitivity: dict[str, str] | None = None

    @property
    def atlas_type_name(self) -> str:
        safe = self.label.replace(" ", "").replace("/", "_")
        return f"ANN_{self.code.replace('.', '_')}_{safe}"


@dataclass
class CategorySet:
    """An ordered collection of reference categories."""

    name: str
    categories: list[ReferenceCategory]

    @cached_property
    def by_code(self) -> dict[str, ReferenceCategory]:
        return {c.code: c for c in self.categories}

    @cached_property
    def by_abbrev(self) -> dict[str, ReferenceCategory]:
        return {c.abbrev: c for c in self.categories if c.abbrev}


class HierarchicalCategorySet(CategorySet):
    """A CategorySet with full parent-child tree navigation.

    ``categories`` (inherited) returns leaf-only for backward compat.
    ``all_categories`` includes both leaves and internal (parent) nodes.
    """

    def __init__(
        self,
        name: str,
        categories: list[ReferenceCategory],
        all_categories: list[ReferenceCategory],
    ) -> None:
        super().__init__(name=name, categories=categories)
        self.all_categories = all_categories

    @cached_property
    def all_by_code(self) -> dict[str, ReferenceCategory]:
        return {c.code: c for c in self.all_categories}

    @cached_property
    def parent(self) -> dict[str, str | None]:
        return {c.code: c.parent_code for c in self.all_categories}

    @cached_property
    def children(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for c in self.all_categories:
            if c.parent_code is not None:
                result.setdefault(c.parent_code, []).append(c.code)
        return result

    @cached_property
    def leaf_codes(self) -> frozenset[str]:
        parents_with_children = set(self.children.keys())
        return frozenset(
            c.code for c in self.all_categories
            if c.code not in parents_with_children
        )

    def descendants(self, code: str) -> frozenset[str]:
        """All descendant leaf codes of *code*."""
        if code in self.leaf_codes:
            return frozenset({code})
        result: set[str] = set()
        stack = [code]
        while stack:
            current = stack.pop()
            for child in self.children.get(current, []):
                if child in self.leaf_codes:
                    result.add(child)
                else:
                    stack.append(child)
        return frozenset(result)

    def ancestors(self, code: str) -> list[str]:
        """Path from parent to root (does not include *code* itself)."""
        result: list[str] = []
        current = self.parent.get(code)
        while current is not None:
            result.append(current)
            current = self.parent.get(current)
        return result


# ── Helpers ──────────────────────────────────────────────────────────


def _camel_to_words(name: str) -> str:
    """Split CamelCase into lowercase words."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).lower()


def _build_parents(
    leaf_rows: list[dict],
) -> list[ReferenceCategory]:
    """Build synthetic parent nodes from dot-notation codes."""
    existing_codes: set[str] = set()
    parents: dict[str, ReferenceCategory] = {}

    for row in leaf_rows:
        existing_codes.add(row["code"])

    for row in leaf_rows:
        parts = row["code"].split(".")
        for depth in range(1, len(parts)):
            parent_code = ".".join(parts[:depth])
            if parent_code in existing_codes or parent_code in parents:
                continue
            grandparent = ".".join(parts[:depth - 1]) if depth > 1 else None
            parents[parent_code] = ReferenceCategory(
                code=parent_code,
                label=f"Level{depth}_{parent_code}",
                embedding_text=f"level {depth} category {parent_code}",
                taxonomy="annotations",
                parent_code=grandparent,
            )

    return list(parents.values())


# ── Hive loader ──────────────────────────────────────────────────────


def load_annotations_from_hive(
    cfg,
    connection_name: str | None = None,
    *,
    hierarchical: bool = True,
) -> CategorySet:
    """Load annotations from default.annotations via CAI Data Platform.

    Args:
        cfg: AtelierConfig with data connection settings.
        connection_name: CAI connection name. Falls back to first configured.
        hierarchical: Return HierarchicalCategorySet if True.

    Returns:
        CategorySet loaded from hive annotations table.
    """
    try:
        import cml.data_v1 as cmldata
    except ImportError:
        raise RuntimeError(
            "cml.data_v1 not available — run on CAI or use load_annotations_from_json()"
        )

    if connection_name is None:
        names = cfg.cml_data_connection_names
        if not names:
            raise ValueError("No data connections configured (ATELIER_DATA_CONNECTIONS)")
        connection_name = names[0]

    conn = cmldata.get_connection(connection_name)
    df = conn.get_pandas_dataframe("SELECT * FROM default.annotations")

    return _build_category_set_from_records(
        df.to_dict("records"),
        hierarchical=hierarchical,
    )


# ── JSON loader ──────────────────────────────────────────────────────


def load_annotations_from_json(
    path: str | Path,
    *,
    hierarchical: bool = True,
) -> CategorySet:
    """Load annotations from a cached JSON file.

    The JSON file is an array of objects with the same schema as the
    hive annotations table.
    """
    path = Path(path)
    with open(path) as f:
        records = json.load(f)

    return _build_category_set_from_records(records, hierarchical=hierarchical)


# ── Mock/fixture loader ──────────────────────────────────────────────


def load_mock_annotations(*, hierarchical: bool = True) -> CategorySet:
    """Load mock annotations from the fixtures directory.

    Used for devenv/CI testing when hive is not available.
    """
    fixtures_dir = Path(__file__).parent / "fixtures"
    path = fixtures_dir / "mock_annotations.json"
    if not path.exists():
        raise FileNotFoundError(f"Mock annotations fixture not found: {path}")
    return load_annotations_from_json(path, hierarchical=hierarchical)


# ── Shared builder ───────────────────────────────────────────────────


def _build_category_set_from_records(
    records: list[dict],
    *,
    hierarchical: bool = True,
) -> CategorySet:
    """Build a CategorySet from annotation records.

    Expected record keys (from hive default.annotations — 11 columns):
    - id: hierarchical code ("0", "0.1", "1.1.1.1.1.1.1")
    - ontology: human-readable label ("Payment Card Number")
    - annotation: formal code / mnemonic ("PAN", "CVV2", "SALARY")
    - definition: description text
    - common_names: pipe-separated aliases
    - specifics: examples and additional context (optional)
    - non_corp, emp_contractor, individual, corp: sensitivity ratings
    - deprecated: "yes"/"no" filter flag
    """
    all_ids = {str(r.get("id", "")).strip() for r in records}

    def _is_leaf(row_id: str) -> bool:
        prefix = row_id + "."
        return not any(
            other_id.startswith(prefix) and other_id != row_id
            for other_id in all_ids
        )

    leaf_rows: list[dict] = []
    refs: list[ReferenceCategory] = []

    for row in records:
        row_id = str(row.get("id", "")).strip()
        if not row_id or not row_id[0].isdigit():
            continue
        deprecated = str(row.get("deprecated", "")).strip().lower()
        if deprecated == "yes":
            continue
        if not _is_leaf(row_id):
            continue

        ontology = str(row.get("ontology", "")).strip()
        annotation = str(row.get("annotation", "")).strip()
        definition = str(row.get("definition", "")).strip()
        common_names = str(row.get("common_names", "")).strip()
        specifics = str(row.get("specifics", "")).strip()

        # ontology = human label ("Payment Card Number")
        # annotation = formal code / mnemonic ("PAN")
        label = ontology or annotation
        formal_code = annotation

        # Build embedding text (matches signals pattern)
        words_label = re.sub(
            r"[^a-z0-9 ]", "",
            label.lower().replace("/", " ").replace("(", "").replace(")", ""),
        ).strip()
        parts = [words_label, label]
        if formal_code and formal_code != label:
            parts.append(formal_code)
        if definition:
            parts.append(definition)
        if common_names:
            parts.append(common_names)
        if specifics:
            parts.append(specifics[:150])
        embedding_text = " | ".join(parts)

        # Derive parent_code from dot notation
        id_parts = row_id.rsplit(".", 1)
        parent_code = id_parts[0] if len(id_parts) > 1 else None

        # Sensitivity ratings per data subject role
        sensitivity = {}
        for role in ("non_corp", "emp_contractor", "individual", "corp"):
            val = str(row.get(role, "")).strip()
            if val:
                sensitivity[role] = val

        leaf_rows.append({"code": row_id})
        refs.append(ReferenceCategory(
            code=row_id,
            label=label,
            embedding_text=embedding_text,
            abbrev=formal_code,
            common_names=common_names,
            description=definition,
            taxonomy="annotations",
            parent_code=parent_code,
            sensitivity=sensitivity or None,
        ))

    if not hierarchical:
        return CategorySet(name="annotations", categories=refs)

    # Build parent nodes from non-leaf rows in the source data
    parent_refs: list[ReferenceCategory] = []
    for row in records:
        row_id = str(row.get("id", "")).strip()
        if not row_id or not row_id[0].isdigit():
            continue
        deprecated = str(row.get("deprecated", "")).strip().lower()
        if deprecated == "yes":
            continue
        if _is_leaf(row_id):
            continue

        ontology = str(row.get("ontology", "")).strip()
        annotation = str(row.get("annotation", "")).strip()
        label = ontology or annotation or f"Level_{row_id}"
        id_parts = row_id.rsplit(".", 1)
        parent_code = id_parts[0] if len(id_parts) > 1 else None
        parent_refs.append(ReferenceCategory(
            code=row_id,
            label=label,
            embedding_text=label,
            abbrev=annotation,
            taxonomy="annotations",
            parent_code=parent_code,
        ))

    # Synthetic parents for any missing intermediate levels
    synthetic_parents = _build_parents(leaf_rows)
    existing_codes = {r.code for r in refs} | {r.code for r in parent_refs}
    extra_parents = [p for p in synthetic_parents if p.code not in existing_codes]

    all_categories = refs + parent_refs + extra_parents

    return HierarchicalCategorySet(
        name="annotations",
        categories=refs,
        all_categories=all_categories,
    )


def save_annotations_json(
    category_set: CategorySet,
    output_path: str | Path,
) -> Path:
    """Serialize a CategorySet to JSON for caching.

    Writes the full annotation records (not just leaf refs) so the JSON
    can round-trip through load_annotations_from_json().
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    cats = (
        category_set.all_categories
        if isinstance(category_set, HierarchicalCategorySet)
        else category_set.categories
    )
    for cat in cats:
        rec = {
            "id": cat.code,
            "ontology": cat.label,        # human-readable label
            "annotation": cat.abbrev,      # formal code / mnemonic
            "definition": cat.description,
            "common_names": cat.common_names,
            "deprecated": "no",
        }
        if cat.sensitivity:
            rec.update(cat.sensitivity)
        records.append(rec)

    output_path.write_text(json.dumps(records, indent=2) + "\n")
    return output_path
