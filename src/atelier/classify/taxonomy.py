"""Taxonomy management for column type annotation.

Manages the controlled vocabulary as a hierarchical category set with
two-layer composition: a BFO-grounded universal vocabulary (always available,
safe for git) plus optional domain extensions (customer annotations loaded
at runtime via hive or cache).

Supports loading from:
- Universal vocabulary fixture (BFO-grounded, mnemonic codes)
- Hive tables (default.annotations via cml.data_v1)
- Cached JSON (build/data/annotations/annotations.json)
- Mock fixtures (for devenv/CI testing)

Ported from signals/src/sigint/category_set.py — extended with SKOS-informed
notation (identity is mnemonic path, numeric codes are metadata).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path


@dataclass(frozen=True)
class ReferenceCategory:
    """A single category in the controlled vocabulary.

    Attributes:
        code: Mnemonic identity path (e.g., "ICE.SENSITIVE.PID.CONTACT.EMAIL").
              Used for tree navigation, DST focal elements, and Atlas type names.
              For legacy domain records this may still be numeric dot-notation.
        label: Human-readable display name ("Email Address").
        notation: SKOS-style classification code — carries the numeric dot-notation
              as queryable metadata, not structural identity.  May be empty for
              universal vocabulary terms that have no legacy numeric mapping.
        abbrev: Formal short code / mnemonic ("EMAIL", "PAN", "C_NOS").
        taxonomy: Namespace discriminator ("universal", "annotations", domain name).
        parent_code: Explicit parent in the hierarchy.  When present, tree
              construction uses this directly instead of re-deriving from code.
    """

    code: str
    label: str
    embedding_text: str
    abbrev: str = ""
    description: str = ""
    common_names: str = ""
    notation: str = ""          # SKOS-style numeric code (metadata, not identity)
    taxonomy: str = "annotations"
    parent_code: str | None = None
    sensitivity: dict[str, str] | None = None

    @property
    def atlas_type_name(self) -> str:
        safe = self.code.replace(".", "_")
        return f"ATELIER_{safe}"


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


def _build_parents(
    leaf_rows: list[dict],
    all_existing_codes: set[str] | None = None,
) -> list[ReferenceCategory]:
    """Build synthetic parent nodes for any missing intermediate levels.

    Uses dot-separated codes (works for both numeric "1.1.1.1" and mnemonic
    "ICE.SENSITIVE.PID.CONTACT.EMAIL") to infer intermediate parents that
    aren't explicitly provided in the source records.
    """
    existing_codes: set[str] = all_existing_codes or set()
    for row in leaf_rows:
        existing_codes.add(row["code"])

    parents: dict[str, ReferenceCategory] = {}

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
                taxonomy=row.get("taxonomy", "annotations"),
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



# ── Record normalization ─────────────────────────────────────────────


def _normalize_record(row: dict) -> dict:
    """Normalize a hive annotation record for reliable key access.

    Handles four known quirks from Hive/Impala:
    - Table-qualified keys: ``annotations.id`` → ``id``
    - Uppercase keys: ``ID`` → ``id``
    - BOM/quote artifacts: ``'id`` → ``id``
    - Spaces in keys: ``Common Names`` → ``common_names``
    """
    normalized: dict = {}
    for k, v in row.items():
        nk = k.lower().strip().strip("'").replace(" ", "_").rsplit(".", 1)[-1]
        normalized[nk] = v
    return normalized


# ── Shared builder ───────────────────────────────────────────────────


def _build_category_set_from_records(
    records: list[dict],
    *,
    hierarchical: bool = True,
) -> CategorySet:
    """Build a CategorySet from annotation records.

    Accepts two record formats:

    **Universal format** (new — mnemonic codes with explicit parent_code):
    - code: mnemonic path ("ICE.SENSITIVE.PID.CONTACT.EMAIL")
    - label: display name
    - parent_code: explicit parent ("ICE.SENSITIVE.PID.CONTACT")
    - notation: SKOS numeric code (optional, metadata only)
    - taxonomy: namespace discriminator ("universal", "annotations", etc.)
    - abbrev, description, common_names, sensitivity fields (optional)

    **Hive format** (legacy — numeric dot-codes, parent derived from code):
    - id: hierarchical code ("0", "0.1", "1.1.1.1.1.1.1")
    - ontology: human-readable label
    - annotation: formal code / mnemonic
    - definition, common_names, specifics, sensitivity, deprecated
    """
    records = [_normalize_record(r) for r in records]

    # Accept both "code" and "id" as the identity field
    def _get_code(row: dict) -> str:
        return str(row.get("code", "") or row.get("id", "")).strip()

    all_codes = {_get_code(r) for r in records}

    # Determine if records have explicit parent_code fields
    has_explicit_parents = any(r.get("parent_code") is not None for r in records)

    # Build set of codes that are referenced as parents (for leaf detection)
    codes_with_children: set[str] = set()
    if has_explicit_parents:
        for r in records:
            pc = r.get("parent_code")
            if pc is not None:
                codes_with_children.add(str(pc).strip())
    else:
        # Legacy: use prefix matching for numeric dot-notation
        for code in all_codes:
            prefix = code + "."
            for other in all_codes:
                if other.startswith(prefix) and other != code:
                    codes_with_children.add(code)
                    break

    def _is_leaf(code: str) -> bool:
        return code not in codes_with_children

    def _derive_parent(code: str, row: dict) -> str | None:
        """Get parent_code: explicit from record, or derived from dot-notation."""
        explicit = row.get("parent_code")
        if explicit is not None:
            val = str(explicit).strip()
            return val if val else None
        # Fall back to dot-notation derivation
        parts = code.rsplit(".", 1)
        return parts[0] if len(parts) > 1 else None

    leaf_rows: list[dict] = []
    refs: list[ReferenceCategory] = []

    for row in records:
        row_code = _get_code(row)
        if not row_code:
            continue
        # Legacy filter: skip rows without digit-starting code in hive format
        if not has_explicit_parents and not row_code[0].isdigit():
            continue
        if not _is_leaf(row_code):
            continue

        # Read fields (support both formats)
        ontology = str(row.get("ontology", "") or row.get("label", "")).strip()
        annotation = str(row.get("annotation", "") or row.get("abbrev", "")).strip()
        definition = str(row.get("definition", "") or row.get("description", "")).strip()
        common_names = str(row.get("common_names", "")).strip()
        specifics = str(row.get("specifics", "")).strip()
        notation = str(row.get("notation", "")).strip()
        taxonomy = str(row.get("taxonomy", "annotations")).strip()

        label = ontology or annotation
        formal_code = annotation

        # Build embedding text
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

        parent_code = _derive_parent(row_code, row)

        # Sensitivity ratings per data subject role
        # Handles both flat keys (from hive) and nested dict (from saved JSON)
        sensitivity = {}
        nested = row.get("sensitivity")
        if isinstance(nested, dict):
            sensitivity = nested
        else:
            for role in ("non_corp", "emp_contractor", "individual", "corp"):
                val = str(row.get(role, "")).strip()
                if val:
                    sensitivity[role] = val

        leaf_rows.append({"code": row_code, "taxonomy": taxonomy})
        refs.append(ReferenceCategory(
            code=row_code,
            label=label,
            embedding_text=embedding_text,
            abbrev=formal_code,
            common_names=common_names,
            description=definition,
            notation=notation,
            taxonomy=taxonomy,
            parent_code=parent_code,
            sensitivity=sensitivity or None,
        ))

    if not hierarchical:
        return CategorySet(name="annotations", categories=refs)

    # Build parent nodes from non-leaf rows in the source data
    parent_refs: list[ReferenceCategory] = []
    for row in records:
        row_code = _get_code(row)
        if not row_code:
            continue
        if not has_explicit_parents and not row_code[0].isdigit():
            continue
        if _is_leaf(row_code):
            continue

        ontology = str(row.get("ontology", "") or row.get("label", "")).strip()
        annotation = str(row.get("annotation", "") or row.get("abbrev", "")).strip()
        notation = str(row.get("notation", "")).strip()
        taxonomy = str(row.get("taxonomy", "annotations")).strip()
        label = ontology or annotation or f"Level_{row_code}"
        parent_code = _derive_parent(row_code, row)
        parent_refs.append(ReferenceCategory(
            code=row_code,
            label=label,
            embedding_text=label,
            abbrev=annotation,
            notation=notation,
            taxonomy=taxonomy,
            parent_code=parent_code,
        ))

    # Synthetic parents for any missing intermediate levels
    existing_codes = {r.code for r in refs} | {r.code for r in parent_refs}
    synthetic_parents = _build_parents(leaf_rows, all_existing_codes=existing_codes)
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
            "code": cat.code,
            "label": cat.label,
            "abbrev": cat.abbrev,
            "description": cat.description,
            "common_names": cat.common_names,
            "notation": cat.notation,
            "taxonomy": cat.taxonomy,
            "parent_code": cat.parent_code,
        }
        if cat.sensitivity:
            rec["sensitivity"] = cat.sensitivity
        records.append(rec)

    output_path.write_text(json.dumps(records, indent=2) + "\n")
    return output_path


# ── Universal vocabulary ────────────────────────────────────────────


def load_universal_vocabulary(*, hierarchical: bool = True) -> CategorySet:
    """Load the BFO-grounded universal vocabulary from fixtures.

    The universal vocabulary is always available (shipped in git) and contains
    only well-known data governance categories grounded in BFO's Information
    Content Entity.  It serves as the base layer for domain composition.
    """
    fixtures_dir = Path(__file__).parent / "fixtures"
    path = fixtures_dir / "universal_vocabulary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Universal vocabulary fixture not found: {path}. "
            "This file is required and should be checked into git."
        )
    return load_annotations_from_json(path, hierarchical=hierarchical)


def compose_vocabularies(
    base: HierarchicalCategorySet,
    domain: list[dict] | CategorySet,
) -> HierarchicalCategorySet:
    """Compose a universal base vocabulary with domain extensions.

    Domain records declare ``parent_code`` referencing universal codes,
    attaching as new leaves (or subtrees) to the universal hierarchy.

    Args:
        base: Universal BFO-grounded vocabulary.
        domain: Either raw records (``list[dict]``) from a customer
            annotations table, or a pre-built ``CategorySet``.

    Returns:
        Merged HierarchicalCategorySet with both layers.
    """
    if isinstance(domain, list):
        domain_cs = _build_category_set_from_records(
            domain, hierarchical=True,
        )
    else:
        domain_cs = domain

    # Merge: domain categories extend base, dedup on code
    base_codes = {c.code for c in base.all_categories}

    merged_leaves = list(base.categories)
    for cat in domain_cs.categories:
        if cat.code not in base_codes:
            merged_leaves.append(cat)

    merged_all = list(base.all_categories)
    for cat in (
        domain_cs.all_categories
        if isinstance(domain_cs, HierarchicalCategorySet)
        else domain_cs.categories
    ):
        if cat.code not in base_codes:
            merged_all.append(cat)

    return HierarchicalCategorySet(
        name="composed",
        categories=merged_leaves,
        all_categories=merged_all,
    )
