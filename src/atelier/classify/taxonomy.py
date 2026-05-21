# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

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
from dataclasses import dataclass, replace
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
        abbrev: Formal short code / mnemonic for the term ("EMAIL",
              "PAN", "TXNAMT", "BAN", "SSN", "C_FD", "A_PHN").  Both
              terminal and parent nodes carry mnemonics in shipped
              Atelier vocabularies — see
              ``src/atelier/classify/fixtures/PROVENANCE.md``.
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

    Every node in the taxonomy is a valid tagging target.  ``categories``
    is the full vocabulary; the legacy filter that restricted tagging
    to terminal nodes was a long-standing mistake and has been removed.
    A column may legitimately be classified at any depth (e.g.
    ``1.1.1.1 Financial Data`` is a correct classification for a
    mixed-content monetary column where no more-specific code captures
    the distribution).

    ``all_categories`` is preserved as an alias of ``categories`` for
    callers that still reference the old name; new code should prefer
    ``categories``.  The ``leaf_codes`` property and the ``_is_leaf``
    constructor helper are retained for internal tree-traversal /
    Atlas-type-graph mechanics — they describe hierarchy *structure*
    and do not gate which codes the pipeline can predict.
    """

    def __init__(
        self,
        name: str,
        categories: list[ReferenceCategory],
        all_categories: list[ReferenceCategory] | None = None,
    ) -> None:
        # ``all_categories`` is retained as a constructor keyword for
        # backward compatibility with callers that still supply both
        # arguments.  When provided it wins because it represents the
        # full taxonomy; otherwise ``categories`` is taken as the
        # complete vocabulary.  No filtering happens at construction.
        unified = all_categories if all_categories is not None else categories
        super().__init__(name=name, categories=unified)
        self.all_categories = unified  # alias — same object, not a copy

    @cached_property
    def all_by_code(self) -> dict[str, ReferenceCategory]:
        return {c.code: c for c in self.all_categories}

    @cached_property
    def all_by_abbrev(self) -> dict[str, ReferenceCategory]:
        # Mnemonic → category over the full tree.  ``by_abbrev``
        # (inherited from CategorySet) is now equivalent since
        # ``categories`` covers the full taxonomy; this property is
        # retained for API stability and is what
        # ``_resolve_to_focal_element`` consults when the LLM emits an
        # annotation string instead of a numeric dot-code.
        return {c.abbrev: c for c in self.all_categories if c.abbrev}

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
        """Codes with no children in the projected hierarchy.

        A *structural* accessor exposing the set of terminal nodes for
        tree traversal, Atlas-type-graph generation, and similar
        hierarchy-mechanics callers.  It does NOT define the tagging
        vocabulary — every code in ``categories`` (parent or terminal)
        is independently a valid prediction.  Callers that conflate
        this set with "the set of predictable codes" are perpetuating
        the leaf-only mistake the refactor is removing.
        """
        parents_with_children = set(self.children.keys())
        return frozenset(
            c.code for c in self.all_categories
            if c.code not in parents_with_children
        )

    def descendants(self, code: str) -> frozenset[str]:
        """All descendants of *code* in the projected hierarchy.

        Returns every code that has *code* as a (transitive) ancestor —
        both terminal and intermediate nodes.  Does NOT include
        *code* itself; callers that want self-inclusion construct
        ``{code} | cs.descendants(code)`` explicitly.

        Used as the subtree-cover for internal-node focal elements
        (DST), for hierarchical-accuracy scoring, and for any consumer
        that needs "everything below this point in the tree."  Under
        the unified tagging-vocabulary semantics, every returned code
        is independently a valid prediction.
        """
        result: set[str] = set()
        stack = [code]
        while stack:
            current = stack.pop()
            for child in self.children.get(current, []):
                result.add(child)
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

    def path_from_root(self, code: str) -> list[str]:
        """Path from root to *code* inclusive (``[root, ..., parent, code]``)."""
        return list(reversed(self.ancestors(code))) + [code]

    def compute_nhsvm_alphas(self) -> dict[str, float]:
        """NHSVM normalization weights with directional constraint (Choi et al. 2015, Eq. 7).

        Solves the LP: max min_n alpha_n, subject to path-sum = 1 for
        every leaf, alpha_child >= alpha_parent for all parent-child
        pairs, alpha_n >= 0.  The directional constraint forces weight
        toward leaves, preventing shallow nodes from absorbing the
        budget on unbalanced trees.

        Falls back to the unconstrained closed-form on solver failure.
        """
        try:
            return self._compute_nhsvm_alphas_constrained()
        except Exception:
            import logging
            logging.getLogger(__name__).warning(
                "NHSVM LP solver failed; falling back to unconstrained alphas"
            )
            return self._compute_nhsvm_alphas_unconstrained()

    def _compute_nhsvm_alphas_constrained(self) -> dict[str, float]:
        """Directional-constraint LP (Eq. 7, Choi et al. 2015)."""
        import numpy as np
        from scipy.optimize import linprog

        all_cats = list(self.all_categories)
        code_to_idx = {cat.code: i for i, cat in enumerate(all_cats)}
        M = len(all_cats)

        parents_with_children = set(self.children.keys())
        leaf_codes = [c.code for c in all_cats if c.code not in parents_with_children]

        # Variables: alpha_0..alpha_{M-1}, t  (maximize t = min alpha)
        c_obj = np.zeros(M + 1)
        c_obj[M] = -1.0

        # Equality: path sums = 1 for each leaf
        A_eq = np.zeros((len(leaf_codes), M + 1))
        b_eq = np.ones(len(leaf_codes))
        for row_i, leaf in enumerate(leaf_codes):
            for node in self.path_from_root(leaf):
                if node in code_to_idx:
                    A_eq[row_i, code_to_idx[node]] = 1.0

        # Inequality (Ax <= b): directional + t <= alpha_n
        ub_rows: list[np.ndarray] = []

        for cat in all_cats:
            parent_code = self.parent.get(cat.code)
            if parent_code is not None and parent_code in code_to_idx:
                row = np.zeros(M + 1)
                row[code_to_idx[parent_code]] = 1.0
                row[code_to_idx[cat.code]] = -1.0
                ub_rows.append(row)

        for i in range(M):
            row = np.zeros(M + 1)
            row[i] = -1.0
            row[M] = 1.0
            ub_rows.append(row)

        A_ub = np.array(ub_rows)
        b_ub = np.zeros(len(ub_rows))
        bounds = [(0, None)] * (M + 1)

        result = linprog(
            c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs",
        )
        if result.status != 0:
            raise RuntimeError(f"NHSVM LP failed: {result.message}")

        return {all_cats[i].code: float(result.x[i]) for i in range(M)}

    def _compute_nhsvm_alphas_unconstrained(self) -> dict[str, float]:
        """Unconstrained closed-form (Lemma 2): ``alpha_n = 1/D_n - 1/D_parent``."""
        def _max_subtree_depth(node: str) -> int:
            ch = self.children.get(node, [])
            if not ch:
                return 1
            return 1 + max(_max_subtree_depth(c) for c in ch)

        depths: dict[str, int] = {}
        for cat in self.all_categories:
            depths[cat.code] = _max_subtree_depth(cat.code)

        alphas: dict[str, float] = {}
        for cat in self.all_categories:
            parent_code = self.parent.get(cat.code)
            if parent_code is None:
                alphas[cat.code] = 1.0 / depths[cat.code]
            else:
                alphas[cat.code] = 1.0 / depths[cat.code] - 1.0 / depths[parent_code]
        return alphas

    def atlas_type_graph(self) -> list[dict]:
        """Export as Apache Atlas Classification type definitions.

        Each category becomes an Atlas type; parent-child relationships
        in the projected taxonomy become Atlas ``superTypes`` chains.
        Each type carries the full mnemonic code path, human label,
        and SKOS notation as attributes.

        The ``entityTypes`` field is currently gated on whether a node
        has children in the projected tree, restricting *direct*
        application to terminal categories.  That gate is a vestige
        of the leaf-only era — every category in the tagging
        vocabulary is independently a valid tag at the Atelier layer,
        and Atlas-side enablement of parent application is tracked
        for Phase 3.

        Returns:
            List of Atlas-compatible type definition dicts.
        """
        types: list[dict] = []
        for cat in self.all_categories:
            parent = self.all_by_code.get(cat.parent_code or "")
            super_types = [parent.atlas_type_name] if parent else []
            # ``is_leaf`` here is a structural Atlas-export concern,
            # not an Atelier tagging-vocabulary filter.
            is_leaf = cat.code in self.leaf_codes

            type_def = {
                "name": cat.atlas_type_name,
                "description": cat.description or cat.label,
                "superTypes": super_types,
                "attributeDefs": [
                    {"name": "ice_code", "typeName": "string",
                     "isOptional": False, "defaultValue": cat.code},
                    {"name": "notation", "typeName": "string",
                     "isOptional": True, "defaultValue": cat.notation},
                ],
                "entityTypes": ["DataSet", "Column"] if is_leaf else [],
            }
            types.append(type_def)
        return types


# ── Helpers ──────────────────────────────────────────────────────────


def _nearest_projected_ancestor(code: str, projected_codes: set[str]) -> str | None:
    """Walk dot-notation prefixes up from ``code`` to the nearest projected ancestor.

    The customer's annotations are a *projection* of a larger ontology
    DAG, so a category's structural parent (immediate dot-prefix) may
    not itself be projected.  This helper returns the nearest
    *projected* ancestor — or ``None`` when the chain hits the top of
    the DAG without finding one, in which case ``code`` is a forest
    root in the projection.
    """
    if "." not in code:
        return None
    cur = code.rsplit(".", 1)[0]
    while True:
        if cur in projected_codes:
            return cur
        if "." not in cur:
            return None
        cur = cur.rsplit(".", 1)[0]


# ── Hive loader ──────────────────────────────────────────────────────


_HIVE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_annotations_from_hive(
    cfg,
    connection_name: str | None = None,
    database: str = "default",
    *,
    hierarchical: bool = True,
) -> CategorySet:
    """Load annotations from ``{database}.annotations`` via CAI Data Platform.

    Args:
        cfg: AtelierConfig with data connection settings.
        connection_name: CAI connection name. Falls back to first configured.
        database: Hive database holding the ``annotations`` table.
            Defaults to ``"default"`` to preserve the pre-parameterization
            behavior; operators can target any database by passing a
            whitelisted identifier (alnum + underscore, leading letter).
        hierarchical: Return HierarchicalCategorySet if True.

    Returns:
        CategorySet loaded from the hive annotations table.

    Raises:
        ValueError: when ``database`` contains characters that can't be
            safely interpolated into the SQL (SQL-injection guard — we
            whitelist rather than quote because Hive dialect handling of
            backtick-quoted identifiers varies across connectors).
    """
    # User-input sanity check first — rejects bad identifiers without
    # needing cml.data_v1 (so we catch misconfigured env vars even on
    # dev machines without the Cloudera runtime).
    if not _HIVE_IDENT_RE.match(database or ""):
        raise ValueError(
            f"refusing to load annotations from unsafe database identifier "
            f"{database!r}; expected alnum + underscore (leading letter)."
        )

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
    df = conn.get_pandas_dataframe(f"SELECT * FROM {database}.annotations")

    return _build_category_set_from_records(
        df.to_dict("records"),
        hierarchical=hierarchical,
    )


# ── JSON loader ──────────────────────────────────────────────────────


_SYNTHETIC_PARENT_LABEL_RE = re.compile(r"^Level\d+_")


def load_annotations_from_json(
    path: str | Path,
    *,
    hierarchical: bool = True,
) -> CategorySet:
    """Load annotations from a cached JSON file.

    The JSON file is an array of objects with the same schema as the
    hive annotations table.

    One-shot self-purge: caches written by older loader versions can
    contain fabricated parent records (label matches ``Level{N}_X``).
    The customer's projection is a slice of the ontology DAG, so those
    records aren't real terms — drop them at read time so the runtime
    taxonomy doesn't carry placeholders into DST, the picker, or the
    operator-visible classification output.  The next save will write
    a clean cache without them.
    """
    import logging
    log = logging.getLogger(__name__)

    path = Path(path)
    with open(path) as f:
        records = json.load(f)

    poisoned = [
        r for r in records
        if _SYNTHETIC_PARENT_LABEL_RE.match(str(r.get("label") or ""))
    ]
    if poisoned:
        log.warning(
            "Filtered %d synthetic-parent placeholder(s) from cache %s "
            "(codes=%s) — these were fabricated by an older loader; "
            "next save will rewrite the cache without them.",
            len(poisoned), path, [r.get("code") for r in poisoned],
        )
        records = [
            r for r in records
            if not _SYNTHETIC_PARENT_LABEL_RE.match(str(r.get("label") or ""))
        ]

    return _build_category_set_from_records(records, hierarchical=hierarchical)


def load_annotations_from_filesystem(
    path: str | Path,
    *,
    hierarchical: bool = True,
    taxonomy: str = "filesystem",
) -> CategorySet:
    """Load annotations from a local ``annotations.csv`` file.

    Accepts either the CSV path directly or a directory containing an
    ``annotations.csv`` entry — mirrors the Hive flow where the source
    is a database whose ``<db>.annotations`` table supplies the vocab.

    Row-shape expectations match the first-column-is-``'ID`` layout the
    UAT reference uses (leading apostrophe preserved from Excel export)
    as well as plain ``ID`` / ``id``.  Ontology labels, annotation
    mnemonics, definitions, common-names, specifics, and the four
    sensitivity flags (NON_CORP / EMP, CONTRACTOR / INDIVIDUAL / CORP)
    map to the same keys ``_build_category_set_from_records`` already
    understands, so category processing, parent derivation, and
    embedding-text composition behave identically between Hive- and
    filesystem-sourced vocabularies.

    The ``taxonomy`` argument stamps the namespace discriminator on
    each loaded category so blended vocabularies can tell sources
    apart; defaults to ``"filesystem"`` for generality but callers
    that maintain a branded namespace (e.g. ``"meta-tagging"``) should
    pass their own.
    """
    import csv

    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "annotations.csv"
    if not path.is_file():
        raise FileNotFoundError(f"annotations.csv not found: {path}")

    records: list[dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = _normalize_annotations_row(row)
            # Tolerate the leading-apostrophe quirk from Excel exports
            # ("'ID") as well as plain "ID"/"id".
            code = (row.get("'ID") or row.get("ID") or row.get("id") or "").strip()
            if not code:
                continue
            records.append({
                "id": code,
                "ontology": (row.get("Ontology") or row.get("ontology") or "").strip(),
                "annotation": (row.get("Annotation") or row.get("annotation") or "").strip(),
                "definition": (row.get("Definition") or row.get("definition") or "").strip(),
                "common_names": (
                    row.get("Common Names") or row.get("common_names") or ""
                ).strip(),
                "specifics": (
                    row.get("Specifics, Examples and/or Additional Context")
                    or row.get("specifics")
                    or ""
                ).strip(),
                "deprecated": (row.get("Deprecated") or row.get("deprecated") or "").strip().lower(),
                "non_corp": (row.get("NON_CORP") or row.get("non_corp") or "").strip(),
                "emp_contractor": (
                    row.get("EMP, CONTRACTOR") or row.get("emp_contractor") or ""
                ).strip(),
                "individual": (row.get("INDIVIDUAL") or row.get("individual") or "").strip(),
                "corp": (row.get("CORP") or row.get("corp") or "").strip(),
                "taxonomy": taxonomy,
            })
    return _build_category_set_from_records(records, hierarchical=hierarchical)


def _normalize_annotations_row(row: dict) -> dict:
    """Strip a ``<table>.`` prefix common to Hive CSV exports.

    The Gopala-vintage Excel export of ``annotations.csv`` used plain
    column names (``ID``, ``Ontology``, …).  The UAT Hive export, by
    contrast, prefixes every column with the table name — ``annotations.id``,
    ``annotations.ontology``, and so on.  We detect a uniform dotted
    prefix and strip it so the downstream ``row.get(…)`` fallbacks match
    either format.  When no uniform prefix is present the row is
    returned unchanged.
    """
    keys = [k for k in row.keys() if k]
    if not keys:
        return row
    prefixes = {k.split(".", 1)[0] for k in keys if "." in k}
    if len(prefixes) != 1:
        return row
    prefix = next(iter(prefixes)) + "."
    # Only strip when *every* key starts with that prefix (avoids
    # accidental stripping when a single column legitimately contains
    # a dot).
    if not all(k.startswith(prefix) for k in keys):
        return row
    return {k[len(prefix):]: v for k, v in row.items()}


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


def _repair_csv_oversplit(row: dict) -> dict:
    """Reassemble a CSV-quoted description that was split across columns.

    Some customer ``annotations`` tables were ingested via a CSV path
    that did not honor quoted fields, so a single ``"a, b, c, d"``
    description got distributed across N adjacent columns: the first
    fragment retains the leading ``"``, the last retains the trailing
    ``"``, and the inner fragments are bare commas-removed slices.

    This boundary-level normalization detects the pattern (description
    starts with ``"``, a later column ends with ``"``) and stitches the
    fragments back together, blanking the consumed columns.  Bounded
    look-ahead (``MAX_LOOKAHEAD``) prevents runaway consumption when
    the closing quote was lost entirely.

    Idempotent: rows without leading-quote pass through unchanged.
    """
    MAX_LOOKAHEAD = 10

    desc_key = "definition" if "definition" in row else (
        "description" if "description" in row else None
    )
    if not desc_key:
        return row
    desc = str(row.get(desc_key) or "").strip()
    if not desc.startswith('"'):
        return row

    keys = list(row.keys())
    try:
        start = keys.index(desc_key)
    except ValueError:
        return row
    candidate_keys = keys[start + 1 : start + 1 + MAX_LOOKAHEAD]

    closer_idx = -1
    for i, k in enumerate(candidate_keys):
        v = row.get(k)
        if isinstance(v, str) and v.strip().endswith('"'):
            closer_idx = i
            break

    out = dict(row)
    if closer_idx < 0:
        # No closer in sight — strip the orphan leading quote and stop.
        out[desc_key] = desc.lstrip('"').strip()
        return out

    fragments: list[str] = [desc.lstrip('"').strip()]
    for k in candidate_keys[: closer_idx + 1]:
        v = str(row.get(k) or "").strip()
        if not v:
            continue
        if v.endswith('"'):
            v = v.rstrip('"').strip()
        fragments.append(v)

    out[desc_key] = ", ".join(p for p in fragments if p)
    for k in candidate_keys[: closer_idx + 1]:
        out[k] = ""
    return out


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
    records = [_repair_csv_oversplit(_normalize_record(r)) for r in records]

    # Accept both "code" and "id" as the identity field
    def _get_code(row: dict) -> str:
        return str(row.get("code", "") or row.get("id", "")).strip()

    all_codes = {_get_code(r) for r in records}

    # Determine if records have explicit parent_code fields
    has_explicit_parents = any(r.get("parent_code") is not None for r in records)

    # Build set of codes that are referenced as parents.  Used only
    # for hierarchy *structure* (parent-derivation and the internal
    # ``_is_leaf`` helper below) — never to filter the tagging
    # vocabulary, since every category is independently predictable.
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
        # Structural predicate — "this node has no children in the
        # projected hierarchy."  Used internally to populate the
        # ``leaf_codes`` accessor for tree-traversal callers; does NOT
        # gate which codes the pipeline can predict.
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

    def _build_ref(row: dict, code: str, parent_code: str | None) -> ReferenceCategory:
        """Construct a fully-populated ReferenceCategory from a raw record.

        Applied uniformly to every record so each category carries the
        same description / common_names / specifics / sensitivity
        regardless of its position in the tree.  The customer's
        projection is a slice of the ontology DAG; every projected
        term is a first-class operator-visible category and
        independently a valid tagging target.
        """
        def _clean(v) -> str:
            # Strip leading/trailing orphan quote chars left over from
            # malformed CSV ingestion that ``_repair_csv_oversplit`` did
            # not stitch back together (no closer found within the
            # look-ahead window).
            return str(v or "").strip().strip('"').strip()

        ontology = _clean(row.get("ontology") or row.get("label"))
        annotation = _clean(row.get("annotation") or row.get("abbrev"))
        definition = _clean(row.get("definition") or row.get("description"))
        common_names = _clean(row.get("common_names"))
        specifics = _clean(row.get("specifics"))
        notation = _clean(row.get("notation"))
        taxonomy = _clean(row.get("taxonomy")) or "annotations"

        label = ontology or annotation
        formal_code = annotation

        # Uniform rich embedding_text for every category — e.g.
        # ``Financial Data`` (a parent node) gets the full label +
        # abbrev + description text it has in Hive, not just the bare
        # label.  Fixes a historical bias where cosine pulled toward
        # categories that happened to carry richer metadata than their
        # neighbors; under the current design every category in the
        # tagging vocabulary is metadata-complete.
        words_label = re.sub(
            r"[^a-z0-9 ]", "",
            label.lower().replace("/", " ").replace("(", "").replace(")", ""),
        ).strip()
        text_parts = [words_label, label]
        if formal_code and formal_code != label:
            text_parts.append(formal_code)
        if definition:
            text_parts.append(definition)
        if common_names:
            text_parts.append(common_names)
        if specifics:
            text_parts.append(specifics[:150])
        embedding_text = " | ".join(p for p in text_parts if p)

        sensitivity: dict[str, str] = {}
        nested = row.get("sensitivity")
        if isinstance(nested, dict):
            sensitivity = nested
        else:
            for role in ("non_corp", "emp_contractor", "individual", "corp"):
                val = str(row.get(role, "")).strip()
                if val:
                    sensitivity[role] = val

        return ReferenceCategory(
            code=code,
            label=label,
            embedding_text=embedding_text,
            abbrev=formal_code,
            common_names=common_names,
            description=definition,
            notation=notation,
            taxonomy=taxonomy,
            parent_code=parent_code,
            sensitivity=sensitivity or None,
        )

    refs: list[ReferenceCategory] = []
    parent_refs: list[ReferenceCategory] = []

    for row in records:
        row_code = _get_code(row)
        if not row_code:
            continue
        # Legacy filter: skip rows without digit-starting code in hive format
        if not has_explicit_parents and not row_code[0].isdigit():
            continue

        parent_code = _derive_parent(row_code, row)
        ref = _build_ref(row, row_code, parent_code)
        # Bucket structurally for the hierarchical/flat construction
        # paths below.  Note: this split is bookkeeping only — both
        # ``refs`` and ``parent_refs`` end up in the unified
        # ``categories`` list under the hierarchical path.
        if _is_leaf(row_code):
            refs.append(ref)
        else:
            parent_refs.append(ref)

    if not hierarchical:
        # Flat construction returns the full tagging vocabulary —
        # terminal and parent categories alike are first-class
        # tagging targets.  (Phase 3 of the leaf-only-removal
        # refactor corrected the historical leaves-only behavior
        # here.)
        return CategorySet(name="annotations", categories=refs + parent_refs)

    # Forest semantics — sanitize parent_code links so that any reference
    # to a code outside the projection (the gaps that are inevitable in a
    # DAG projection of an ontology) snaps to the nearest *projected*
    # ancestor.  No synthetic parents are invented — codes whose chain
    # has no projected ancestor become forest roots (parent_code=None).
    projected_codes = {r.code for r in refs} | {r.code for r in parent_refs}

    def _sanitize(items: list[ReferenceCategory]) -> list[ReferenceCategory]:
        out: list[ReferenceCategory] = []
        for r in items:
            pc = r.parent_code
            if pc is not None and pc not in projected_codes:
                pc = _nearest_projected_ancestor(r.code, projected_codes)
            if pc != r.parent_code:
                out.append(replace(r, parent_code=pc))
            else:
                out.append(r)
        return out

    refs = _sanitize(refs)
    parent_refs = _sanitize(parent_refs)
    # Single tagging vocabulary — every category is a first-class
    # prediction target regardless of its tree position.  The internal
    # split between ``refs`` and ``parent_refs`` is retained only so
    # the ``leaf_codes`` accessor (structural / tree-traversal /
    # Atlas-type-graph use cases) can still report which nodes happen
    # to have no children.
    all_categories = refs + parent_refs

    return HierarchicalCategorySet(
        name="annotations",
        categories=all_categories,
    )


def save_annotations_json(
    category_set: CategorySet,
    output_path: str | Path,
) -> Path:
    """Serialize a CategorySet to JSON for caching.

    Writes every category record so the JSON round-trips through
    ``load_annotations_from_json`` with the full tagging vocabulary
    intact.
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


# ── Vocabulary validation ───────────────────────────────────────────
#
# Detects post-normalization label collisions and other vocabulary-data
# quality issues that classifier-side logic cannot work around.  Common
# example: two distinct codes whose labels differ only by whitespace
# ("Web Browser" vs "WebBrowser") both normalize to ``web_browser`` and
# match the same column-name lookups, so the name-match evidence picks
# one arbitrarily based on iteration order.  ``validate_taxonomy`` is
# the load-time check that surfaces the issue with a structured
# finding rather than letting it manifest as inscrutable
# misclassifications downstream.


@dataclass(frozen=True)
class TaxonomyFinding:
    """A single validation finding from ``validate_taxonomy``.

    Attributes:
        kind: One of ``"label_collision"``, ``"duplicate_code"``,
            ``"orphaned_alias"``.
        severity: ``"warning"`` or ``"error"``.  All findings are
            warnings under default settings; operators who want to
            fail-fast on any finding can flip
            ``classify.taxonomy.strict_validation = true``.
        codes: The category codes involved in the finding.
        detail: Human-readable explanation suitable for log lines.
    """

    kind: str
    severity: str
    codes: tuple[str, ...]
    detail: str


def _normalize_label(label: str) -> str:
    """Match the column-name normalization used elsewhere in the package.

    Mirrors ``meta_tagging_source._normalize`` so the validator detects
    the same collisions the runtime name-match path would conflate.
    Preserves underscores as separators (so ``Web Browser`` →
    ``web_browser``).
    """
    s = re.sub(r"[^a-z0-9_]", "_", label.lower().strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _canonical_label(label: str) -> str:
    """Aggressive canonicalization: strip ALL non-alphanumerics including
    underscores.  Catches the visual-collision class where labels look
    like alternative renderings of the same term (e.g. ``Web Browser``
    vs ``WebBrowser`` both collapse to ``webbrowser``) — these confuse
    LLMs that don't know the canonical form when emitting predictions,
    even if the strict column-name normalization keeps them distinct.
    """
    return re.sub(r"[^a-z0-9]", "", label.lower().strip())


def validate_taxonomy(
    category_set: CategorySet,
    frame=None,  # FrameOfDiscernment | None — typed via duck-typing to avoid the import cycle
) -> list[TaxonomyFinding]:
    """Return a list of vocabulary-quality findings (empty when clean).

    Detects:

    - **label_collision** — two or more categories whose labels
      normalize to the same string (e.g. "Web Browser" / "WebBrowser"
      → both ``web_browser``).  These trigger non-deterministic name-
      match resolution and should be reconciled in the source data.
    - **duplicate_code** — two categories sharing the same ``code``
      (should never occur; defensive check).
    - **orphaned_alias** — a label or alias that, after normalization,
      collides with another category's normalized label even though
      the source labels were distinct.  Subset of ``label_collision``;
      reported separately for clarity.
    - **abbrev_unreachable_in_frame** *(R8 — only when ``frame`` is
      supplied)* — an abbrev present in ``all_by_abbrev`` whose code
      cannot be resolved by ``frame.resolve_annotation`` even after
      the R7 walk-down/walk-up chain.  Surfaces genuine projection
      gaps where the LLM may emit a code the active frame has no
      path to.

    Returns a list (possibly empty) rather than raising.  Callers
    decide whether to log warnings, raise on findings, or aggregate
    for a vocabulary-cleanup report.

    The validator is read-only — it does not mutate the input.
    """
    findings: list[TaxonomyFinding] = []

    cats = list(category_set.categories)
    # On a HierarchicalCategorySet, prefer ``all_categories`` so internal
    # nodes (which can also collide) are checked.
    all_attr = getattr(category_set, "all_categories", None)
    if all_attr is not None:
        cats = list(all_attr)

    # Duplicate codes
    by_code: dict[str, list[ReferenceCategory]] = {}
    for c in cats:
        by_code.setdefault(c.code, []).append(c)
    for code, group in by_code.items():
        if len(group) > 1:
            findings.append(TaxonomyFinding(
                kind="duplicate_code",
                severity="error",
                codes=tuple(c.code for c in group),
                detail=(
                    f"code {code!r} appears {len(group)} times in the "
                    f"category set; codes must be unique"
                ),
            ))

    # Label collisions — detect under both normalizations.
    # Strict (with underscores): catches column-name lookup collisions
    # like the runtime name-match path would conflate.
    # Canonical (alphanumeric-only): catches visual collisions like
    # "Web Browser" / "WebBrowser" that confuse LLM predictions even
    # when strict normalization keeps them distinct.
    seen_collisions: set[tuple[str, ...]] = set()
    for normalizer, kind_label in (
        (_normalize_label, "strict"),
        (_canonical_label, "canonical"),
    ):
        by_norm: dict[str, list[ReferenceCategory]] = {}
        for c in cats:
            if not c.label:
                continue
            key = normalizer(c.label)
            if not key:
                continue
            by_norm.setdefault(key, []).append(c)
        for norm_label, group in by_norm.items():
            if len(group) <= 1:
                continue
            distinct_labels = {c.label for c in group}
            if len(distinct_labels) <= 1:
                continue
            sorted_codes = tuple(sorted(c.code for c in group))
            # Don't double-report when the strict pass already caught it
            # under the same code-set.
            if sorted_codes in seen_collisions:
                continue
            seen_collisions.add(sorted_codes)
            findings.append(TaxonomyFinding(
                kind="label_collision",
                severity="warning",
                codes=sorted_codes,
                detail=(
                    f"labels {sorted(distinct_labels)!r} all collapse to "
                    f"{norm_label!r} under {kind_label} normalization; "
                    f"column-name matching cannot reliably disambiguate "
                    f"between codes {list(sorted_codes)!r}"
                ),
            ))

    # Orphaned aliases — any common_names or abbrev that, after
    # normalization, collides with a different category's normalized
    # label.  Surfaces "alias X is shadowed by category Y" cases where
    # a lookup for X would never reach the category that declares it.
    label_keys: dict[str, ReferenceCategory] = {}
    for c in cats:
        if c.label:
            label_keys[_normalize_label(c.label)] = c
    for c in cats:
        aliases = []
        if c.abbrev:
            aliases.append(c.abbrev)
        if c.common_names:
            aliases.extend(
                a.strip() for a in re.split(r"[,|]", c.common_names) if a.strip()
            )
        for alias in aliases:
            key = _normalize_label(alias)
            if not key:
                continue
            other = label_keys.get(key)
            if other is not None and other.code != c.code:
                findings.append(TaxonomyFinding(
                    kind="orphaned_alias",
                    severity="warning",
                    codes=(c.code, other.code),
                    detail=(
                        f"alias {alias!r} on {c.code} normalizes to {key!r}, "
                        f"which is also the normalized label of {other.code} "
                        f"({other.label!r}) — alias lookup will be shadowed"
                    ),
                ))

    # R8 — abbrev unreachability in the active frame.  Only enumerated
    # when a frame is supplied; this is post-projection (e.g., a vocab
    # restriction may drop nodes the abbrev index still references).
    # After R7, ``frame.resolve_annotation`` already walks down/up to
    # find a covering FE, so a None here is a genuine "no path" signal
    # the operator should reconcile in the source data, not a
    # resolution-path bug.
    if frame is not None:
        idx = getattr(category_set, "all_by_abbrev", None) or {}
        for abbrev, cat in sorted(idx.items()):
            if not abbrev or not cat.code:
                continue
            try:
                fe = frame.resolve_annotation(abbrev)
            except Exception:  # defensive — never let a probe abort validation
                fe = None
            if fe is None:
                findings.append(TaxonomyFinding(
                    kind="abbrev_unreachable_in_frame",
                    severity="warning",
                    codes=(cat.code,),
                    detail=(
                        f"abbrev {abbrev!r} (code {cat.code!r}) cannot be "
                        f"resolved by the active frame — neither a singleton, "
                        f"a curated internal node, nor any walk-down/walk-up "
                        f"path covers it.  LLM votes for {abbrev!r} will be "
                        f"discarded; reconcile vocabulary projection."
                    ),
                ))

    return findings


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


def load_sample_vocabulary(*, hierarchical: bool = True) -> CategorySet:
    """Load the expanded OOTB sample vocabulary from data/sample/ontology.json.

    This is the 300-node BFO-grounded vocabulary used for the OOTB
    sample source. It extends the universal vocabulary with
    domain-specific categories across the CCO ICE trichotomy.  All
    nodes (parent and terminal alike) are valid tagging targets.
    """
    sample_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "sample"
    path = sample_dir / "ontology.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Sample vocabulary not found: {path}. "
            "Run 'uv run python scripts/expand_vocabulary.py' to generate it."
        )
    return load_annotations_from_json(path, hierarchical=hierarchical)


def compose_vocabularies(
    base: HierarchicalCategorySet,
    domain: list[dict] | CategorySet,
) -> HierarchicalCategorySet:
    """Compose a universal base vocabulary with domain extensions.

    Domain records declare ``parent_code`` referencing universal codes,
    attaching as new categories (or subtrees) to the universal
    hierarchy.  Every attached node — terminal or parent — is a
    first-class tagging target in the merged vocabulary.

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

    # Merge: domain categories extend base, dedup on code.  Single
    # category list now — the legacy leaves/all split has been removed
    # because parent and terminal nodes are equally first-class
    # prediction targets.
    base_codes = {c.code for c in base.categories}
    merged = list(base.categories)
    for cat in domain_cs.categories:
        if cat.code not in base_codes:
            merged.append(cat)

    return HierarchicalCategorySet(
        name="composed",
        categories=merged,
    )
