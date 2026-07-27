"""Read Ægir's Apache Atlas projection as an Atelier resource.

This is the first *direct* Atlas integration beyond the health probe.  Ægir
projects its deterministic DDL spine into a live Atlas as the honest
``rdbms_*`` model (``rdbms_db`` → ``rdbms_table`` → ``rdbms_column`` +
``rdbms_foreign_key``) and publishes the ontology as the ``Aegir Ontology``
glossary (540 terms, each carrying SKOS surface forms).  This module turns
those two surfaces into Atelier-consumable inputs:

* :func:`read_tables` — table/column **schema** (discovery source).  Atlas
  carries the *real* column names (``legalname``, ``email`` …), unlike the
  obfuscated names in the blind release corpus, but it holds **no sample
  values** — so schema-only ``TableSample``\\ s exercise the name/pattern
  evidence channels, and are meant to be *joined* with release values (see
  :mod:`atelier.classify.aegir_release`) for the value-driven channels.

* :func:`read_glossary` — the ontology **vocabulary** (term name == Ægir
  ``template_id``, plus surface forms for retrieval / ``name_match`` /
  ``maxsim``).  Surface forms are parsed from the term ``longDescription``
  text convention emitted by Ægir's projector (no structured altLabel field
  exists yet); :func:`_parse_surface_forms` is defensive about format drift.

All reads are read-only.  Writing Atelier's predictions back onto these same
``rdbms_column`` entities (the "shared scoreboard") is the sync path in
:mod:`atelier.governance.sync`; note that Ægir registers ``cta_*`` /
``domain_*`` classification typedefs but does **not** currently apply them to
entities, so a distinct Atelier classification namespace avoids collision.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from atelier.classify.sampler import ColumnSample, TableSample
from atelier.governance.atlas import AtlasClient

logger = logging.getLogger(__name__)

# Ægir projects its DDL footprint under this db; the corpus views live under
# ``corpus``.  Default reads target the footprint (base tables), not views.
DEFAULT_DB_PREFIX = "footprint"
DEFAULT_GLOSSARY = "Aegir Ontology"

# Surrogate PK columns Ægir adds per table; non-semantic, dropped from the
# release.  Filtered by default so schema reads match the release's columns.
_SURROGATE_COLS = frozenset({"id"})

# "Surface forms (SKOS altLabel / retrieval): a, b, c" — text convention in
# the glossary term longDescription emitted by Ægir's Atlas projector.
_SURFACE_RE = re.compile(r"Surface forms[^:]*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_AXIOM_RE = re.compile(r"Axiom \(Manchester\):\s*(.+?)(?:\n\n|$)", re.IGNORECASE | re.DOTALL)


def _db_of(qualified_name: str) -> str:
    """``footprint.t_foo@aegir`` → ``footprint`` (the db component)."""
    head = qualified_name.split("@", 1)[0]
    return head.split(".", 1)[0] if "." in head else head


def _col_name_of(ref: dict) -> str:
    """Bare column name from a column relationship ref (``displayText``)."""
    return ref.get("displayText") or ref.get("attributes", {}).get("name") or ""


def read_table_schema(
    atlas: AtlasClient,
    table_qn: str,
    *,
    drop_surrogate: bool = True,
) -> TableSample | None:
    """Build a value-less :class:`TableSample` from one ``rdbms_table`` entity.

    Column names come from the table's ``columns`` relationship.  No sample
    values are available from Atlas — join with the release corpus for those.
    """
    ent = atlas.get_entity_by_qualified_name("rdbms_table", table_qn)
    if not ent:
        return None
    entity = ent.get("entity", {})
    attrs = entity.get("attributes", {})
    rel = entity.get("relationshipAttributes", {})

    table_name = attrs.get("name") or table_qn.split("@", 1)[0]
    database = _db_of(table_qn)

    col_refs = rel.get("columns") or attrs.get("columns") or []
    names = [_col_name_of(c) for c in col_refs]
    names = [n for n in names if n]
    if drop_surrogate:
        names = [n for n in names if n not in _SURROGATE_COLS]

    columns = [
        ColumnSample(
            name=n,
            column_type=None,
            values=[],
            table_name=table_name,
            database=database,
            siblings=names,
        )
        for n in names
    ]
    return TableSample(name=table_name, database=database, columns=columns)


def read_tables(
    atlas: AtlasClient,
    *,
    db_prefix: str = DEFAULT_DB_PREFIX,
    limit: int = 50,
    drop_surrogate: bool = True,
) -> list[TableSample]:
    """Read up to ``limit`` ``rdbms_table`` schemas under ``db_prefix``.

    Tables are discovered via ``basic_search`` (filtered to the requested db,
    excluding ``corpus`` views by default), then each is expanded to a
    value-less ``TableSample`` via :func:`read_table_schema`.
    """
    # Over-fetch to survive the db-prefix filter, then trim to ``limit``.
    res = atlas.basic_search("rdbms_table", limit=max(limit * 3, limit), offset=0)
    out: list[TableSample] = []
    for e in res.get("entities", []):
        qn = e.get("attributes", {}).get("qualifiedName", "")
        if db_prefix and _db_of(qn) != db_prefix:
            continue
        ts = read_table_schema(atlas, qn, drop_surrogate=drop_surrogate)
        if ts and ts.columns:
            out.append(ts)
        if len(out) >= limit:
            break
    logger.info(
        "Read %d %s tables from Atlas (%d columns)",
        len(out), db_prefix, sum(len(t.columns) for t in out),
    )
    return out


@dataclass
class GlossaryTerm:
    """One ontology term from the Atlas glossary, as a vocabulary entry.

    ``name`` equals the Ægir ``template_id`` — the join key to the release
    reference codes (``reference.parquet.template_id``).
    """

    name: str
    qualified_name: str
    definition: str = ""
    surface_forms: list[str] = field(default_factory=list)
    category: str = ""
    axiom: str = ""

    def retrieval_text(self) -> str:
        """Name + surface forms joined — the text fed to name/maxsim channels."""
        parts = [self.name.replace("_", " "), *self.surface_forms]
        return " ".join(dict.fromkeys(p for p in parts if p))


def _parse_surface_forms(long_description: str) -> list[str]:
    if not long_description:
        return []
    m = _SURFACE_RE.search(long_description)
    if not m:
        return []
    return [s.strip() for s in m.group(1).split(",") if s.strip()]


def _parse_category(usage: str) -> str:
    """Ægir encodes the family in ``usage`` ("… in the 'directive_governance' category.")."""
    if not usage:
        return ""
    m = re.search(r"'([^']+)'\s+category", usage)
    return m.group(1) if m else ""


def read_glossary(
    atlas: AtlasClient,
    *,
    glossary_name: str = DEFAULT_GLOSSARY,
    page_size: int = 100,
) -> list[GlossaryTerm]:
    """Read the Ægir ontology glossary as a list of :class:`GlossaryTerm`.

    Pages through ``GET /glossary/{guid}/terms`` (the rich endpoint) and
    parses surface forms / category from the term text conventions.
    """
    glossaries = atlas.get_all_glossaries()
    guid = next(
        (g.get("guid") for g in glossaries if g.get("name") == glossary_name), None
    )
    if not guid:
        logger.warning("Glossary %r not found in Atlas", glossary_name)
        return []

    terms: list[GlossaryTerm] = []
    offset = 0
    while True:
        page = atlas.get_glossary_terms(guid, limit=page_size, offset=offset)
        if not page:
            break
        for t in page:
            long_desc = t.get("longDescription") or ""
            axiom_m = _AXIOM_RE.search(long_desc)
            terms.append(
                GlossaryTerm(
                    name=t.get("name", ""),
                    qualified_name=t.get("qualifiedName", ""),
                    definition=t.get("shortDescription") or "",
                    surface_forms=_parse_surface_forms(long_desc),
                    category=_parse_category(t.get("usage") or ""),
                    axiom=axiom_m.group(1).strip() if axiom_m else "",
                )
            )
        if len(page) < page_size:
            break
        offset += page_size

    logger.info("Read %d glossary terms from %r", len(terms), glossary_name)
    return terms
