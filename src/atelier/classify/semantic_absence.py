"""Positively manage semantic absence.

When the system lacks a semantic axis — a value's **unit**, a monetary
amount's **currency**, an entity's **relation** — that absence must be
represented *explicitly as a first-class epistemic quantity*, never left as
a silent zero or an implicit assumption. This is the same epistemic honesty
the DST core already applies to *within-frame* uncertainty (mass on Θ /
ignorance; the ``Pl − Bel`` belief gap), extended to *frame-incompleteness*:

  - Θ (``belief.py``) says: "uncertain *which code* in the frame."
  - Semantic absence says: "a whole *axis* isn't in the frame at all."

Both are uncertainty in Audun Jøsang's sense (the ``u`` of a subjective-logic
opinion). Omitting an un-modeled axis is not neutral: the classification
looks complete while silently dropping a load-bearing dimension, and because
the axis isn't in the frame, the omission is un-flaggable at the source and
only surfaces where a downstream consumer assumes a value (kg-vs-lb,
USD-vs-EUR, a pivot/materialized view copied without its association table).
Representing it explicitly is the epistemically honest alternative. See
docs/src/architecture/cco-coverage.md and the
``feedback_positively_manage_semantic_absence`` /
``feedback_cco_completeness_is_correctness`` directives.

A future richer representation is a full subjective-logic opinion (b/d/u)
over each axis's domain; for now the explicit ``UNRESOLVED`` token + claim
gating is the minimum that makes absence visible and non-assumable.
"""
from __future__ import annotations

UNRESOLVED = "UNRESOLVED"

# Each absence axis is the *unfilled* instance of a canonical CCO annotation
# property from the ExtendedRelationOntology — grounding the absence in CCO
# rather than an ad-hoc string. `unit: UNRESOLVED` is literally an unfilled
# `cco:has token unit`. (These annotation properties apply to information
# content entities — which our taxonomy terms are — so the same module also
# gives us `acronym` (ont00001753) ≡ our abbrev/mnemonic and `definition
# source` (ont00001754) ≡ our DBpedia-IRI provenance, reachable now.)
from atelier.classify.cco_annotations import HAS_TOKEN_UNIT

AXIS_CCO_PROPERTY: dict[str, str] = {
    # "has token unit" — the measurement/currency unit of the token used to
    # express an ICE; the formal slot a resolved unit or currency fills.
    "unit": HAS_TOKEN_UNIT,
    "currency_unit": HAS_TOKEN_UNIT,
}


def cco_property_for_axis(axis: str) -> str | None:
    """Canonical CCO annotation property an absence axis is the unfilled
    instance of (ExtendedRelationOntology), or None if not yet grounded."""
    return AXIS_CCO_PROPERTY.get(axis)

# Value-level interface axes a referent CCO module carries — the dimensions
# whose silent omission causes boundary failures. A measurement magnitude is
# meaningless without its unit; a monetary amount without its currency unit.
# Keyed by CCO module code (see cco_modules.json).
INTERFACE_AXES: dict[str, tuple[str, ...]] = {
    "QUAL": ("unit",),
    "CUR": ("currency_unit",),
}

# The relation axis is context-dependent: a relational / EAV read (or a
# pivot/materialized view detached from its association table) whose relation
# to the subject is not resolved.
RELATION_AXIS = "relation"


def unresolved_axes(
    referent_module: str | None,
    *,
    relational_context: bool = False,
    resolved: set[str] | None = None,
) -> list[str]:
    """Interface axes for ``referent_module`` not present in ``resolved``.

    In wide-table CTA these axes are never resolved (units/currency live
    inside values; relations in schema), so they surface as absent. An
    EAV / unit / relation resolver fills ``resolved`` to clear them.
    """
    have = set(resolved or ())
    axes = [a for a in INTERFACE_AXES.get(referent_module or "", ()) if a not in have]
    if relational_context and RELATION_AXIS not in have:
        # Per-read resolution, NOT module coverage: even once the Extended
        # Relation vocabulary exists ('covered'), a relational read whose
        # relation isn't resolved still leaves the relation axis absent.
        axes.append(RELATION_AXIS)
    return axes


def semantic_absence(
    referent_module: str | None,
    *,
    relational_context: bool = False,
    resolved: set[str] | None = None,
) -> dict[str, str]:
    """Explicit, queryable absence record: ``{axis: 'UNRESOLVED'}`` per
    un-resolved interface axis. Attach to a classification so no downstream
    consumer can silently assume a unit / currency / relation. An empty dict
    means nothing is absent.
    """
    return {
        a: UNRESOLVED
        for a in unresolved_axes(
            referent_module, relational_context=relational_context, resolved=resolved
        )
    }


def gate_claim(
    referent_module: str | None,
    claim_axes: set[str],
    *,
    relational_context: bool = False,
    resolved: set[str] | None = None,
) -> tuple[bool, list[str]]:
    """Gate a downstream claim that depends on ``claim_axes``.

    Returns ``(ok, blocking)``: the claim is blocked when any axis it depends
    on is unresolved — *refuse, don't guess*. E.g. asserting two ``mass``
    columns are compatible depends on ``{'unit'}`` and is blocked until the
    unit axis is resolved on both.
    """
    absent = set(
        unresolved_axes(
            referent_module, relational_context=relational_context, resolved=resolved
        )
    )
    blocking = sorted(absent & set(claim_axes))
    return (not blocking, blocking)
