# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Per-iteration scoring against an Opus-crafted reference artifact.

The pipeline writes a single ``classifications.json`` at run completion,
but the bootstrap loop iteratively refines ``state.labels``.  Without
intra-run visibility we cannot tell whether iteration N's revisit pass
helps or hurts — and during a six-threshold sweep we cannot catch a
misconfigured vocabulary or an over-aggressive bel_threshold until we
have already burned hours of compute.  Incremental scoring closes that
gap: each iteration's fused labels are scored against the reference
artifact, the trend is appended to ``scoring_trend.json``, and a
per-iteration grouped-error TSV is written for failure-mode analysis.

Design contract
---------------

- **Fail-fast on misconfiguration.**  Operators tune more than the
  bel_threshold with this scaffold, so the cost of bad scoring data is
  much higher than the cost of a hard error.  If the reference file is
  missing, malformed, or references annotations the loaded vocabulary
  cannot resolve, we raise before LLM_SWEEP starts.  Better to bail at
  iteration 0 than to ship trend data computed against the wrong codes.

- **Vocabulary alignment is mandatory.**  The reference and the loaded
  vocabulary must agree on every annotation mnemonic the reference
  cites.  This is the exact guard that would have caught the
  bel_threshold-2026-05-15T22:42:57Z sweep regression at iteration 0
  rather than at hour 8.  ``validate_reference_against_vocab`` raises
  with the offending mnemonics so the operator can fix the source
  vocabulary or the reference, not both at once.

- **Hierarchy-aware failure-mode buckets.**  The reference POC ``code``
  field is a dot-numeric hierarchy (``1.1.1.9.1`` is a descendant of
  ``1.1.1.9``).  ``classify_failure_mode`` walks that hierarchy to
  distinguish the patterns operators actually act on:
  ``parent_instead_of_leaf`` (model went too shallow),
  ``child_instead_of_parent`` (too deep, e.g. PAN vs C_FD parent),
  ``sibling_within_subtree`` (same depth-3 prefix, different leaf —
  the prefix-3 boundary you specified), ``wrong_subtree`` (different
  depth-3 prefix), plus ``hallucinated_annotation`` and
  ``missing_prediction`` for the input-side failures.

- **Aggregated error rows.**  At each iteration we group errors by
  (predicted_label, predicted_annotation, reference_label,
  reference_annotation, wrong_subtree) and emit one row per cluster
  with a count and up to five example columns.  The aggregated view
  surfaces failure clusters that the per-column view would scatter
  across thousands of rows.

Output artifacts (all under ``build/results/{run_id}/``)
--------------------------------------------------------

  * ``scoring_trend.json`` — list of per-iteration aggregate records.
    Each record carries strict accuracy, hierarchy-aware accuracy,
    per-mode counts, belief diagnostics, and disk-headroom snapshots.
  * ``scoring_errors_iter{N}.tsv`` — grouped error table for iteration
    ``N``, sorted by descending count.  TSV (not CSV) so labels with
    embedded commas survive a naive split.
  * ``scoring_summary.md`` — written at run completion.  Iteration
    trend table + final-iteration grouped errors + delta highlights.

Reference artifact format
-------------------------

A flat JSON object mapping ``"table_name.column_name"`` to the
expected annotation mnemonic (``abbrev``), or ``null`` to mark a
column as unreviewed::

  {
    "academic_records.row_id": "INOS",
    "academic_records.comm_val": "A_PHN",
    "academic_records.start_date": "TRANSDATE",
    "academic_records.period_val": null,    # excluded from scoring
    ...
  }

Nested form is also accepted and auto-flattened (see
``_load_reference_artifact``).
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atelier.classify.taxonomy import HierarchicalCategorySet

log = logging.getLogger(__name__)

_PREFIX_DEPTH = 3
# Minimum number of completed historical runs before we trust their
# size statistics for the disk-guard projection.  Below this floor we
# fall back to a hardcoded byte budget (see ``DiskGuardConfig``).
MIN_HISTORICAL_RUNS_FOR_STATS = 3

# Vocab-mismatch handling: distinguish a systemic mismatch (wrong vocab
# loaded entirely — the bel_threshold-2026-05-15T22:42:57Z signature)
# from an ad-hoc mismatch (a handful of GT mnemonics absent from an
# otherwise-correct vocab snapshot — the BRCPF signature).  When the
# missing-fraction exceeds ``CATASTROPHIC_MISSING_FRACTION`` AND the
# affected column count clears ``MIN_AFFECTED_FOR_CATASTROPHIC``, we
# hard-fail; otherwise the affected GT entries are marked unscorable
# and the run continues with a logged warning.
#
# The two-gate design avoids the false positive where a tiny GT
# happens to cite 1 missing mnemonic out of 4 (would be 25%, but only
# one column — clearly not a systemic vocab swap).
CATASTROPHIC_MISSING_FRACTION = 0.20
MIN_AFFECTED_FOR_CATASTROPHIC = 10


# ── Reference artifact loading + validation ─────────────────────────


def _norm(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def load_reference_artifact(path: Path) -> dict[str, str | None]:
    """Load and flatten the Opus-crafted reference artifact.

    Returns ``{table_name.column_name: annotation_or_None}``.  Keys with
    ``null`` values are kept so the scoring path can distinguish
    "explicitly unreviewed" from "missing from the reference".

    Raises:
        FileNotFoundError: when ``path`` does not exist.
        ValueError: when the file is not a JSON object, or when a value
            is not a string / None / dict.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Reference artifact not found at {path}.  Set "
            f"ATELIER_GROUND_TRUTH_PATH or classify.evaluation."
            f"ground_truth_path to a valid Opus-crafted reference JSON."
        )
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Reference artifact {path} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise ValueError(
            f"Reference artifact {path} root must be a JSON object, "
            f"got {type(raw).__name__}"
        )

    flat: dict[str, str | None] = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            for col, ann in v.items():
                flat[f"{k}.{col}"] = (
                    None if ann in (None, "") else _norm(ann)
                )
        elif isinstance(v, (str, type(None))):
            flat[_norm(k)] = None if v in (None, "") else _norm(v)
        else:
            raise ValueError(
                f"Reference artifact {path} key {k!r}: value must be "
                f"str/null/dict, got {type(v).__name__}"
            )
    return flat


@dataclass
class VocabAlignment:
    """Result of comparing a reference artifact to a loaded vocabulary.

    Captures the inputs to the systemic-vs-ad-hoc decision so the
    scorer, summary report, and audit log all read from one source of
    truth.  ``unscorable_mnemonics`` is empty when the reference is
    fully aligned; ``unscorable_columns`` is the set of GT keys whose
    annotations are in ``unscorable_mnemonics`` (the columns the
    scorer will skip in strict / hierarchy-aware comparisons).
    """

    cited_count: int
    vocab_abbrev_count: int
    unscorable_mnemonics: list[str]
    unscorable_columns: set[str]
    missing_fraction: float
    classification: str  # "aligned" | "ad_hoc_mismatch" | "systemic_mismatch"


def validate_reference_against_vocab(
    reference: dict[str, str | None],
    category_set: HierarchicalCategorySet,
    *,
    reference_path: Path | None = None,
    catastrophic_fraction: float = CATASTROPHIC_MISSING_FRACTION,
    min_affected_for_catastrophic: int = MIN_AFFECTED_FOR_CATASTROPHIC,
) -> VocabAlignment:
    """Classify reference-vs-vocab alignment as aligned / ad-hoc / systemic.

    Two failure modes, two responses:

    * **Systemic mismatch** (the bel_threshold-2026-05-15T22:42:57Z
      signature).  The pipeline loaded the wrong vocabulary; most or
      all of the reference's annotations are unresolvable.  Raises
      ``ValueError`` — the run cannot produce trustworthy scoring data.

    * **Ad-hoc mismatch** (the BRCPF signature).  The right vocabulary
      is loaded but a handful of reference mnemonics aren't in this
      snapshot.  Possible causes worth chasing offline:

        - GT was authored against an older / different vocab revision
          and a mnemonic was renamed or dropped.
        - LLM hallucination during GT construction (the Agent SDK loop
          can emit a mnemonic that looks plausible but isn't in the
          vocab — particularly if a Bedrock KV-cache artifact let an
          earlier turn's prediction leak into a later prompt without
          the system message restating the vocab).
        - Vocab snapshot is stale relative to the upstream Hive
          ``annotations`` table.

      Affected columns are marked unscorable; the scorer treats them
      like ``None`` entries in the reference.  The run continues with
      a logged warning so the operator can audit the mnemonics offline
      without losing the rest of the scoring data.

    The two gates that distinguish the cases:

    * ``missing_fraction = |missing| / |cited|`` ≥ ``catastrophic_fraction``
      (default 20%): the proportion suggests systemic vocab swap.
    * ``affected_column_count`` ≥ ``min_affected_for_catastrophic``
      (default 10): a tiny GT with one stray mnemonic must never trip
      the catastrophic branch even at 100% rate.

    Both gates must clear before we hard-fail.  ``VocabAlignment.classification``
    is the resolved verdict.
    """
    vocab_abbrevs = set(category_set.all_by_abbrev.keys())
    cited = {a for a in reference.values() if a is not None and a != ""}
    missing = sorted(cited - vocab_abbrevs)
    if not missing:
        log.info(
            "Reference artifact aligned with vocabulary: "
            "%d cited annotations all present (%d total reference entries).",
            len(cited), len(reference),
        )
        return VocabAlignment(
            cited_count=len(cited),
            vocab_abbrev_count=len(vocab_abbrevs),
            unscorable_mnemonics=[],
            unscorable_columns=set(),
            missing_fraction=0.0,
            classification="aligned",
        )

    # Compute which reference columns are affected.
    missing_set = set(missing)
    affected_columns = {
        k for k, v in reference.items()
        if v is not None and v in missing_set
    }
    missing_fraction = len(missing) / max(1, len(cited))

    is_catastrophic = (
        missing_fraction >= catastrophic_fraction
        and len(affected_columns) >= min_affected_for_catastrophic
    )

    sample = ", ".join(missing[:10])
    extra = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
    src = f" from {reference_path}" if reference_path else ""

    if is_catastrophic:
        # Systemic mismatch.  This is the load-the-wrong-vocab case —
        # not recoverable by marking some columns unscorable, because
        # the scoring numbers would not reflect the same algorithm we
        # intend to evaluate.  Hard-fail at startup so 8 hours of
        # compute aren't wasted on the wrong codebook.
        raise ValueError(
            f"Reference artifact{src} cites {len(missing)} annotation(s) "
            f"not present in the loaded vocabulary "
            f"({missing_fraction * 100:.1f}% of {len(cited)} cited; "
            f"{len(affected_columns)} columns affected): {sample}{extra}.  "
            f"Vocabulary loaded {len(vocab_abbrevs)} abbrev(s).  "
            f"This rate crosses the catastrophic-mismatch threshold "
            f"({catastrophic_fraction * 100:.0f}% with at least "
            f"{min_affected_for_catastrophic} affected columns), which "
            f"signals the wrong vocabulary is loaded — not a few stray "
            f"mnemonics in an otherwise-correct codebook.  Either select "
            f"the correct data source (source_id → vocab_uri) or replace "
            f"the reference artifact with one keyed to the loaded "
            f"vocabulary.  See the bel_threshold-2026-05-15T22:42:57Z "
            f"sweep regression for the failure mode this guard prevents."
        )

    # Ad-hoc mismatch.  Could be a stale GT field, an LLM hallucination
    # during GT construction, a Bedrock KV-cache artifact (the Agent
    # SDK loop occasionally lets a prior turn's mnemonic leak into a
    # later prompt when the cache aggressively reuses prefix context),
    # or a vocab snapshot that lags GT.  Worth investigating offline,
    # but not a blocker for this run.
    log.warning(
        "Reference artifact%s cites %d annotation(s) absent from the "
        "loaded vocabulary (%.1f%% of %d cited; %d columns affected): %s%s. "
        "Below the catastrophic-mismatch threshold (%.0f%% with at least "
        "%d affected columns) — treating affected columns as UNSCORABLE "
        "and continuing.  Causes worth chasing offline: stale GT field, "
        "LLM hallucination during GT construction, Bedrock KV-cache "
        "artifact that leaked a prior-turn mnemonic into a later GT "
        "prompt, or a vocab snapshot lagging the upstream annotations "
        "table.  See scoring_summary.md for the affected column list.",
        src, len(missing), missing_fraction * 100, len(cited),
        len(affected_columns), sample, extra,
        catastrophic_fraction * 100, min_affected_for_catastrophic,
    )

    return VocabAlignment(
        cited_count=len(cited),
        vocab_abbrev_count=len(vocab_abbrevs),
        unscorable_mnemonics=missing,
        unscorable_columns=affected_columns,
        missing_fraction=missing_fraction,
        classification="ad_hoc_mismatch",
    )


# ── Hierarchy + failure-mode classification ─────────────────────────


def _split_code(code: str) -> list[str]:
    """Split a dot-numeric code into parts.  Empty string → empty list."""
    s = (code or "").strip()
    return s.split(".") if s else []


def is_ancestor(maybe_ancestor: str, descendant: str) -> bool:
    """True iff ``maybe_ancestor`` is a strict prefix of ``descendant``.

    Example: ``1.1.1`` is an ancestor of ``1.1.1.9.1``; ``1.1.1`` is
    NOT an ancestor of itself or of ``1.1.2``.
    """
    a = _split_code(maybe_ancestor)
    d = _split_code(descendant)
    if not a or not d or len(a) >= len(d):
        return False
    return d[: len(a)] == a


def share_prefix(code_a: str, code_b: str, *, depth: int = _PREFIX_DEPTH) -> bool:
    """True when both codes have at least ``depth`` parts and those parts agree.

    For codes shorter than ``depth`` parts, falls back to "shorter is an
    ancestor of the longer or vice versa".  This handles short top-level
    codes (e.g. ``0.1`` for INOS) without falsely flagging them as
    wrong-subtree against unrelated leaves.
    """
    a = _split_code(code_a)
    b = _split_code(code_b)
    if not a or not b:
        return False
    short_len = min(len(a), len(b))
    if short_len < depth:
        # One (or both) sits above the depth-3 boundary.  Treat them as
        # "on the same path" only when one is a prefix of the other.
        return a[:short_len] == b[:short_len]
    return a[:depth] == b[:depth]


def classify_failure_mode(
    predicted_code: str,
    predicted_annotation: str,
    reference_code: str,
    vocab_abbrevs: set[str],
) -> str:
    """Bucket a wrong prediction.  See module docstring for definitions.

    Returns one of:
      - ``missing_prediction`` — predicted_annotation is empty.
      - ``hallucinated_annotation`` — predicted_annotation not in vocab.
      - ``parent_instead_of_leaf`` — predicted is a strict ancestor.
      - ``child_instead_of_parent`` — predicted is a strict descendant.
      - ``sibling_within_subtree`` — same depth-3 prefix, neither ancestor.
      - ``wrong_subtree`` — different depth-3 prefix.
    """
    if not predicted_annotation:
        return "missing_prediction"
    if predicted_annotation not in vocab_abbrevs:
        return "hallucinated_annotation"
    if is_ancestor(predicted_code, reference_code):
        return "parent_instead_of_leaf"
    if is_ancestor(reference_code, predicted_code):
        return "child_instead_of_parent"
    if share_prefix(predicted_code, reference_code):
        return "sibling_within_subtree"
    return "wrong_subtree"


# ── Per-iteration scoring core ──────────────────────────────────────


@dataclass
class IterationScore:
    """Aggregate per-iteration score record (one row in scoring_trend.json)."""

    iteration: int
    phase: str  # "post_fusion_iter_N" or "final"
    wall_clock_s: float

    # Coverage / strictness
    scored: int           # columns the reference annotates AND the pipeline predicted
    missing_in_run: int   # reference-annotated columns the pipeline didn't see
    unscorable: int       # GT columns whose mnemonic isn't in the loaded vocab
    strict_match: int
    strict_pct: float
    delta_strict_pct_vs_prev: float | None  # None on iteration 1

    # Hierarchy-aware accuracy
    on_right_path: int    # prefix-3 match (incl. ancestor/descendant)
    on_right_path_pct: float
    parent_match: int     # predicted is an ancestor of reference
    child_match: int      # predicted is a descendant of reference
    sibling_within_subtree: int
    wrong_subtree: int
    missing_prediction: int
    hallucinated_annotation: int

    # Belief diagnostics (mean across scored columns; -1.0 when unset)
    mean_belief: float
    mean_plausibility: float
    mean_gap: float
    mean_conflict: float
    needs_clarification_count: int

    # Cost / resource accounting
    llm_calls_total: int
    llm_attempts_total: int
    tokens_input: int
    tokens_output: int
    revisited_columns: int  # how many columns _llm_revisit touched this iter
    disk_free_bytes: int

    # Failure mode bucket counts (mutually exclusive — sum == scored - strict_match)
    failure_modes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "iteration": self.iteration,
            "phase": self.phase,
            "wall_clock_s": round(self.wall_clock_s, 2),
            "scored": self.scored,
            "missing_in_run": self.missing_in_run,
            "unscorable": self.unscorable,
            "strict_match": self.strict_match,
            "strict_pct": round(self.strict_pct, 2),
            "delta_strict_pct_vs_prev": (
                None if self.delta_strict_pct_vs_prev is None
                else round(self.delta_strict_pct_vs_prev, 2)
            ),
            "on_right_path": self.on_right_path,
            "on_right_path_pct": round(self.on_right_path_pct, 2),
            "parent_match": self.parent_match,
            "child_match": self.child_match,
            "sibling_within_subtree": self.sibling_within_subtree,
            "wrong_subtree": self.wrong_subtree,
            "missing_prediction": self.missing_prediction,
            "hallucinated_annotation": self.hallucinated_annotation,
            "mean_belief": round(self.mean_belief, 4),
            "mean_plausibility": round(self.mean_plausibility, 4),
            "mean_gap": round(self.mean_gap, 4),
            "mean_conflict": round(self.mean_conflict, 4),
            "needs_clarification_count": self.needs_clarification_count,
            "llm_calls_total": self.llm_calls_total,
            "llm_attempts_total": self.llm_attempts_total,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "revisited_columns": self.revisited_columns,
            "disk_free_bytes": self.disk_free_bytes,
            "failure_modes": dict(self.failure_modes),
        }
        return out


@dataclass
class ErrorRow:
    """One row in the per-iteration aggregated error table.

    The grouping key is (predicted_label, predicted_annotation,
    reference_label, reference_annotation, wrong_subtree).  Count and
    example_columns are accumulated; up to five examples are retained
    per cluster so the operator has at least one drill-in path per
    failure pattern.
    """

    predicted_label: str
    predicted_annotation: str
    predicted_code: str
    reference_label: str
    reference_annotation: str
    reference_code: str
    wrong_subtree: bool
    failure_mode: str
    count: int = 0
    example_columns: list[str] = field(default_factory=list)

    def add_example(self, table_column: str, max_examples: int = 5) -> None:
        self.count += 1
        if len(self.example_columns) < max_examples:
            self.example_columns.append(table_column)

    def tsv_row(self) -> str:
        """Render as a TSV row matching the spec from the design doc.

        Columns: count, predicted_label – predicted_annotation,
        reference_label – reference_annotation, wrong_subtree (Y/N),
        predicted_code, reference_code, example_columns (comma-joined).

        Tabs inside any cell are replaced with a space so the row stays
        single-line.
        """
        wrong = "Y" if self.wrong_subtree else "N"
        cells = [
            str(self.count),
            f"{self.predicted_label} – {self.predicted_annotation}",
            f"{self.reference_label} – {self.reference_annotation}",
            wrong,
            self.failure_mode,
            self.predicted_code,
            self.reference_code,
            ", ".join(self.example_columns),
        ]
        return "\t".join(c.replace("\t", " ") for c in cells)


@dataclass
class ScoringContext:
    """Per-run scoring state.  Reused across iterations + final pass.

    Built once at pipeline entry (after vocab load) and threaded into
    the bootstrap loop.  ``disabled`` is the no-op flag: when no
    reference path is configured, every ``score_iteration`` call is a
    cheap return.

    ``alignment`` captures the outcome of
    ``validate_reference_against_vocab`` — aligned, ad-hoc mismatch
    (with the affected GT keys marked unscorable), or systemic mismatch
    (the context never gets built; the validator raises instead).
    """

    enabled: bool
    reference: dict[str, str | None]  # flattened reference artifact
    category_set: HierarchicalCategorySet | None
    results_dir: Path
    started_at_monotonic: float
    alignment: VocabAlignment | None = None
    # Cached: previous strict_pct so we can emit delta_vs_prev cheaply.
    prev_strict_pct: float | None = None
    # Accumulator for scoring_trend.json — appended-and-rewritten so a
    # mid-run abort still leaves a parseable artifact on disk.
    trend: list[dict[str, Any]] = field(default_factory=list)

    @property
    def vocab_abbrevs(self) -> set[str]:
        if self.category_set is None:
            return set()
        return set(self.category_set.all_by_abbrev.keys())

    @property
    def unscorable_columns(self) -> set[str]:
        return self.alignment.unscorable_columns if self.alignment else set()


def build_scoring_context(
    *,
    enabled: bool,
    reference_path: Path | None,
    category_set: HierarchicalCategorySet,
    results_dir: Path,
) -> ScoringContext:
    """Load + validate the reference artifact and return a context.

    When ``enabled`` is False (or ``reference_path`` is None), returns a
    no-op context whose ``score_iteration`` is a fast skip.

    When ``enabled`` is True, loads the reference, validates it against
    ``category_set``, and raises on any misalignment.  The pipeline is
    expected to surface that as an FSM.ERROR transition.
    """
    if not enabled or reference_path is None:
        log.info(
            "Incremental scoring disabled (enabled=%s, reference_path=%r) — "
            "trend files will not be written.",
            enabled, reference_path,
        )
        return ScoringContext(
            enabled=False,
            reference={},
            category_set=None,
            results_dir=results_dir,
            started_at_monotonic=time.monotonic(),
        )

    reference = load_reference_artifact(reference_path)
    alignment = validate_reference_against_vocab(
        reference, category_set, reference_path=reference_path,
    )
    # Systemic mismatch raises above and never reaches here.  Aligned
    # and ad-hoc-mismatch both proceed, with ad-hoc carrying the
    # unscorable-column set on the alignment object.

    # Emit a startup-time alignment audit so the operator has a
    # machine-parseable record even when the summary report doesn't
    # land (e.g. the run aborts later for some other reason).  This
    # sidecar is the durable trace of "we noticed BRCPF early but
    # didn't fail the run."
    if alignment.classification != "aligned":
        try:
            results_dir.mkdir(parents=True, exist_ok=True)
            audit_path = results_dir / "scoring_vocab_alignment.json"
            audit_path.write_text(json.dumps({
                "schema_version": 1,
                "classification": alignment.classification,
                "missing_fraction": alignment.missing_fraction,
                "cited_count": alignment.cited_count,
                "vocab_abbrev_count": alignment.vocab_abbrev_count,
                "unscorable_mnemonics": alignment.unscorable_mnemonics,
                "affected_column_count": len(alignment.unscorable_columns),
                "affected_columns": sorted(alignment.unscorable_columns),
                "reference_path": str(reference_path),
            }, indent=2) + "\n")
        except OSError as exc:
            log.warning("Could not write scoring_vocab_alignment.json: %s", exc)

    return ScoringContext(
        enabled=True,
        reference=reference,
        category_set=category_set,
        results_dir=results_dir,
        started_at_monotonic=time.monotonic(),
        alignment=alignment,
    )


def _belief_diagnostics(
    state,
    column_qkeys: list[str],
) -> tuple[float, float, float, float, int]:
    """Compute (mean_bel, mean_pl, mean_gap, mean_K, needs_clar_count).

    Walks ``state.ml_belief`` / ``state.ml_plausibility`` /
    ``state.ml_conflict`` / ``state.ml_uncertainty`` over the supplied
    qkeys, ignoring entries the state hasn't yet populated.  Returns
    -1.0 for any aggregate computed over an empty set.
    """
    bels: list[float] = []
    pls: list[float] = []
    gaps: list[float] = []
    confs: list[float] = []
    needs_clar = 0
    for k in column_qkeys:
        bel = state.ml_belief.get(k)
        pl = state.ml_plausibility.get(k)
        conf = state.ml_conflict.get(k)
        if bel is not None:
            bels.append(bel)
        if pl is not None:
            pls.append(pl)
        if bel is not None and pl is not None:
            gap = pl - bel
            gaps.append(gap)
            # Mirror cautious_review's needs_clarification heuristic:
            # bel < 0.45 OR gap > 0.25 (the bel_floor / clarity_target
            # defaults the operator tunes).  We don't import those
            # thresholds here because the diagnostic is meant to be
            # comparable across runs, not to enforce the same gate.
            if bel < 0.45 or gap > 0.25:
                needs_clar += 1
        if conf is not None:
            confs.append(conf)
    return (
        sum(bels) / len(bels) if bels else -1.0,
        sum(pls) / len(pls) if pls else -1.0,
        sum(gaps) / len(gaps) if gaps else -1.0,
        sum(confs) / len(confs) if confs else -1.0,
        needs_clar,
    )


def _disk_free_bytes(path: Path) -> int:
    import shutil
    try:
        return shutil.disk_usage(str(path)).free
    except OSError:
        return -1


def score_iteration(
    ctx: ScoringContext,
    *,
    iteration: int,
    phase: str,
    state,                    # BootstrapState (avoid circular import for typing)
    column_qkeys: list[str],  # ordered list of qualified_name keys
    samples_by_name: dict,    # qkey → TableSample (for label + col name)
    revisited_count: int = 0,
) -> IterationScore | None:
    """Score the current ``state.labels`` against the reference.

    Writes ``scoring_trend.json`` (full file, indented) and
    ``scoring_errors_iter{N}.tsv`` (one TSV per iteration, header + sorted
    rows).  Returns the IterationScore for in-process consumers (the
    summary builder) or None when scoring is disabled.

    The grouped-error TSV is the operator's primary failure-mode lens —
    it stays small even on a 920-column corpus because errors cluster
    tightly (10-20 unique patterns covers the long tail in practice).
    """
    if not ctx.enabled:
        return None

    cat_set = ctx.category_set
    assert cat_set is not None  # enforced by build_scoring_context
    by_code = cat_set.all_by_code
    by_abbrev = cat_set.all_by_abbrev
    vocab_abbrevs = set(by_abbrev.keys())
    reference = ctx.reference
    unscorable_keys = ctx.unscorable_columns

    # Build the set of qkeys we will actually score: the reference
    # provides a non-None annotation that resolves in the loaded vocab
    # AND the column is known to the pipeline (state has a label for
    # it).  Three exclusion buckets:
    #
    #   * ``None`` reference value (operator marked unreviewed)
    #   * Reference key not in samples_by_name (pipeline never saw it)
    #     → counted under ``missing_in_run``
    #   * Reference annotation not in vocab (ad-hoc mismatch — BRCPF)
    #     → counted under ``unscorable``
    #
    # The ``scored`` denominator is what remains.  None of the
    # excluded entries contribute to strict_hits or on_right_path.
    scored = 0
    strict_hits = 0
    on_right_path = 0
    mode_counts: Counter[str] = Counter()
    errors: dict[tuple, ErrorRow] = {}

    missing_in_run = 0
    unscorable = 0
    for ref_key, expected_abbrev in reference.items():
        if expected_abbrev is None:
            continue
        if ref_key in unscorable_keys:
            unscorable += 1
            continue
        if ref_key not in samples_by_name:
            missing_in_run += 1
            continue
        scored += 1
        col = samples_by_name[ref_key]

        predicted_code = state.labels.get(ref_key, "")
        predicted_cat = by_code.get(predicted_code) if predicted_code else None
        predicted_annotation = (
            predicted_cat.abbrev if predicted_cat is not None else ""
        )
        predicted_label = (
            predicted_cat.label if predicted_cat is not None else ""
        )
        reference_cat = by_abbrev.get(expected_abbrev)
        reference_code = reference_cat.code if reference_cat is not None else ""
        reference_label = reference_cat.label if reference_cat is not None else ""

        if predicted_annotation == expected_abbrev:
            strict_hits += 1
            continue

        # Wrong — bucket it
        mode = classify_failure_mode(
            predicted_code, predicted_annotation, reference_code,
            vocab_abbrevs,
        )
        mode_counts[mode] += 1

        is_wrong_subtree = mode == "wrong_subtree"
        if not is_wrong_subtree:
            on_right_path += 1

        group_key = (
            predicted_label,
            predicted_annotation,
            reference_label,
            expected_abbrev,
            is_wrong_subtree,
            mode,
        )
        if group_key not in errors:
            errors[group_key] = ErrorRow(
                predicted_label=predicted_label,
                predicted_annotation=predicted_annotation,
                predicted_code=predicted_code,
                reference_label=reference_label,
                reference_annotation=expected_abbrev,
                reference_code=reference_code,
                wrong_subtree=is_wrong_subtree,
                failure_mode=mode,
            )
        errors[group_key].add_example(ref_key)

    # Strict-match columns are trivially on-the-right-path; add them
    # so the metric reflects "predictions whose code shares the
    # depth-3 prefix with the reference."
    on_right_path += strict_hits

    strict_pct = (100.0 * strict_hits / scored) if scored else 0.0
    delta = (
        None if ctx.prev_strict_pct is None
        else strict_pct - ctx.prev_strict_pct
    )

    mean_bel, mean_pl, mean_gap, mean_conf, needs_clar = _belief_diagnostics(
        state, column_qkeys,
    )

    iter_score = IterationScore(
        iteration=iteration,
        phase=phase,
        wall_clock_s=time.monotonic() - ctx.started_at_monotonic,
        scored=scored,
        missing_in_run=missing_in_run,
        unscorable=unscorable,
        strict_match=strict_hits,
        strict_pct=strict_pct,
        delta_strict_pct_vs_prev=delta,
        on_right_path=on_right_path,
        on_right_path_pct=(100.0 * on_right_path / scored) if scored else 0.0,
        parent_match=mode_counts.get("parent_instead_of_leaf", 0),
        child_match=mode_counts.get("child_instead_of_parent", 0),
        sibling_within_subtree=mode_counts.get("sibling_within_subtree", 0),
        wrong_subtree=mode_counts.get("wrong_subtree", 0),
        missing_prediction=mode_counts.get("missing_prediction", 0),
        hallucinated_annotation=mode_counts.get("hallucinated_annotation", 0),
        mean_belief=mean_bel,
        mean_plausibility=mean_pl,
        mean_gap=mean_gap,
        mean_conflict=mean_conf,
        needs_clarification_count=needs_clar,
        llm_calls_total=getattr(state, "llm_calls_total", 0),
        llm_attempts_total=getattr(state, "llm_attempts_total", 0),
        tokens_input=getattr(state, "tokens_input", 0),
        tokens_output=getattr(state, "tokens_output", 0),
        revisited_columns=revisited_count,
        disk_free_bytes=_disk_free_bytes(ctx.results_dir),
        failure_modes=dict(mode_counts),
    )

    ctx.prev_strict_pct = strict_pct
    ctx.trend.append(iter_score.to_dict())

    # Persist trend JSON every iteration so an aborted run still leaves
    # a parseable artifact.  Rewrites the whole file each tick — the
    # file is small (one record per iteration, capped at max_iterations).
    _write_trend(ctx)
    _write_error_tsv(ctx, iteration, errors)

    log.info(
        "Iteration %d scoring: strict=%.2f%% (Δ=%s), on_path=%.2f%%, "
        "%d errors across %d patterns",
        iteration,
        strict_pct,
        "n/a" if delta is None else f"{delta:+.2f}pp",
        iter_score.on_right_path_pct,
        sum(mode_counts.values()),
        len(errors),
    )
    return iter_score


def _write_trend(ctx: ScoringContext) -> None:
    path = ctx.results_dir / "scoring_trend.json"
    payload = {
        "schema_version": 1,
        "iterations": ctx.trend,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_error_tsv(
    ctx: ScoringContext,
    iteration: int,
    errors: dict[tuple, ErrorRow],
) -> None:
    if not errors:
        return
    path = ctx.results_dir / f"scoring_errors_iter{iteration}.tsv"
    rows = sorted(errors.values(), key=lambda r: (-r.count, r.failure_mode))
    header = "\t".join([
        "count",
        "predicted_label – predicted_annotation",
        "reference_label – reference_annotation",
        "wrong_subtree",
        "failure_mode",
        "predicted_code",
        "reference_code",
        "example_columns",
    ])
    path.write_text(header + "\n" + "\n".join(r.tsv_row() for r in rows) + "\n")


# ── Run-completion summary ───────────────────────────────────────────


def write_summary_report(
    ctx: ScoringContext,
    *,
    convergence_reason: str | None,
    run_id: str,
) -> None:
    """Emit ``scoring_summary.md`` at run completion.

    Produces a single-page operator-facing artifact summarizing the
    iteration trend (table), the deltas between consecutive iterations
    (highlights for >1pp swings either direction), and the final-
    iteration grouped error table (top 20 patterns).  This is the file
    the operator reads first when triaging a sweep run, so it earns its
    keep by being human-readable rather than machine-parseable.

    A no-op when scoring was disabled.
    """
    if not ctx.enabled or not ctx.trend:
        return

    path = ctx.results_dir / "scoring_summary.md"
    lines: list[str] = []
    lines.append(f"# Run {run_id} scoring summary")
    lines.append("")
    lines.append(
        f"- Iterations recorded: **{len(ctx.trend)}**"
    )
    lines.append(f"- Convergence reason: `{convergence_reason or 'unknown'}`")
    final = ctx.trend[-1]
    lines.append(
        f"- Final strict accuracy: **{final['strict_pct']}%** "
        f"({final['strict_match']}/{final['scored']})"
    )
    lines.append(
        f"- Final on-right-path accuracy: **{final['on_right_path_pct']}%** "
        f"({final['on_right_path']}/{final['scored']})"
    )
    lines.append(f"- Missing in run: {final['missing_in_run']}")
    lines.append(f"- Unscorable (vocab gap): {final['unscorable']}")
    lines.append("")

    # Vocab-mismatch report — only present when alignment is ad-hoc
    # (systemic mismatch raises before the context is built, so a
    # summary report never gets written in that case).  Surfaces the
    # exact mnemonics + affected GT keys so the operator can audit
    # offline (stale GT field? GT-construction hallucination? Bedrock
    # KV-cache artifact?  Stale vocab snapshot?) without rerunning.
    if ctx.alignment and ctx.alignment.classification == "ad_hoc_mismatch":
        lines.append("## Reference / vocabulary alignment — ad-hoc mismatch")
        lines.append("")
        lines.append(
            f"The reference cites **{len(ctx.alignment.unscorable_mnemonics)} "
            f"annotation(s)** ({ctx.alignment.missing_fraction * 100:.1f}% "
            f"of {ctx.alignment.cited_count} cited) that the loaded "
            f"vocabulary does not contain.  The "
            f"**{len(ctx.alignment.unscorable_columns)} affected GT "
            f"column(s)** were excluded from scoring; the rest of the "
            f"reference scored normally.  Investigate these offline as "
            f"either stale GT entries, GT-construction LLM hallucinations, "
            f"Bedrock KV-cache artifacts in the GT-building loop, or a "
            f"vocab snapshot lagging the upstream annotations table."
        )
        lines.append("")
        lines.append("| mnemonic | affected column(s) |")
        lines.append("|:---|:---|")
        # Group affected columns by mnemonic.
        by_mnem: dict[str, list[str]] = {}
        for k, v in ctx.reference.items():
            if v in ctx.alignment.unscorable_mnemonics:
                by_mnem.setdefault(v, []).append(k)
        for mnem in sorted(by_mnem):
            cols = sorted(by_mnem[mnem])
            shown = ", ".join(cols[:5])
            extra = "" if len(cols) <= 5 else f" (+{len(cols) - 5} more)"
            lines.append(f"| `{mnem}` | {shown}{extra} |")
        lines.append("")

    # Trend table
    lines.append("## Iteration trend")
    lines.append("")
    lines.append(
        "| iter | phase | strict % | Δ strict | on-path % | mean gap | "
        "mean K | needs_clar | LLM calls | revisited | disk free (GB) |"
    )
    lines.append(
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in ctx.trend:
        delta = (
            "—" if r["delta_strict_pct_vs_prev"] is None
            else f"{r['delta_strict_pct_vs_prev']:+.2f}"
        )
        disk_gb = r["disk_free_bytes"] / (1024 ** 3) if r["disk_free_bytes"] >= 0 else -1.0
        lines.append(
            f"| {r['iteration']} | {r['phase']} | {r['strict_pct']} | {delta} | "
            f"{r['on_right_path_pct']} | {r['mean_gap']} | {r['mean_conflict']} | "
            f"{r['needs_clarification_count']} | {r['llm_calls_total']} | "
            f"{r['revisited_columns']} | {disk_gb:.2f} |"
        )
    lines.append("")

    # Failure-mode breakdown for final iteration
    lines.append("## Final-iteration failure-mode counts")
    lines.append("")
    lines.append(
        "| mode | count | "
        "share of errors |"
    )
    lines.append("|:---|---:|---:|")
    total_errors = final["scored"] - final["strict_match"]
    # IterationScore stores the parent / child counts under the
    # ancestor-relationship names (parent_match = predicted is parent,
    # child_match = predicted is child of reference); display labels
    # use the more operator-friendly "X_instead_of_Y" phrasing.
    _mode_to_field = {
        "wrong_subtree": "wrong_subtree",
        "sibling_within_subtree": "sibling_within_subtree",
        "parent_instead_of_leaf": "parent_match",
        "child_instead_of_parent": "child_match",
        "hallucinated_annotation": "hallucinated_annotation",
        "missing_prediction": "missing_prediction",
    }
    for display_mode, field_name in _mode_to_field.items():
        n = final.get(field_name, 0)
        share = (100.0 * n / total_errors) if total_errors else 0.0
        lines.append(f"| {display_mode} | {n} | {share:.1f}% |")
    lines.append("")

    # Top error patterns from the final iteration's TSV
    err_path = ctx.results_dir / f"scoring_errors_iter{final['iteration']}.tsv"
    if err_path.exists():
        lines.append("## Top failure patterns (final iteration)")
        lines.append("")
        lines.append(
            "| count | predicted → reference | wrong subtree | "
            "failure mode | examples |"
        )
        lines.append("|---:|:---|:---:|:---|:---|")
        rows = err_path.read_text().splitlines()[1:]  # skip header
        for line in rows[:20]:
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            count, pred_pair, ref_pair, wrong, mode, _pcode, _rcode, examples = parts
            ex = (examples[:80] + "…") if len(examples) > 80 else examples
            lines.append(
                f"| {count} | `{pred_pair}` → `{ref_pair}` | {wrong} | "
                f"{mode} | {ex} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n")
    log.info("Wrote scoring summary: %s", path)


# ── Disk-space guard ─────────────────────────────────────────────────


@dataclass
class DiskGuardConfig:
    """Tunables for the disk-space guard.

    Defaults follow the design discussion:
      * ``headroom_multiplier``: 1.25 on top of mean+2σ.
      * ``bootstrap_floor_bytes``: 200 MiB when run history is thin.
      * ``min_runs_for_stats``: 3 historical runs needed before we trust
        run-size statistics.
    """

    headroom_multiplier: float = 1.25
    bootstrap_floor_bytes: int = 200 * 1024 * 1024
    min_runs_for_stats: int = MIN_HISTORICAL_RUNS_FOR_STATS


@dataclass
class DiskGuardReading:
    """Snapshot of one disk-guard check.

    The disk-guard decision is "do we have enough free space to finish
    one more pipeline run (or one more iteration), given the variance
    of recent history?"  Capture the inputs so failures can be triaged
    without re-reading the filesystem.
    """

    free_bytes: int
    required_bytes: int
    historical_runs_sampled: int
    historical_mean_bytes: float
    historical_stdev_bytes: float
    headroom_bytes: int
    fits: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "free_bytes": self.free_bytes,
            "required_bytes": self.required_bytes,
            "historical_runs_sampled": self.historical_runs_sampled,
            "historical_mean_bytes": round(self.historical_mean_bytes, 1),
            "historical_stdev_bytes": round(self.historical_stdev_bytes, 1),
            "headroom_bytes": self.headroom_bytes,
            "fits": self.fits,
        }


def _dir_size_bytes(path: Path) -> int:
    """Recursively sum file sizes under ``path``.  Symlinks are skipped."""
    total = 0
    if not path.is_dir():
        return 0
    for child in path.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
        except OSError:
            continue
    return total


_RUN_ID_RE = re.compile(r"^[0-9a-f]{6,}$")


def _completed_run_dirs(results_root: Path) -> list[Path]:
    """Return run directories that look complete.

    Heuristic: the directory name is a hex-ish run_id AND both
    ``settings_snapshot.json`` and ``classifications.json`` are present
    (the two artifacts that only land at run completion).
    """
    if not results_root.is_dir():
        return []
    out: list[Path] = []
    for child in results_root.iterdir():
        if not child.is_dir() or not _RUN_ID_RE.match(child.name):
            continue
        if (
            (child / "settings_snapshot.json").exists()
            and (child / "classifications.json").exists()
        ):
            out.append(child)
    return out


def estimate_required_bytes(
    results_root: Path,
    *,
    config: DiskGuardConfig | None = None,
) -> DiskGuardReading:
    """Project how many free bytes the next run will need.

    Reads completed-run sizes from ``results_root``, computes
    ``mean + 2σ × headroom_multiplier``, and falls back to
    ``bootstrap_floor_bytes`` when fewer than ``min_runs_for_stats``
    completed runs exist.
    """
    cfg = config or DiskGuardConfig()
    completed = _completed_run_dirs(results_root)
    sizes = [_dir_size_bytes(d) for d in completed]
    sizes = [s for s in sizes if s > 0]

    if len(sizes) < cfg.min_runs_for_stats:
        # Insufficient history — use the bootstrap floor with the
        # headroom multiplier on top.  ``stdev`` is 0 because we don't
        # know it yet; the operator sees that and understands the
        # estimate is conservative-by-floor rather than data-driven.
        free = _disk_free_bytes(results_root)
        required = int(cfg.bootstrap_floor_bytes * cfg.headroom_multiplier)
        return DiskGuardReading(
            free_bytes=free,
            required_bytes=required,
            historical_runs_sampled=len(sizes),
            historical_mean_bytes=float(sum(sizes) / len(sizes)) if sizes else 0.0,
            historical_stdev_bytes=0.0,
            headroom_bytes=max(0, free - required),
            fits=free >= required,
        )

    mean = sum(sizes) / len(sizes)
    variance = sum((s - mean) ** 2 for s in sizes) / len(sizes)
    stdev = variance ** 0.5
    raw = mean + 2 * stdev
    required = int(raw * cfg.headroom_multiplier)
    free = _disk_free_bytes(results_root)
    return DiskGuardReading(
        free_bytes=free,
        required_bytes=required,
        historical_runs_sampled=len(sizes),
        historical_mean_bytes=mean,
        historical_stdev_bytes=stdev,
        headroom_bytes=max(0, free - required),
        fits=free >= required,
    )


def assert_disk_capacity(
    results_root: Path,
    *,
    config: DiskGuardConfig | None = None,
    context: str = "run start",
) -> DiskGuardReading:
    """Raise ``RuntimeError`` if projected free space is insufficient.

    Returns the ``DiskGuardReading`` on success so callers can log /
    persist it.  This is the function the pipeline calls at run start
    and the bootstrap loop calls at each iteration boundary.
    """
    reading = estimate_required_bytes(results_root, config=config)
    if reading.fits:
        log.info(
            "disk_guard (%s): required=%s, free=%s (mean+2σ of %d run(s), "
            "headroom=%s)",
            context,
            _human_bytes(reading.required_bytes),
            _human_bytes(reading.free_bytes),
            reading.historical_runs_sampled,
            _human_bytes(reading.headroom_bytes),
        )
        return reading

    raise RuntimeError(
        f"disk_guard ({context}): projected free space {_human_bytes(reading.free_bytes)} "
        f"is below the required threshold of {_human_bytes(reading.required_bytes)} "
        f"(mean+2σ of {reading.historical_runs_sampled} historical run(s), "
        f"headroom multiplier 1.25).  Free up space under "
        f"{results_root} or set a different ATELIER_BUILD_DIR before "
        f"continuing — running through this guard risks a half-written "
        f"classifications.json and a corrupt run artifact set."
    )


def _human_bytes(n: int | float) -> str:
    """Format a byte count as a short human-readable string."""
    if n < 0:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    f = float(n)
    for unit in units:
        if f < 1024.0:
            return f"{f:.1f}{unit}"
        f /= 1024.0
    return f"{f:.1f}PiB"
