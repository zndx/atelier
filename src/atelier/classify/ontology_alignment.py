# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Ontology-to-taxonomy alignment for the per-vocabulary SVM.

The BFO/CCO synth corpus (``synth_generators.GENERATORS``) is keyed
by ICE.* leaf codes.  The pipeline at runtime fuses evidence in the
operator's **user-taxonomy** code space (numeric dot codes from a
customer ``annotations`` table, ICE.* itself for the OOTB sample,
an enterprise schema, …).  Bridging the two requires a mapping from
each synth-generator's ICE leaf to the user code that best
represents that concept in the operator's vocabulary.

This module computes that mapping — the alignment — by reusing the
same ``LLMBackend.classify_batch`` interface the classifier already
uses.  Each ICE leaf becomes a synthetic column sample (its
generator's outputs are the values), and the LLM classifies the
batch into the user vocabulary just as it would any real column.
Result is cached on disk under a stable (ICE leaves, user codes,
model) key, reloaded on subsequent vocab-loads.

The alignment is consumed at **training time**, not runtime:
``ml_train.train_svm_for_vocab`` projects the synth corpus's ICE
labels through the alignment and trains a per-vocabulary SVM whose
output classes ARE user codes.  Inference (``predict_svm`` →
``svm_to_mass``) reads user-code-keyed proba directly, so there is
no per-column translation step and no LLM in the SVM's classify-time
critical path.

═══════════════════════════════════════════════════════════════════
NOTE ON THEORETICAL RIGOR — please read before claiming independence
═══════════════════════════════════════════════════════════════════

Under Dempster-Shafer fusion, distinct sources must come from
independent evidence (Dempster 1968; Shafer 1976 §11.3).  Denoeux
2008 generalizes to a non-distinct regime where sources share
provenance and Dempster's rule double-counts overlap.

This alignment achieves *weakly non-distinct* evidence:

  • The SVM's **features** stay independent — TF-IDF char/word n-grams
    over column metadata, no embedding-stack dependency, no LLM
    column-vote dependency.
  • The SVM's **training labels** are user codes derived from
    (synth-generator-output, alignment[ICE]) pairs.  The alignment
    LLM call happens once at vocab-load time, not per-column at
    classify time.
  • The SVM's **prediction-to-frame-element mapping** is now baked
    into the model weights at training time.  If the LLM
    systematically misaligns ICE.X → user-Y when user-Z is correct,
    the SVM trained on that alignment will tend to predict user-Y
    on ICE.X-shaped columns at inference, and so will the runtime
    LLM sweep — they'll wrongly agree on those columns.

That shared error mode is real but is **vocabulary-level, not
column-level** — it doesn't grow with corpus size and it doesn't
correlate with run-time evidence streams.  The discount calibration
``classify.discounts.svm`` compensates; under this design the
correct value is materially lower than the M9-era 0.55 (which
compensated for *per-column* label-copying via the now-excised
``train_svm_on_frontier_labels`` retrain).

Strict independence would require an alignment source that doesn't
share weights or training data with the runtime LLM.  Future work:

  • Overwatch as alignment mediator — review the proposed mapping
    before training proceeds (Phase 2).
  • BM25 over (ICE labels + descriptions, user-vocab labels +
    descriptions) with a transformer reranker.  More plumbing,
    truly independent.
  • Hand-curated alignment per vocabulary (the predecessor approach
    — see deleted ``meta_tagging_overlay.py`` for the M-T example).
    Truly independent of the runtime LLM but does not scale.

For the current scope, the LLM-mediated alignment is the chosen
trade-off: keeps the refactor tractable, restores user-taxonomy SVM
evidence (the SVM was contributing nothing pre-alignment), and
preserves *most* of the independence the M9 retrain destroyed.
Don't read this module's success as confirmation that the SVM is a
strictly independent fifth source — it's a mostly-independent source
with an honest, calibrated discount.

═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atelier.classify.llm_backend import LLMBackend
    from atelier.classify.taxonomy import HierarchicalCategorySet

logger = logging.getLogger(__name__)


_DEFAULT_CACHE_DIR = Path("build/cache/alignment")
_SAMPLES_PER_ICE_LEAF = 5


def _alignment_cache_key(
    ice_leaves: list[str],
    user_codes: list[str],
    model_name: str,
) -> str:
    """Stable sha256 over (ICE leaves, user codes, model) — cache identity."""
    payload = json.dumps(
        {
            "ice_leaves": sorted(ice_leaves),
            "user_codes": sorted(user_codes),
            "model": model_name,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _ice_to_humanized_name(ice_code: str) -> str:
    """Convert ICE.SENSITIVE.PID.IDENTITY.GOVID.SSN → 'ssn (govid identity pid)'.

    The classifier prompt benefits from a column-name-shaped string
    rather than a raw dot-code so it routes through the same name-
    matching heuristics it uses on real columns.
    """
    parts = ice_code.split(".")
    # Drop the universal-substrate prefix ("ICE", "SENSITIVE"|"OPERATIONAL"
    # etc.) so the LLM sees only the semantically-loaded tail.
    tail = parts[2:] if len(parts) > 2 else parts
    if not tail:
        return ice_code
    leaf = tail[-1].lower()
    context = " ".join(p.lower() for p in tail[:-1])
    return f"{leaf} ({context})" if context else leaf


def _build_synthetic_samples(ice_leaves: list[str]) -> list:
    """Create one fake ColumnSample per ICE leaf for the alignment batch.

    Reuses the synth generators (the same source of truth that produces
    ``data/synth/`` for SVM/CatBoost training) so the alignment LLM sees
    the same value distribution the SVM was trained against.  Generator
    failures are tolerated — those ICE codes drop out of the alignment
    and their predictions stay unmapped (silently no-op via
    ``svm_to_mass`` frame check).
    """
    import random
    from atelier.classify.sampler import ColumnSample
    from atelier.classify.synth_generators import GENERATORS

    rng = random.Random(0xA11167)  # stable seed — alignment cache stays valid
    samples: list = []
    for ice in ice_leaves:
        gen = GENERATORS.get(ice)
        if gen is None:
            continue
        try:
            values = [str(gen(rng)) for _ in range(_SAMPLES_PER_ICE_LEAF)]
        except Exception as exc:
            logger.debug("alignment: generator %s raised — skipping: %s", ice, exc)
            continue
        samples.append(
            ColumnSample(
                name=_ice_to_humanized_name(ice),
                column_type="STRING",
                values=values,
                table_name="_synth_alignment",
            ),
        )
    return samples


def _ice_codes_in_canonical_order(ice_leaves: list[str]) -> list[str]:
    """Filter to ICE leaves that have a generator and return them stably sorted.

    Synth generators are the source-of-truth for "things the SVM was
    trained on."  Any ICE leaf without a generator can't be predicted
    by the SVM in the first place, so leaving it out of the alignment
    is correct — there's nothing to translate.
    """
    from atelier.classify.synth_generators import GENERATORS
    return sorted(c for c in ice_leaves if c in GENERATORS)


def _parse_alignment_response(
    response,
    ice_leaves_by_humanized: dict[str, str],
) -> dict[str, str]:
    """Map ``LLMResponse.classifications`` back to {ice_code: user_code}.

    The classifier emits one ``ColumnClassification`` per humanized-name
    column we synthesized; we invert the humanized→ICE mapping to land
    the user-code on the original ICE key.  Vacuous (None / empty)
    user codes are dropped — the SVM should not contribute mass for
    ICE concepts the LLM cannot place in the user vocabulary.
    """
    out: dict[str, str] = {}
    classifications = getattr(response, "classifications", None) or []
    for cls in classifications:
        humanized = getattr(cls, "column_name", None)
        user_code = getattr(cls, "category_code", None)
        if not humanized or not user_code:
            continue
        ice = ice_leaves_by_humanized.get(humanized)
        if ice is None:
            continue
        out[ice] = str(user_code)
    return out


def build_alignment(
    category_set: HierarchicalCategorySet,
    llm_backend: LLMBackend,
    system_prompt: str,
    *,
    model_name: str,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> dict[str, str]:
    """Build (or load from cache) the ICE.* → user-taxonomy alignment.

    The alignment is a function of three things — the ICE leaf set
    (from ``synth_generators.GENERATORS``), the user vocab, and the
    model — and is cached on disk under that triple's stable key, so
    a re-run on the same vocab against the same model is a single
    file read.  When the cache is missing or the build fails, returns
    the empty dict; callers should treat the empty case as "no
    translation" — the SVM falls back to its pre-alignment behavior
    of contributing nothing on user-taxonomy runs.
    """
    user_codes = sorted(getattr(category_set, "leaf_codes", set()))
    ice_leaves = _ice_codes_in_canonical_order(list(_synth_generator_keys()))
    if not user_codes or not ice_leaves:
        return {}

    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _alignment_cache_key(ice_leaves, user_codes, model_name)
    cache_path = cache_dir / f"{key}.json"

    if cache_path.exists():
        try:
            blob = json.loads(cache_path.read_text())
            mapping = blob.get("alignment", {})
            if isinstance(mapping, dict):
                logger.info(
                    "ontology_alignment: loaded %d entries from cache (%s)",
                    len(mapping), cache_path,
                )
                return {str(k): str(v) for k, v in mapping.items() if v}
        except Exception as exc:
            logger.warning(
                "ontology_alignment: cache read failed at %s: %s — rebuilding",
                cache_path, exc,
            )

    samples = _build_synthetic_samples(ice_leaves)
    if not samples:
        return {}
    humanized_to_ice = {s.name: ice for s, ice in zip(samples, ice_leaves) if s}

    try:
        response = llm_backend.classify_batch(
            samples=samples,
            system_prompt=system_prompt,
            table_name="_synth_alignment",
        )
    except Exception as exc:
        logger.warning(
            "ontology_alignment: LLM call failed — proceeding without "
            "alignment, SVM evidence will be vacuous on this run: %s", exc,
        )
        return {}

    alignment = _parse_alignment_response(response, humanized_to_ice)

    try:
        cache_path.write_text(json.dumps({
            "alignment": alignment,
            "ice_leaf_count": len(ice_leaves),
            "user_code_count": len(user_codes),
            "model": model_name,
            "mapped": len(alignment),
        }, indent=2) + "\n")
        logger.info(
            "ontology_alignment: built %d/%d entries via %s, cached at %s",
            len(alignment), len(ice_leaves), model_name, cache_path,
        )
    except Exception as exc:
        logger.warning("ontology_alignment: cache write failed: %s", exc)

    return alignment


def _synth_generator_keys():
    """Lazy import to avoid circular module load at module-import time."""
    from atelier.classify.synth_generators import GENERATORS
    return GENERATORS.keys()
