"""Ontology-to-taxonomy alignment for the per-vocabulary SVM.

The BFO/CCO synth corpus (``synth_generators.GENERATORS``) is keyed
by ICE.* leaf codes.  The pipeline at runtime fuses evidence in the
operator's **user-taxonomy** code space (numeric dot codes from a
customer ``annotations`` table, ICE.* itself for the OOTB sample,
an enterprise schema, …).  Bridging the two requires a mapping from
each synth-generator's ICE leaf to the user code that best
represents that concept in the operator's vocabulary.

This module computes that mapping — the alignment — via subsumption
prediction over sentence-transformer embeddings.  Each ICE leaf's
concept signature (humanized name + generator samples) is embedded
alongside enriched annotation payloads from the operator's Qdrant
collection; cosine similarity identifies the best user-code match.
Result is cached on disk under a stable (ICE leaves, user codes,
embedding model, method) key, reloaded on subsequent vocab-loads.

The alignment is available for diagnostic and validation use.
The pipeline now generates user-code-labeled corpora directly from
enrichment payloads (see ``enrichment_loader.py`` +
``synth_registry.from_enrichment_payloads``), so ICE.* → user-code
alignment is no longer in the training hot path.  Inference
(``predict_svm`` → ``svm_to_mass``) reads user-code-keyed proba
directly, so there is no per-column translation step and no LLM in
the SVM's classify-time critical path.

═══════════════════════════════════════════════════════════════════
NOTE ON THEORETICAL RIGOR — DST source independence
═══════════════════════════════════════════════════════════════════

Under Dempster-Shafer fusion, distinct sources must come from
independent evidence (Dempster 1968; Shafer 1976 §11.3).  Denoeux
2008 generalizes to a non-distinct regime where sources share
provenance and Dempster's rule double-counts overlap.

This alignment achieves *weakly non-distinct* evidence via
enrichment-mediated subsumption prediction:

  • The SVM's **features** stay independent — TF-IDF char/word n-grams
    over column metadata, no embedding-stack dependency, no LLM
    column-vote dependency.
  • The SVM's **training labels** are user codes derived from
    (synth-generator-output, alignment[ICE]) pairs.  The alignment
    is computed once at vocab-load time via sentence-transformer
    embeddings — structurally independent of the runtime LLM.
  • The **weak non-distinctness** is via shared enrichment-LLM
    upstream: the enriched annotation payloads (prototype_values,
    name_hints, descriptions) were generated offline by an LLM.
    This is the same structural dependency the late-interaction
    cosine source carries — the enrichment is a one-time offline
    artifact, not a per-column inference coupling.

The discount calibration ``classify.discounts.svm = 0.22`` reflects
this: slightly above cosine's 0.20 (because subsumption prediction
is a single decision per ICE code, structurally more brittle than
per-column cosine evidence), but materially below the prior
LLM-mediated alignment's 0.30 (which compensated for shared LLM
weights between alignment and runtime classification).

Historical note: the predecessor LLM-mediated approach (removed in
the P7 subsumption-alignment intervention) called
``LLMBackend.classify_batch`` at vocab-load time to route synthetic
column samples through the runtime LLM.  That introduced a
vocabulary-level shared error mode where alignment mistakes and
runtime classification mistakes were correlated through shared
model weights.  The subsumption prediction approach eliminates that
correlation by using a structurally different model family (BERT
embeddings vs autoregressive LLM) for the alignment decision.

═══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atelier.classify.taxonomy import HierarchicalCategorySet

logger = logging.getLogger(__name__)


_DEFAULT_CACHE_DIR = Path("build/cache/alignment")


def build_alignment(
    category_set: "HierarchicalCategorySet",
    llm_backend=None,
    system_prompt: str | None = None,
    *,
    model_name: str | None = None,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
    cfg=None,
    taxonomy_id: str | None = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    score_threshold: float = 0.35,
) -> dict[str, str]:
    """Build (or load from cache) the ICE.* → user-taxonomy alignment.

    Delegates to :func:`subsumption_alignment.build_alignment_via_subsumption`
    which computes cosine-similarity alignment between ICE concept
    signatures and enriched user-vocab annotations from Qdrant.

    The ``llm_backend``, ``system_prompt``, and ``model_name`` parameters
    are retained for call-site compatibility but are no longer used —
    the alignment is now computed via sentence-transformer embeddings,
    not LLM classification.  These parameters will be removed in a
    future cleanup pass.

    Returns:
        dict mapping ICE.* leaf codes to user taxonomy codes.
        Empty dict on failure (SVM contributes vacuous mass).
    """
    if llm_backend is not None:
        logger.debug(
            "ontology_alignment: llm_backend parameter is deprecated and ignored; "
            "alignment now uses subsumption prediction via sentence-transformer embeddings"
        )

    from atelier.classify.subsumption_alignment import build_alignment_via_subsumption

    # Read score_threshold from cfg if available
    if cfg is not None:
        cfg_threshold = getattr(cfg, "classify_subsumption_score_threshold", None)
        if cfg_threshold is not None:
            score_threshold = float(cfg_threshold)
        cfg_embedding = getattr(cfg, "classify_embedding_model", None)
        if cfg_embedding:
            embedding_model = cfg_embedding

    return build_alignment_via_subsumption(
        category_set,
        cfg=cfg,
        taxonomy_id=taxonomy_id,
        embedding_model=embedding_model,
        score_threshold=score_threshold,
        cache_dir=cache_dir,
    )
