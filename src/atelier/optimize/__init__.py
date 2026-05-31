"""atelier.optimize — cycle orchestration for the `just optimize` channels.

Subpackages:
    svm/      — factorized NHSVM training, k-fold reference-primary, gate, refine
    cosine/   — semantic optimization (GEPA loop) of the ColBERT/Qdrant collection
    corpus/   — synthetic corpus generation, metrology, coverage audit

The cycle-orchestration logic lives here; `scripts/` retains thin CLI
shims that dispatch into these modules.  This package is the home of
algorithmic code that the `just optimize` commands and the runtime
classification pipeline both consume.
"""
