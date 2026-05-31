#!/usr/bin/env python
"""scripts/audit_reference_quality.py — CLI shim.

Algorithmic core: ``atelier.optimize.svm.audit_reference``.

This script preserves the CLI surface for back-compatibility with the
orchestrator (``scripts/run_corpus_expansion_pipeline.sh`` Phase 0.5)
and operator habits.  All logic lives in the atelier package.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atelier.optimize.svm.audit_reference import main

if __name__ == "__main__":
    raise SystemExit(main())
