"""atelier.optimize.corpus — synthetic corpus generation + audit + metrology.

Modules:
    generate         — produce synth_rows.jsonl from generators_v1.py
    metrology        — per-code fidelity / separability / spread + k-robustness
    audit_coverage   — generator-coverage audit (which codes are gap)

The synth corpus is a load-bearing artifact (see
feedback_synth_corpus_is_load_bearing memory).  Functions in this
package APPEND to or measure the existing corpus; they NEVER provide
"regenerate from blank state" entry points.
"""
