"""atelier.optimize.svm — factorized NHSVM training + refinement + gate.

Modules:
    encoder          — canonical ModernBERT encoder (encode_modernbert),
                       shared by runtime inference + offline training
    audit_reference  — confidence-tier weights from agent_mediated/audit.json
    reference        — load reference rows + join with per-row weights
    kfold            — stratified k-fold partitioning over reference labels
    augment          — synth augmentation for under-represented codes
    train            — Phase D: reference-primary k-fold training dispatcher
    refine           — Phase E: metrology-driven refinement loop
    gate             — Gate B: cosine-SVM mutual-affirmation
    promote          — head serialization + registry promotion utilities

The CLI surface (`scripts/reflect_nhsvm_eval_shap_v2.py`,
`scripts/svm_cosine_uplift_gate.py`, etc.) imports from here.
"""
