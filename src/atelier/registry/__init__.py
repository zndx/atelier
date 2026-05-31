"""atelier.registry — unified model-version DAO over registry tables.

Modules (subpackages):
    nhsvm_head  — factorized NHSVM head registry (taxonomy_id, encoder, status)
    taxonomy    — Qdrant collection registry (re-exports from atelier.db.dao
                  pattern; consolidates the status='current'/'stale'/'archived'
                  semantics)

The registry is the bridge between the optimize/ side (which produces
versioned models) and the classify/ side (which consumes the currently-
promoted model).  Both registries share the same status semantics so a
future `compound_model_registry` (for CAI continuous-classification
endpoints) can join them as a third sibling without refactoring.
"""
