# Classify: idempotent no-op, then unpause

Pause is the wrong control. If Ægir’s Atlas `rdbms_table` set under
`footprint.*` has not grown, `ClassificationFlow` should **succeed
without reclassification** (no LIGHT). If tables appear, classify the
delta and merge.

Today `probe` is vocab/encoder only. Gateway already has
`new_table_count`. Atlas already has `read_tables`.

Work: scope fingerprint → Δ=0 no-op → prove one K8s Metaflow run →
`enabled=True` on `atelier_sdg_classify`. Do not unpause first.

Pointer: Gaius `docs/scratch/2026-09-21/034500_idempotent_not_paused_deck.md`.
