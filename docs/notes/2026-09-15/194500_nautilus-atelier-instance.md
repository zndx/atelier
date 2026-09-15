# Atelier Nautilus instance (federated, not in-process)

Shared `nautilus.rs` + `config/supervision/atelier.textproto`. Engine
serves `EngineSupervision` on `:50251` (observe-only, ring replay).
Resident bind `:50261`. In-process `NautilusWatcher` default off.

Classification machine `sdg_classify` + objective `classification_sdg`
score against pinned sdg-corpora SKOS (`just sdg-classify-verify`).
