# Session note — SOTAB v2 coverage strategy

## What landed

- Verified canonical CCO labels for the ICE trichotomy (`cco:ont00000958`,
  `ont00000686`, `ont00000853`, `ont00000965`) against
  `CommonCoreOntology/CommonCoreOntologies@master`.
- Reconciled `atelier-vocab.ttl`, `ontology/README.md`, and
  `054600_cco-module-inventory.md` to cite canonical labels.  Initially
  documented the `skos:altLabel "Directive ICE"`, then purged the legacy
  altLabel mention entirely (operator preference: prototype phase, no
  reason to carry the legacy term forward).
- Verified the SOTAB v2 Schema.org CTA label space: **82 distinct labels**
  across all splits (training, validation, test, three robustness test
  sets).  Aegir's `_LABEL_DIMS["sotab"] = 91` is stale; should be 82.
- Wrote `docs/src/architecture/sotab-coverage.md` — the strategy doc
  Ægir's M2/M3 implementation will consume.
- Updated `docs/src/SUMMARY.md` to include the new doc.

## Coverage at a glance

| Bucket | Count | Action |
|---|---|---|
| Direct hits with current 20 mappings | 14 | None |
| Subsumption-reachable (under existing CCO grounding) | ~20 | Tier-B subclass plumbing |
| Genuinely missing | ~48 | Tier-A measurement zoo + Tier-C Product/JobPosting/economics |

## Aegir touchpoints (informative — don't implement here)

- `~/local/src/zndx/aegir/scripts/download_sotab.py` — already wired.
- `~/local/src/zndx/aegir/src/aegir/data/table_dataset.py` — `_LABEL_DIMS["sotab"]` needs `82` not `91`.
- `~/local/src/zndx/aegir/scripts/sotab_diagnostic.py` — extend to surface
  per-tier coverage of predictions.
- M2 roadmap entry in `aegir/README.md` already names "ontology editor with
  Postgres write paths, per-class F1 bars" — strategy doc fits cleanly.

## Pre-existing aegir state worth knowing

- `aegir/docs/notes/2026-04-19/234700_sotab_diagnostic_representation_collapse.md`
  documents that `outputs/best_model.pt` (sotab-small, 3 epochs) collapsed
  to a single embedding point.  This is a model issue (RWKV time-mix /
  dynamic-chunking interaction), orthogonal to vocabulary coverage.
- Aegir's `_LABEL_DIMS` knows `sotab-dbp`, `sotab-dbp-re` (DBpedia variants
  101 and 53 labels respectively).  DBpedia coverage is a separate
  follow-on doc once Schema.org tiers land.

## Open questions for next session

- Should `sotab_label_map.json` live in atelier or aegir?  ~~Strategy doc
  puts it under `src/atelier/classify/ontology/`~~ → **resolved**:
  ontology ownership moves to Ægir entirely; the map lives there.
- Should we extend coverage to SOTAB DBpedia CTA (101 labels) in the same
  doc or a sibling doc?  DBpedia label set has overlap with our existing
  `dbo:*` mappings (15 today), suggesting a fourth "Tier-D" or a parallel
  document.  This work also lands in Ægir.
- Synth pipeline: ~~do we generate from atelier into a flat dump that
  aegir consumes by path, or do we expose a thin atelier API~~ →
  **resolved**: synth migrates to Ægir.  Open question instead: how does
  Atelier's BDD/pytest get synth output during local dev — sibling-repo
  path, vendored snapshot, or thin client?

## Directional shift (added 2026-05-09 mid-session)

**Ontology / vocab / synth move to Ægir going forward.**  The label
space conditions model pre-training directly, so it lives next to the
model.  Atelier becomes the consumer:

- Loads H-Net/RWKV checkpoints from Ægir as DST evidence sources.
- Loads SVMs trained on Ægir-curated datasets.
- Keeps DST fusion, belief/plausibility logic, FSM, gateway, UI.
- The classification pipeline's *vocabulary* migrates; its *machinery*
  stays.

Strategy doc updated to reflect new ownership flow.  Memory entry
saved at `project_ontology_migration_to_aegir.md`.  No code changes in
this session — vocab continues operational in atelier-vocab.ttl until
the migration is scheduled.
