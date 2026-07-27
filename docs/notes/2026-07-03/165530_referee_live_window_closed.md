# Referee served + verified; pin advanced to 436fe94; window released

**Date:** 2026-07-03 (follows `141219_engine_landed.md`)
**Context:** Ægir's replies note (`aegir/docs/scratch/2026-07-03/143050_…`)
accepted all asks, opened a GPU window (their engine down), and advanced the
world: corpora `436fe94`, preview re-cut with `.key` separation,
`holdout_partition` in manifests, `zndx.engine.v1` shared service identity
landing in a new `signals-protocol` repo.

## Referee smoke (task 7 CLOSED)

- First attempt failed fast + clean: NVIDIA repo ships
  `configuration_nemotron_h.py` → needs `--trust-remote-code`. Added as a
  `trust_remote_code` capability key (HOCON) → serve flag.
- The failure exposed a real relaunch bug: `_next_gpu` advanced on every
  `_launch`, so a retry would walk off the device range. Fixed with sticky
  per-capability GPU slots + ports; relaunch verified live (reload reused
  GPUs 0–3).
- **Cold-serve: 109s** to healthy (proof-of-progress transitions logged).
  1-token answer + retained reasoning trace via gRPC.
- **Structured output**: top-level `guided_json` is SILENTLY IGNORED by
  vLLM 0.19.0 (required fields missing) — switched the manager to
  `response_format: {type: json_schema, strict: true}`, which enforces
  fully. Verified end-to-end through our proto's `json_schema` field:
  curation-shaped classification returned all required keys, right code,
  0.95 confidence, ~2.3s/call hot. **Correction posted to Ægir** — their
  landed passthrough uses `guided_json` and will no-op on the same venv.
- Engine shut down after the smoke; killpg released all TP workers
  (GPUs back to 394 MiB baseline); shared-dir lease released; done-note
  posted (running-observations §9) so Ægir can restart.

## Pin + layout updates (from the replies note)

- `external/sdg-corpora` → **436fe94** (clean artifact: 10,570 individuals,
  0 real-particulars, 0 withheld clashes). Vocab still 944.
- `resolve_reference_path()` — reads `<release>.key/reference.parquet`
  first (key separation LANDED their side; blind-integrity now structural),
  legacy in-dir fallback. Loader + golden emitter updated.
- Guard: `holdout_partition` presence required; `scored=True` refuses
  `preview*` cuts (the RWKV eval-design ask, adopted their side as
  manifest field). Verified live against the re-cut preview both ways.
- Golden emit → `just score-atelier` re-validated post-recut: 2033 rows,
  coverage 1.0, hierarchical 1.0.
- Passthrough rung = **built-in null control** (their §2.3: six
  register-invariant structural tokens — `role`, `notes`, `event_count`,
  `since`, + 2 composition collisions; none are vocabulary labels).
  Ablation expectation: name_match lift ≈0 there, by construction — a
  cross-side consistency check.

## Deferred / blocked

- `signals-protocol` submodule add (for `zndx.engine.v1`): permission
  classifier declined the autonomous add (repo named only in Ægir's note,
  not by RH) — awaiting RH's explicit go, then register the shared service
  beside our native one.
- Classify-LLM-channel default still points at its existing backend;
  switching to local `instruct` is one HOCON override
  (`classify.llm.base_url=http://127.0.0.1:8200/v1`, `model="instruct"`)
  once the instruct endpoint has a resident window.
- Supervisor/staged-readiness hardening pass before multi-day optimize runs.

Tests: 129 passed (engine 10 + classify/governance). Next: task 8 — SDG
working-set builder + blind curation loop on `referee` (Arm T stage 2),
Arm I cold stack alongside.
