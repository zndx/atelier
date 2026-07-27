# Engine landed — local inference capability, live-verified against a co-tenant

**Date:** 2026-07-03
**Spec:** `140044_engine_inference_capability_spec.md` (implemented as written,
one addition: shared-lock-dir guard default `/tmp/zndx-gpu-leases`).

## What landed

`src/atelier/engine/` — minimal mirror of Ægir's engine (which mirrors Gaius):
- `proto/atelier_engine.proto` + generated stubs — Complete / EnsureEndpoint /
  EngineStatus; our `json_schema` field (guided-JSON) added to CompleteRequest
  (proposed upstream via the running note).
- `config.py` — HOCON `engine.*` subtree (divergence from Ægir's pure-env, per
  our config directive); capability registry: `instruct` =
  Qwen/Qwen3.6-35B-A3B-FP8 (TP=4, hermes tool parser), `referee` =
  nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (TP=4, 32K ctx, no tool parser —
  guided-JSON path). Path probes resolve the proven cu129 vLLM venv
  (`agent-evals/.venv-vllm`, vLLM 0.19.0 — NemotronH + Qwen3.6-MoE archs
  verified registered) and Ægir's `build/cuda-driver-libs`.
- `vllm_manager.py` — Ægir's manager + three grafts: proof-of-progress startup
  log scanning (fatal OOM/CUDA patterns short-circuit; progress transitions
  logged), per-endpoint GPU lease claimed BEFORE launch, `guided_json`
  support in complete(). Foreign-venv isolation + `start_new_session`/killpg
  preserved verbatim (TP-worker VRAM-leak defense).
- `gpu_guard.py` — filelock + owner-json (shared lock dir) + authoritative
  nvidia-smi compute-apps probe.
- `server.py` / `client.py` / `__main__.py`; just recipes `engine-serve`,
  `engine-ping [capability]`, `engine-status`.
- `config/base.conf` `engine { … }` block; `tests/engine/test_engine.py`
  (10 tests). Full sweep: 126 passed; preflight 7/7.

## Live verification (on the shared 6×4090 host)

1. **Cross-engine capability request** — via Ægir's own client to their live
   engine (:50151): Qwen3.6-35B-A3B-FP8 answered in 1.7s, thinking trace
   retained separately. First Atelier-session → Ægir-engine inference.
2. **Guard refusal** — our engine (started on :50251) refused
   `instruct` while Ægir's TP workers hold GPUs 0–3, naming each holder pid
   and its VRAM, before any weight-loading — exactly the fail-fast the
   co-tenancy design wants.

## Federation finding (important, in running note §8)

Mirrored proto *shape* ≠ interoperable wire contract: gRPC method paths embed
package+service (`/aegir.engine.AegirEngine/Complete` vs
`/atelier.engine.AtelierEngine/Complete`), so cross-engine federation with our
own stub gets UNIMPLEMENTED. Convergence options: shared `zndx.engine.Engine`
service registered by all engines, or KServe OIP as the peer protocol (Gaius's
choice, for exactly this reason). Until settled, federate via the target
engine's client.

## Pending

- Nemotron-3-Nano-30B-A3B-BF16 download in progress (~27G/60G at note time).
- Referee cold-serve smoke needs a GPU window (Ægir's endpoint holds 0–3;
  we will not touch it — coordinate a pause or wait).
- Classify-LLM-channel config pointing at our engine's instruct endpoint
  (inc-0: `classify.llm.base_url=http://127.0.0.1:8200/v1`, model="instruct")
  — after first local serve smoke.
- Supervisor + staged readiness (Ægir's `supervisor.py`/`readiness.py`
  equivalents) — hardening pass, before multi-day optimize runs.
- Then task 8: SDG working-set builder + curation loop on `referee`.
