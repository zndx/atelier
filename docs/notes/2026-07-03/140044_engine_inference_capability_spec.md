# Atelier engine "inference capability" — build spec

**Date:** 2026-07-03
**Sources:** deep reads of Gaius (`src/gaius/engine/` — the origin pattern) and
Ægir (`src/aegir/engine/` — the "minimal mirror of the Gaius engine" we copy in
turn). RH decisions: local inference for the `just agent` phase; **two distinct
model stacks** — runtime channel vs referee — restoring architectural referee
independence.

## Topology (one 6× RTX 4090 host, three engines eventually co-resident)

| | Gaius | Ægir | **Atelier (new)** |
|---|---|---|---|
| engine gRPC | 50051 (⚠ = our servicer default) | 50151 | **50251** |
| vLLM ports | 8080–8095 | 8100+ | **8200+** |
| GPU practice | fleet 0–5, flows pinned 4–5 | contiguous from GPU0, TP=4 | lease-guarded, lazy per capability |

## Capabilities (the request surface; model choice is the engine's business)

- **`instruct`** — runtime LLM evidence channel: `Qwen/Qwen3.6-35B-A3B-FP8`,
  TP=4, ~9GB/GPU. Already in the shared cache
  `/raid/cache/rch/huggingface` (Ægir's engine cache — we point at the same
  one; no duplicate 35B download).
- **`referee`** — agent-mediated curation ONLY:
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (`NemotronHForCausalLM`,
  hybrid Mamba-MoE, 128 experts, 262K ctx). Confirmed registered in the
  proven cu129 vLLM 0.19.0. BF16 ≈60GB → TP=4. Download in progress to the
  shared cache.
- Boundary rule: Nemotron never serves runtime evidence; Qwen never
  referees. Distinct families ⇒ referee independence is architectural again
  (running-note caveat resolved).
- GPU budget: both at TP=4 don't co-reside on 6×24GB alongside embedding
  work — capabilities load lazily and phases run sequentially under the GPU
  guard; that's fine because referee (curation) and runtime (pipeline sweep)
  phases are naturally sequential in the protocol.

## Components (mirror `aegir/src/aegir/engine/`, names atelier-ized)

`src/atelier/engine/`:
1. **proto** — copy `aegir_engine.proto` shape: `Complete / EnsureEndpoint /
   EngineStatus`, response carries `reasoning_content` + `finish_reason`.
   **Extension (proposed to Ægir):** optional `json_schema` field on
   `CompleteRequest` → vLLM guided-json (`response_format`/guided decoding).
   Wire-compatible proto3 addition; a federated peer that ignores it simply
   returns unconstrained text (documented). Closes the structured-output gap
   BOTH sibling engines currently have on the local path (Gaius checklist
   item: "don't inherit the omission").
2. **`vllm_manager.py`** — lazy `subprocess.Popen` of
   `python -m vllm.entrypoints.openai.api_server --model <spec.model>
   --served-model-name <capability> --host 127.0.0.1 --port <8200+n>
   --tensor-parallel-size --gpu-memory-utilization --max-model-len
   --max-num-seqs --enable-auto-tool-choice --tool-call-parser hermes`.
   Foreign-venv isolation (scrub `PYTHONPATH`/`PYTHONHOME`/`VIRTUAL_ENV`;
   `LD_LIBRARY_PATH` = cuda-driver-libs dir ONLY — reuse
   `aegir/build/cuda-driver-libs` initially, config-keyed);
   `VLLM_PYTHON` default = sibling `agent-evals/.venv-vllm` (vLLM 0.19.0
   cu129, proven). `start_new_session=True` + `killpg` on shutdown (orphaned
   TP workers pin ~24GB/GPU). Health-wait `/health`, 900s.
3. **Gaius hardening grafts:** proof-of-progress startup (regex-scan vLLM
   log → 0→1 progress, negative on `OOM|CUDA error` — matches our
   proof-of-progress directive), port `socket.bind` pre-check, auto-restart
   ≤3 attempts then surface.
4. **`gpu_guard.py`** — filelock + owner-json per GPU set AND authoritative
   `nvidia-smi --query-compute-apps` probe (≥512MiB foreign → refuse).
   **Cross-project note:** filelocks only guard within one lock dir; the
   nvidia-smi probe is what actually protects against Ægir/Gaius co-tenancy.
   Proposal to Ægir: shared lock dir convention (`/tmp/zndx-gpu-leases`) as
   the next federation-prep increment.
5. **`server.py` / `client.py`** — gRPC on 50251; client `_target()` =
   `ATELIER_ENGINE_FEDERATE or 127.0.0.1:50251` (the federation seam:
   pointing at Ægir's 50151 is a config change, day one).
6. **`supervisor.py` + `readiness.py`** — exp backoff, rolling-window retry
   cap, JSONL events; staged `UNREACHABLE < CONNECTED < ENDPOINT_HEALTHY <
   SERVING` where the SERVING probe (1-token Complete) doubles as warm-up.

## Config — HOCON, per Atelier directive (divergence from Ægir's pure-env)

Ægir's engine is env-only; Atelier's design directive is HOCON-as-single-
source with `${?ENV}` substitution. `config/base.conf`:
```
engine {
  grpc_port = 50251            # ${?ATELIER_ENGINE_PORT}
  vllm_base_port = 8200        # ${?ATELIER_VLLM_BASE_PORT}
  vllm_python = ""             # ${?ATELIER_VLLM_PYTHON} (default: sibling probe)
  hf_hub_cache = "/raid/cache/rch/huggingface"   # ${?ATELIER_HF_HUB_CACHE}
  cuda_driver_libs = ""        # ${?ATELIER_CUDA_DRIVER_LIBS}
  federate = ""                # ${?ATELIER_ENGINE_FEDERATE}
  log_dir = "/tmp/atelier-engine"
  gpu0 = 0
  capabilities {
    instruct { model = "Qwen/Qwen3.6-35B-A3B-FP8",  tp = 4, gpu_mem = 0.90,
               max_model_len = 16384, max_num_seqs = 8 }
    referee  { model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", tp = 4,
               gpu_mem = 0.92, max_model_len = 32768, max_num_seqs = 4 }
  }
}
```
just recipes: `engine-serve`, `engine-ping`, `engine-supervise`,
`engine-ready` (mirror Ægir's four).

## Consumption

- **Pipeline LLM channel:** inc-0 = `OpenAICompatibleBackend` pointed at
  `http://127.0.0.1:8200/v1`, `model="instruct"` (served-model-name ==
  capability). Matches Ægir's own current practice (their vibe-acp talks to
  8100 directly; strict-layering proxy is their documented follow-up).
  inc-1 = thin OpenAI-compat proxy over engine gRPC for strict layering.
- **Referee harness:** NOT vibe-acp — a procedural curation loop
  (`scripts/run_curate_local.py`) that walks the SDG working set, drives the
  `referee` capability through the engine client (guided-json when the proto
  extension lands; soft-parse fallback: strip `</think>`, extract JSON),
  runs the skill's deterministic cross-checks (pattern detectors,
  value-format validators, sibling context), and writes
  `agent_mediated.json` + `audit.json` + the sha256 ingress manifest.
  "Harness > model" — the curate skill's own principle; also keeps the
  membrane-oracle property (harness owns the working set and the checks;
  the model only proposes).

## Risks / open items

- vLLM 0.19.0's NemotronH support must handle the MoE variant at runtime
  (arch registered; hybrid-Mamba serving flags may need tuning —
  discovered at first `EnsureEndpoint`).
- Guided-json on NemotronH under 0.19.0 untested — soft-parse fallback is
  the day-one path.
- 4090s lack P2P: TP=4 over PCIe works but slower; A3B active params keep
  throughput acceptable for curation batch sizes.
- Thinking-trace policy: retain (Ægir convention) — `reasoning_content`
  lands in the referee audit trail, which strengthens procedural
  reproduction (principle 2 of the curate skill).
