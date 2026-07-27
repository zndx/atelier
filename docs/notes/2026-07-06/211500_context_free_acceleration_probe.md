# Context-free acceleration probe — UI never touches CUDA under occupancy

**Date:** 2026-07-06 (extends 2026-07-04 zero-share policy note)

## Decision (RH)

The UI startup path must work under FULL co-tenant GPU occupancy with no
GPU interaction at all: when foreign compute-apps hold the cards, the
acceleration probe skips `torch.cuda.mem_get_info()` (the sole
context-pinner — `is_available()` verified context-free empirically) and
reports VRAM from nvidia-smi instead.

## What changed

- `classify/gpu.py preflight_gpu()` — step 1 now also captures
  `memory.total,memory.free` (nounits) from nvidia-smi plus a context-free
  compute-apps occupancy probe. Step 2 branches: foreign pids present →
  nvidia-smi VRAM + loud zero-share warning + `occupied=True` (new GpuInfo
  field, in `to_dict()`); cards free → runtime `mem_get_info` loop as
  before (more accurate under CUDA_VISIBLE_DEVICES).
- `preflight_gpu_isolated()` (added 07-06 earlier in session) — runs the
  probe in a child process for long-lived callers; `/api/acceleration` in
  the gateway now uses it. Under occupancy the child is now ALSO
  context-free — the transient-context caveat is gone.
- `tests/classify/test_gpu_probe.py` — 3 hermetic tests; the occupied
  test monkeypatches `mem_get_info` to raise, pinning the invariant.
  Full sweep: **155 passed**.

## Live verification (Ægir serving on GPUs 0–3)

`preflight_gpu_isolated()`: `occupied=True`, summary correct, VRAM from
nvidia-smi (`[2178, 2178, 2180, 2180, 23699, 23699]`), zero-share warning
naming 5 holders (Ægir's 4 TP workers + our stale-context gateway).

## UI startup procedure under full occupancy (settled)

1. `devenv up` — no UI-stack component touches CUDA at startup (verified;
   no `preflight_gpu()` on any boot path).
2. Ports disjoint from Ægir by construction (3000/8090/50071/5533/6333 vs
   5173/8080/5006/50151/8100/21000).
3. Optional invariant check: no Atelier pid in
   `nvidia-smi --query-compute-apps`.
4. Engine may start too (`just engine-serve`) — claims only at
   `ensure()`, per-capability, before weight-loading.

## Pending

- **Gateway restart** (sheds the 6×384 MiB contexts from the 07-06
  in-process probe + loads the fixed endpoint): kill the uvicorn pair
  (uv wrapper + child); the live `devenv up` (pid 3822228) process-compose
  restarts it. Agent permission-blocked from killing operator processes —
  operator action.
- Pipeline-run GPU gating (claim_gpus integration) — deliberately
  deferred; UI-only scope this pass.
