# Zero-share GPU policy — guard threshold below bare-context size + full vacate

**Date:** 2026-07-04
**Decision (RH):** co-tenants vacate the GPUs entirely; no memory-sharing on
consumer cards. An idle CUDA context now counts as occupancy.

## What changed

- `src/atelier/engine/gpu_guard.py` — `claim_gpus(min_mib=...)` default
  **512 → 64 MiB**, i.e. below bare-CUDA-context size (~300–500 MiB on the
  4090s). Module docstring states the policy. Threshold stays >0 (not
  any-entry) so a malformed compute-apps row can't become a phantom holder.
- `src/atelier/engine/config.py` — `EngineConfig.gpu_min_free_mib` default
  **512 → 64**.
- `config/base.conf` — `engine.gpu_min_free_mib = 64`, new env override
  `ATELIER_ENGINE_GPU_MIN_FREE_MIB`, comment records the rationale.
- `tests/engine/test_engine.py` — new
  `test_default_threshold_counts_bare_cuda_context` pins BOTH defaults
  (guard signature + EngineConfig) below 300 MiB and asserts a 384 MiB
  bare context registers as a holder. Full sweep: **152 passed**;
  preflight 7/7.

## The motivating incident

The devenv gateway (uvicorn, pid 3822781, **4 days up**, orphaned — its
process-compose was long gone) had been idling a **384 MiB CUDA context on
all six GPUs** — torch init from an in-process pipeline run. Under the old
512 MiB threshold it was invisible to both our guard and Ægir's: either
engine would have launched TP workers on top of it. Killed today; `nvidia-smi`
compute-apps verified **empty** — all six cards clean, Ægir's window genuinely
free.

## Cross-project

Running-observations note **§19** (aegir
`docs/scratch/2026-07-03/135240_atelier_running_observations.md`) posts the
policy + asks Ægir to mirror the 512 → 64 default in their `gpu_guard` so
protection is symmetric.

## Operational corollary (recurrence vector)

Any in-process GPU phase in a long-lived process (gateway pipeline runs,
notebook kernels) re-pins a context for the life of the process — which is
now a claim-blocker by design. Until GPU work moves out-of-process, restart
the gateway after GPU phases. Orphan hygiene observed while here: three
qdrant generations (pids 2659577, 3346830, 3822562) from successive
`devenv up` sessions are still running — none touch the GPUs, left alone,
but worth a cleanup pass.
