<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# GPU Acceleration

Atelier auto-detects CUDA at pipeline start via `preflight_gpu()` and
transparently routes SAGE, SHAP (embedding path), CatBoost training,
and optionally UMAP 2D projection onto GPU kernels. The pipeline is
unchanged on CPU hosts — every GPU path has a CPU fallback.

## What accelerates

| Component | Implementation |
|---|---|
| Sentence-transformer encode | `MultiDeviceEncoder` with optional multi-GPU sharding above `classify.gpu.shard_threshold` |
| SAGE | Custom vectorized kernel in [`gpu_importance.py`](https://github.com/zndx/atelier/blob/trunk/src/atelier/classify/gpu_importance.py); replaces `sage-importance` on GPU hosts |
| PermutationSHAP (embedding) | Same kernel; per-item aggregation; replaces `shap.PermutationExplainer` on GPU hosts |
| TreeSHAP (CatBoost) | Native CatBoost; already fast, unchanged |
| CatBoost training | `task_type="GPU"`; `posterior_sampling` disabled on GPU (not supported); try/except CPU fallback |
| UMAP 2D projection | `cuml.manifold.UMAP` when `scripts/install_deps.py` has pip-installed RAPIDS on a GPU host, else `umap-learn` |

## Why SAGE/SHAP are fast on GPU

Upstream `sage-importance` and `shap.PermutationExplainer` do a Python
permutation loop that calls the model once per context, reencoding
strings each call. Our kernel:

1. Flattens the whole permutation chunk into a single `(chunk × (F+1) × N, F)` fingerprint array — "which sample supplies each feature at each step."
2. Deduplicates via `np.unique(axis=0)` and builds text once per unique fingerprint.
3. Encodes the unique text set on GPU in a single call.
4. Computes all `(chunk × (F+1) × N)` losses via one `torch.matmul` + `log_softmax`.
5. Accumulates SAGE values with running mean + variance for convergence.

Two tricks keep the unique-context count tractable:
- **Fixed global donors**: one donor sample per feature for the whole run. Coarser Monte Carlo estimator than per-sample donors, but lifts cache reuse dramatically (orders of magnitude fewer unique texts).
- **Chunk-bounded memory**: chunk=16 permutations keeps per-chunk unique embeddings under a few hundred thousand, fitting a single 4090.

## Why not multi-GPU for MiniLM-L6 encode

`all-MiniLM-L6-v2` saturates a single 4090 at ~13K texts/s for typical
SAGE context lengths. Splitting a batch across six devices introduces
Python-thread + GIL coordination that exceeds the compute savings at
the batch sizes we see in practice. `shard_threshold` is set to 200K
to keep MiniLM on a single device; lower it when running a larger
embedding model (BGE-large, E5-mistral, etc.) where multi-GPU pays off.

## Config

```hocon
classify {
  gpu {
    enabled = "auto"                # auto | true | false
    shard_threshold = 200000        # texts before multi-GPU fan-out
    sage_chunk_permutations = 16    # permutations per GPU minibatch
  }
}
```

Environment overrides: `ATELIER_GPU_ENABLED`,
`ATELIER_GPU_SHARD_THRESHOLD`, `ATELIER_GPU_SAGE_CHUNK`.

## Surfacing

- `GET /api/acceleration` — device list, resolved methods, config
- `/settings` page shows an "Acceleration" card with active methods and warnings
- `preflight_gpu().to_dict()` — same shape, programmatically

## Optional RAPIDS install

`scripts/install_deps.py` pip-installs RAPIDS directly when
`nvidia-smi` detects a GPU:

```python
subprocess.run([*pip, "install",
                "--extra-index-url", "https://pypi.nvidia.com/",
                "cuml-cu12>=24.10", "cupy-cuda12x>=13.0"], check=True)
```

This is a **CAI-WORKAROUND** for the deferred ideal of a uv-native
`[gpu]` extra in `pyproject.toml` (see comment block in pyproject and
install_deps for the full rationale): cuml-cu12's wheel-stub install
mechanism downloads the real wheel from `pypi.nvidia.com` gated on
nvidia-smi presence at install time, which uv's resolver cannot
model.  When the CAI Session-pod runtime gains GPU access — or cuml
ships uv-resolvable wheels — the install moves back into the
pyproject `[gpu]` extra.

For manual install on a GPU host:

```bash
pip install --extra-index-url https://pypi.nvidia.com/ \
    cuml-cu12>=24.10 cupy-cuda12x>=13.0
```

The pipeline imports `cuml` at UMAP projection time with an
`ImportError` fallback to `umap-learn`.
