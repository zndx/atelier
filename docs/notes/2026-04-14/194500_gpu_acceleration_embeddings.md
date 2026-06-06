# GPU Acceleration for Sentence-Transformer Embeddings

## Summary

Ported GPU detection and acceleration pattern from Signals (`sigint/config.py`)
to Atelier. Embeddings now use CUDA when available, with automatic batch size
scaling from 32 to 256 on GPU.

## Results

- **Hardware**: 6x NVIDIA GeForce RTX 4090 (24 GB VRAM each)
- **Throughput**: 1,503 texts/s (GPU) vs ~54 texts/s (CPU) — **28x speedup**
- **Model**: all-MiniLM-L6-v2 on cuda:0

## Changes

### New File
- `src/atelier/classify/gpu.py` — `GpuInfo` dataclass + `preflight_gpu()` with
  nvidia-smi probe, PyTorch CUDA validation, version mismatch detection

### Modified
- `src/atelier/classify/embedding.py` — Added `configure(device, batch_size)`,
  device-aware `_get_model()`, batch_size in all `.encode()` calls
- `config/base.conf` — `embedding_device`, `embedding_batch_size` under classify
- `src/atelier/config.py` — 2 HOCON mappings + 2 dataclass fields
- `src/atelier/classify/pipeline.py` — Calls `configure_embeddings()` early;
  `_compute_projection()` uses shared model instead of creating redundant instance
- `src/atelier/classify/sage.py` — Passes `_batch_size` to `.encode()`
- `src/atelier/preflight.py` — GPU status in preflight report (advisory)
- `devenv.nix` — LD_LIBRARY_PATH includes CUDA toolkit + nvidia driver symlinks
- Feature files — `@gpu` tag on embedding-heavy scenarios

## Nix + CUDA Gotcha

PyTorch's `torch.cuda.is_available()` returns False in nix devenv because:
1. `libcuda.so.1` lives in `/usr/lib/x86_64-linux-gnu/` (host OS driver)
2. Adding that directory to `LD_LIBRARY_PATH` causes glibc version conflicts with nix
3. Solution: symlink just `libcuda.so.1` into `build/lib/nvidia/` and add that to
   `LD_LIBRARY_PATH`. The `enterShell` hook creates this symlink automatically.
4. `/usr/local/cuda/lib64` (CUDA toolkit) doesn't have this conflict.

## Pipeline Integration

`configure_embeddings()` is called early in `run_classification_pipeline()`:
```python
from atelier.classify.embedding import configure as configure_embeddings
configure_embeddings(
    device=cfg.classify_embedding_device,
    batch_size=cfg.classify_embedding_batch_size,
)
```

When `device="auto"` (default), calls `preflight_gpu().resolved_device` → "cuda" or "cpu".
Auto-scales batch_size 32→256 on GPU. Graceful CPU fallback always safe.
