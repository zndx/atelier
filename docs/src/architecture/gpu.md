# GPU Acceleration

Atelier uses GPU acceleration for sentence-transformer embedding computation
and CatBoost training/inference. GPU support is auto-detected at startup
with graceful fallback to CPU.

## Detection

`gpu.preflight_gpu()` runs once at config load time and caches the result
for the process lifetime. Three-step detection:

1. **nvidia-smi probe**: subprocess call to detect device count, names,
   VRAM, and driver CUDA version
2. **CUDA version extraction**: parse nvidia-smi header for driver
   compatibility
3. **PyTorch check**: `torch.cuda.is_available()` confirms runtime support

The result is a `GpuInfo` dataclass with:

- `available` — whether CUDA is usable
- `device_count` — number of GPUs
- `devices` — device names with VRAM (e.g., "NVIDIA RTX 4090 24GB")
- `resolved_device` — `"cuda"` or `"cpu"` for model initialization
- `warnings` — non-blocking issues (version mismatches, library path hints)

## NVIDIA Driver Symlink (nix + CUDA)

In devenv (nix-managed), CUDA libraries are isolated from the host system.
The GPU module handles the nix+CUDA compatibility pattern by detecting
the driver library path and ensuring PyTorch can find it. This avoids
the common nix pitfall where `torch.cuda.is_available()` returns `False`
despite GPUs being present.

## Integration Points

### Sentence-Transformer Embedding

`embedding.py` calls `preflight_gpu()` before initializing the
`SentenceTransformer` model, passing `device=gpu_info.resolved_device`:

```python
gpu_info = preflight_gpu()
model = SentenceTransformer("all-MiniLM-L6-v2", device=gpu_info.resolved_device)
```

GPU batch encoding achieves ~2,768 texts/second on RTX 4090 (vs ~400/s
on CPU). This matters at scale: 15M columns takes ~90 minutes on GPU
vs ~10 hours on CPU.

### CatBoost Training

CatBoost automatically uses GPU when available via its `task_type`
parameter. The virtual ensemble posterior sampling that drives uncertainty
quantification benefits from GPU parallelism.

### Preflight Reporting

GPU status appears in `just preflight` output and in the `/api/status`
gateway endpoint, giving operators immediate visibility into whether
GPU acceleration is active.

## Configuration

GPU detection is automatic — no configuration needed. The system probes
hardware and falls back gracefully:

- **GPU available**: uses CUDA for all embedding and training operations
- **GPU detected but CUDA unavailable**: warns about library path issues,
  falls back to CPU
- **No GPU**: runs entirely on CPU with no warnings

## CAI Considerations

CAI ML workloads can request GPU runtimes. When running on a GPU-enabled
CAI session:

- The NVIDIA drivers are provided by the container runtime
- PyTorch CUDA support depends on the Python runtime image
- GPU memory is shared with other processes in the session
- Background SHAP computation can be memory-intensive; monitor with
  `nvidia-smi` if running alongside large models
