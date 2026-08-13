"""Engine config — capability→model registry, loaded from HOCON.

Divergence from the Ægir engine (pure env): Atelier's design directive is
HOCON-as-single-source-of-truth — the engine reads the ``engine`` subtree of
``config/base.conf``, and environment overrides flow through ``${?VAR}``
substitution there, never via direct ``os.environ`` reads for config values.
(The two ``os.environ`` touches below are path *probes* for machine-local
toolchain discovery, mirroring Ægir's sibling-venv probe, not config values.)

The registry is PROVISIONAL scaffolding in the same sense as Ægir's: the
capability indirection is the design; the model choices evolve with workloads.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT_CONF = _REPO / "config" / "base.conf"

# vLLM runs in a DEDICATED FOREIGN venv (see vllm_manager). Default: the sibling
# agent-evals venv PROVEN on this machine (vLLM 0.19.0, torch 2.10.0+cu129 —
# cu129 runs on the 12.8 driver via minor-version compat; NemotronH + Qwen3.6-MoE
# architectures verified registered 2026-07-03).
_VLLM_CANDIDATES = [
    Path.home() / "local/src/zndx/agent-evals/.venv-vllm/bin/python",
    _REPO / ".venv-vllm" / "bin" / "python",
]

# The libcuda unmask dir Ægir proved sufficient for CUDA init from the foreign
# venv (their build/cuda-driver-libs). We borrow it until Atelier materializes
# its own copy.
_CUDA_LIBS_CANDIDATES = [
    _REPO / "build" / "cuda-driver-libs",
    Path.home() / "local/src/zndx/aegir/build/cuda-driver-libs",
]


@dataclass
class ModelSpec:
    model: str                      # vLLM model id (resolved from hf_hub_cache) or local path
    tensor_parallel_size: int = 4
    gpu_memory_utilization: float = 0.90
    extra_args: list[str] = field(default_factory=list)


@dataclass
class EngineConfig:
    grpc_port: int = 50251          # Gaius squats 50051, Ægir 50151 — co-tenancy by construction
    vllm_base_port: int = 8200      # Gaius 8080-8095, Ægir 8100+
    vllm_python: str = ""
    hf_hub_cache: str = "/raid/cache/rch/huggingface"  # shared with Ægir's engine (no /hub subdir)
    cuda_driver_libs: str = ""
    federate: str = ""              # host:port of a sibling engine; empty = local
    log_dir: str = "/tmp/atelier-engine"
    gpu0: int = 0
    lock_dir: str = "/tmp/zndx-gpu-leases"  # SHARED across sibling engines (proposed convention)
    gpu_min_free_mib: int = 64     # below bare-context size: idle CUDA contexts count as occupancy
    capabilities: dict[str, ModelSpec] = field(default_factory=dict)

    def resolve_paths(self) -> None:
        if not self.vllm_python:
            for cand in _VLLM_CANDIDATES:
                if cand.exists():
                    self.vllm_python = str(cand)
                    break
        if not self.cuda_driver_libs:
            for cand in _CUDA_LIBS_CANDIDATES:
                if cand.exists():
                    self.cuda_driver_libs = str(cand)
                    break


def _spec_from_tree(tree) -> ModelSpec:
    extra = list(tree.get("extra_args", []) or [])
    max_len = tree.get("max_model_len", None)
    max_seqs = tree.get("max_num_seqs", None)
    if max_len is not None:
        extra += ["--max-model-len", str(max_len)]
    if max_seqs is not None:
        extra += ["--max-num-seqs", str(max_seqs)]
    if tree.get("tool_parser", ""):
        extra += ["--enable-auto-tool-choice", "--tool-call-parser", str(tree["tool_parser"])]
    if bool(tree.get("trust_remote_code", False)):
        extra += ["--trust-remote-code"]
    return ModelSpec(
        model=str(tree["model"]),
        tensor_parallel_size=int(tree.get("tp", 4)),
        gpu_memory_utilization=float(tree.get("gpu_mem", 0.90)),
        extra_args=extra,
    )


def load_engine_config(conf_path: str | Path | None = None) -> EngineConfig:
    """Parse the ``engine`` subtree of the HOCON config (env via ``${?VAR}``)."""
    from pyhocon import ConfigFactory

    path = Path(conf_path) if conf_path else Path(os.environ.get("ATELIER_CONF", _DEFAULT_CONF))
    conf = ConfigFactory.parse_file(str(path)) if path.exists() else ConfigFactory.parse_string("")
    tree = conf.get("engine", None)

    cfg = EngineConfig()
    if tree is not None:
        cfg.grpc_port = int(tree.get("grpc_port", cfg.grpc_port))
        cfg.vllm_base_port = int(tree.get("vllm_base_port", cfg.vllm_base_port))
        cfg.vllm_python = str(tree.get("vllm_python", "") or "")
        cfg.hf_hub_cache = str(tree.get("hf_hub_cache", cfg.hf_hub_cache))
        cfg.cuda_driver_libs = str(tree.get("cuda_driver_libs", "") or "")
        cfg.federate = str(tree.get("federate", "") or "")
        cfg.log_dir = str(tree.get("log_dir", cfg.log_dir))
        cfg.gpu0 = int(tree.get("gpu0", cfg.gpu0))
        cfg.lock_dir = str(tree.get("lock_dir", cfg.lock_dir))
        cfg.gpu_min_free_mib = int(tree.get("gpu_min_free_mib", cfg.gpu_min_free_mib))
        caps = tree.get("capabilities", None)
        if caps:
            cfg.capabilities = {name: _spec_from_tree(sub) for name, sub in caps.items()}
    # Lattice / unit override (signals peer unit)
    if os.environ.get("ATELIER_ENGINE_PORT"):
        cfg.grpc_port = int(os.environ["ATELIER_ENGINE_PORT"])
    cfg.resolve_paths()
    return cfg
