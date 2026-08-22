"""The engine's **sole** vLLM client. Launches/manages an OpenAI-compatible
vLLM server per capability on local GPUs (tensor-parallel), health-waits with
proof-of-progress log scanning, and runs ``Complete`` by calling it INTERNALLY.
The vLLM endpoint is never exposed outside the engine — workloads reach
inference only through the gRPC ``Complete``.

Mirrors Ægir's ``vllm_manager`` with three grafts:
- **Proof-of-progress startup** (Gaius pattern + this project's directive):
  the vLLM log is regex-scanned during health-wait; fatal patterns
  (OOM/CUDA error) abort immediately instead of burning the full timeout,
  and progress transitions are logged so a 4-minute cold load is observable.
- **GPU lease** (:mod:`atelier.engine.gpu_guard`) claimed per endpoint before
  launch — co-tenancy with the Ægir/Gaius engines on the shared 6-GPU host.
- **Guided JSON**: ``complete(..., json_schema=...)`` forwards vLLM's
  ``guided_json`` param for schema-constrained decoding (the structured-output
  gap both sibling engines currently leave open on the local path).
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from atelier.engine.config import EngineConfig, ModelSpec
from atelier.engine.events import emit
from atelier.engine.gpu_guard import claim_gpus

logger = logging.getLogger(__name__)

# (regex, label) scanned over the vLLM log during startup, in order; the last
# match wins. Fatal patterns short-circuit the health wait.
_PROGRESS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Loading model weights|Starting to load model"), "loading weights"),
    (re.compile(r"Loading (?:checkpoint shards|safetensors).*?(\d+)%"), "loading shards"),
    (re.compile(r"Model loading took|loaded in \d"), "weights loaded"),
    (re.compile(r"Capturing CUDA graph|CUDA graphs"), "capturing CUDA graphs"),
    (re.compile(r"Warming up|warmup"), "warming up"),
    (re.compile(r"Uvicorn running|Application startup complete"), "API server up"),
]
_FATAL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"OutOfMemoryError|CUDA out of memory|\bOOM\b"), "GPU out of memory"),
    (re.compile(r"CUDA error|CUDA_ERROR"), "CUDA error"),
    (re.compile(r"Free memory.*less than", re.IGNORECASE), "insufficient free VRAM"),
]


def split_thinking(content: str, reasoning: str) -> tuple[str, str]:
    """(answer, trace). Thinking models wrap the trace in <think>…</think>; the
    opening tag is often template-injected (absent from output) and the closing
    present. When no vLLM reasoning parser separated it, split here so the
    trace is retained SEPARATELY instead of bleeding into the answer."""
    if not reasoning and "</think>" in content:
        head, _, tail = content.partition("</think>")
        reasoning = head.replace("<think>", "").strip()
        content = tail.strip()
    return content, reasoning


@dataclass
class Endpoint:
    capability: str
    spec: ModelSpec
    port: int
    gpu_ids: list[int]
    proc: subprocess.Popen | None = None
    healthy: bool = False
    log_path: Path | None = None
    _lease: contextlib.AbstractContextManager | None = field(default=None, repr=False)


class VllmManager:
    """On-demand vLLM endpoints, one per capability. Thread-safe ensure()."""

    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self._ep: dict[str, Endpoint] = {}
        self._lock = threading.Lock()
        self._next_gpu = cfg.gpu0
        # capability → GPU block, assigned once — a relaunch after a failed
        # or dead endpoint must REUSE its slot, not walk off the end of the
        # device range.
        self._gpu_slots: dict[str, list[int]] = {}
        # capability → port, sticky for the same reason.
        self._ports: dict[str, int] = {}
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

    def ensure(self, capability: str) -> Endpoint:
        if capability not in self.cfg.capabilities:
            raise KeyError(
                f"unknown capability {capability!r}; known: {list(self.cfg.capabilities)}"
            )
        with self._lock:
            ep = self._ep.get(capability)
            if ep and ep.proc and ep.proc.poll() is None and ep.healthy:
                return ep
            # Occupancy intent before GPU launch. UNIMPLEMENTED = Signals
            # not-yet; SHAREFAIL fails admit. Never write queues.yaml.
            from atelier.engine.queue_share import notify_admit

            notify_admit(capability)
            if ep is None or (ep.proc and ep.proc.poll() is not None):
                ep = self._launch(capability)
                self._ep[capability] = ep
            self._wait_healthy(ep)
            return ep

    def _launch(self, capability: str) -> Endpoint:
        spec = self.cfg.capabilities[capability]
        tp = spec.tensor_parallel_size
        gpu_ids = self._gpu_slots.get(capability)
        if gpu_ids is None:
            gpu_ids = list(range(self._next_gpu, self._next_gpu + tp))
            self._next_gpu += tp
            self._gpu_slots[capability] = gpu_ids
        port = self._ports.setdefault(
            capability, self.cfg.vllm_base_port + len(self._ports)
        )
        log_path = Path(self.cfg.log_dir) / f"vllm_{capability}_{port}.log"

        # Claim the lease BEFORE spending minutes loading weights — fail fast
        # when a sibling engine already serves on these GPUs.
        lease = claim_gpus(
            gpu_ids, lock_dir=self.cfg.lock_dir,
            min_mib=self.cfg.gpu_min_free_mib, role=f"atelier-engine/{capability}",
        )
        lease.__enter__()

        env = dict(os.environ)
        env["HF_HUB_CACHE"] = self.cfg.hf_hub_cache
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        # The vLLM venv is FOREIGN (cu129, built against the system glibc).
        # Do NOT inherit the nix/devenv LD_LIBRARY_PATH (gcc-15 libstdc++ wants
        # a GLIBC the system libc lacks and breaks the foreign torch import).
        # Only the libcuda unmask dir — proven sufficient for CUDA init.
        if self.cfg.cuda_driver_libs:
            env["LD_LIBRARY_PATH"] = self.cfg.cuda_driver_libs
        else:
            env.pop("LD_LIBRARY_PATH", None)
        for k in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(k, None)

        cmd = [
            self.cfg.vllm_python, "-m", "vllm.entrypoints.openai.api_server",
            "--model", spec.model, "--served-model-name", capability,
            "--port", str(port), "--host", "127.0.0.1",
            "--tensor-parallel-size", str(tp),
            "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
            *spec.extra_args,
        ]
        fh = open(log_path, "w")
        # start_new_session=True puts vLLM + its TP workers in one process
        # group so shutdown() can kill the WHOLE tree — killing only the parent
        # orphans workers that keep holding ~24 GB VRAM each (observed in Ægir).
        proc = subprocess.Popen(
            cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True
        )
        logger.info("launched vLLM[%s] pid=%s port=%s gpus=%s model=%s",
                    capability, proc.pid, port, gpu_ids, spec.model)
        emit(self.cfg.log_dir, "endpoint_launch", capability=capability,
             model=spec.model, port=port, gpu_ids=gpu_ids)
        return Endpoint(capability=capability, spec=spec, port=port, gpu_ids=gpu_ids,
                        proc=proc, log_path=log_path, _lease=lease)

    # ── Proof-of-progress health wait ────────────────────────────────

    def _scan_log(self, ep: Endpoint) -> tuple[str, str | None]:
        """(latest progress label, fatal reason or None) from the vLLM log."""
        if not ep.log_path or not ep.log_path.exists():
            return ("starting", None)
        text = ep.log_path.read_text(errors="ignore")
        for pat, reason in _FATAL_PATTERNS:
            if pat.search(text):
                return ("failed", reason)
        label = "starting"
        for pat, name in _PROGRESS_PATTERNS:
            if pat.search(text):
                label = name
        return (label, None)

    def _wait_healthy(self, ep: Endpoint, timeout: float = 900.0) -> None:
        url = f"http://127.0.0.1:{ep.port}/health"
        t0 = time.time()
        last_label = ""
        while time.time() - t0 < timeout:
            if ep.proc and ep.proc.poll() is not None:
                self._release(ep)
                raise RuntimeError(
                    f"vLLM for {ep.capability!r} exited (rc={ep.proc.returncode}); "
                    f"see {ep.log_path}\n{self._tail(ep.log_path)}"
                )
            label, fatal = self._scan_log(ep)
            if fatal:
                emit(self.cfg.log_dir, "endpoint_fatal", capability=ep.capability,
                     reason=fatal)
                self.stop(ep.capability)
                raise RuntimeError(
                    f"vLLM for {ep.capability!r} failed during startup: {fatal}; "
                    f"see {ep.log_path}\n{self._tail(ep.log_path)}"
                )
            if label != last_label:
                logger.info("vLLM[%s] startup: %s (t+%.0fs)", ep.capability, label, time.time() - t0)
                last_label = label
            try:
                if httpx.get(url, timeout=2.0).status_code == 200:
                    ep.healthy = True
                    logger.info("vLLM[%s] healthy in %.0fs", ep.capability, time.time() - t0)
                    emit(self.cfg.log_dir, "endpoint_healthy", capability=ep.capability,
                         model=ep.spec.model, elapsed_s=round(time.time() - t0, 1))
                    return
            except Exception:  # noqa: BLE001 — not up yet
                pass
            time.sleep(3.0)
        raise TimeoutError(
            f"vLLM for {ep.capability!r} not healthy in {timeout}s; see {ep.log_path}"
        )

    @staticmethod
    def _tail(path: Path | None, n: int = 25) -> str:
        if not path or not path.exists():
            return ""
        return "\n".join(path.read_text(errors="ignore").splitlines()[-n:])

    # ── Inference ────────────────────────────────────────────────────

    def complete(self, capability: str, prompt: str, system_prompt: str = "",
                 max_tokens: int = 512, temperature: float = 0.7,
                 json_schema: str = "") -> dict:
        ep = self.ensure(capability)
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
        msgs.append({"role": "user", "content": prompt})
        payload: dict = {"model": capability, "messages": msgs,
                         "max_tokens": max_tokens, "temperature": temperature}
        if json_schema:
            # Schema-constrained decoding via the OpenAI-style response_format.
            # NB: the legacy top-level `guided_json` param is SILENTLY IGNORED
            # by vLLM 0.19.0 (verified live 2026-07-03 — required fields came
            # back missing); response_format json_schema + strict enforces
            # fully. Guided decoding suppresses the thinking trace by
            # construction.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "constrained_output",
                    "schema": json.loads(json_schema),
                    "strict": True,
                },
            }
        t0 = time.time()
        # RETAIN thinking (Ægir convention): no enable_thinking=False; generous
        # read timeout — long traces are expected and the workload waits.
        r = httpx.post(f"http://127.0.0.1:{ep.port}/v1/chat/completions",
                       json=payload, timeout=900.0)
        r.raise_for_status()
        d = r.json()
        choice = d["choices"][0]
        msg = choice["message"]
        usage = d.get("usage", {})
        content, reasoning = split_thinking(
            msg.get("content") or "", msg.get("reasoning_content") or ""
        )
        emit(self.cfg.log_dir, "complete", capability=capability,
             model=ep.spec.model,
             prompt_tokens=usage.get("prompt_tokens", 0),
             completion_tokens=usage.get("completion_tokens", 0),
             latency_ms=round((time.time() - t0) * 1000.0, 1),
             finish_reason=choice.get("finish_reason") or "",
             schema_constrained=bool(json_schema),
             reasoning_retained=bool(reasoning))
        return {"text": content, "model": ep.spec.model,
                "reasoning_content": reasoning,
                "finish_reason": choice.get("finish_reason") or "",
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "latency_ms": (time.time() - t0) * 1000.0}

    # ── Lifecycle ────────────────────────────────────────────────────

    def status(self) -> list[Endpoint]:
        return list(self._ep.values())

    def _release(self, ep: Endpoint) -> None:
        if ep._lease is not None:
            with contextlib.suppress(Exception):
                ep._lease.__exit__(None, None, None)
            ep._lease = None

    def stop(self, capability: str) -> None:
        ep = self._ep.get(capability)
        if not ep:
            return
        if ep.proc and ep.proc.poll() is None:
            try:
                os.killpg(os.getpgid(ep.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                ep.proc.kill()
        self._release(ep)
        ep.healthy = False
        emit(self.cfg.log_dir, "endpoint_stop", capability=capability)

    def shutdown(self) -> None:
        for capability in list(self._ep):
            self.stop(capability)
