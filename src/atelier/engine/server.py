"""Atelier capability/gRPC engine server. Implements ``Complete`` (forwards to
vLLM *inside* the engine via the manager), plus admin ``EnsureEndpoint`` /
``EngineStatus``. The vLLM endpoint is never exposed to callers.

    just engine-serve            # or: uv run python -m atelier.engine.server

NB this is a SEPARATE process from the Atelier gRPC servicer (:50051 default,
`atelier.server`) — the engine owns GPUs and model processes; the servicer owns
classification runs. Distinct ports by design (engine default :50251).
"""
from __future__ import annotations

import logging
import signal
import sys
from concurrent import futures

import grpc

from atelier.engine.config import load_engine_config
from atelier.engine.proto import atelier_engine_pb2 as pb
from atelier.engine.proto import atelier_engine_pb2_grpc as pbg
from atelier.engine.vllm_manager import VllmManager

logger = logging.getLogger(__name__)


class AtelierEngineServicer(pbg.AtelierEngineServicer):
    def __init__(self, cfg=None) -> None:
        self.cfg = cfg or load_engine_config()
        self.mgr = VllmManager(self.cfg)

    def Complete(self, request, context):
        cap = request.capability or "instruct"
        try:
            out = self.mgr.complete(
                cap, request.prompt, request.system_prompt or "",
                request.max_tokens or 512, request.temperature or 0.7,
                json_schema=request.json_schema or "",
            )
            return pb.CompleteResponse(
                text=out["text"], model=out["model"],
                prompt_tokens=out["prompt_tokens"],
                completion_tokens=out["completion_tokens"],
                latency_ms=out["latency_ms"],
                reasoning_content=out["reasoning_content"],
                finish_reason=out["finish_reason"])
        except Exception as e:  # noqa: BLE001 — surface as gRPC error, keep engine up
            context.abort(grpc.StatusCode.INTERNAL, f"complete[{cap}] failed: {e}")

    def EnsureEndpoint(self, request, context):
        try:
            ep = self.mgr.ensure(request.capability or "instruct")
            return pb.EndpointStatus(
                capability=ep.capability, model=ep.spec.model, healthy=ep.healthy,
                port=ep.port, gpu_ids=ep.gpu_ids, detail=str(ep.log_path))
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.INTERNAL, f"ensure failed: {e}")

    def EngineStatus(self, request, context):
        eps = [pb.EndpointStatus(capability=e.capability, model=e.spec.model,
                                 healthy=e.healthy, port=e.port, gpu_ids=e.gpu_ids)
               for e in self.mgr.status()]
        return pb.EngineStatusResponse(endpoints=eps, total_gpus=_gpu_count())


class ZndxEngineServicer:
    """The shared federation face — ``zndx.engine.v1.Engine``.

    Registered ADDITIONALLY beside the native service (signals-protocol design:
    one stub, any engine). Delegates to the same manager; engine-private
    details (vLLM ports, log paths) do not cross this boundary.
    """

    def __init__(self, native: AtelierEngineServicer) -> None:
        self._native = native

    def Complete(self, request, context):
        from zndx.engine.v1 import engine_pb2 as zpb
        cap = request.capability or "instruct"
        try:
            out = self._native.mgr.complete(
                cap, request.prompt, request.system_prompt or "",
                request.max_tokens or 512, request.temperature or 0.7,
                json_schema=request.json_schema or "",
            )
            return zpb.CompleteResponse(
                text=out["text"], model=out["model"],
                prompt_tokens=out["prompt_tokens"],
                completion_tokens=out["completion_tokens"],
                latency_ms=out["latency_ms"],
                reasoning_content=out["reasoning_content"],
                finish_reason=out["finish_reason"])
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.INTERNAL, f"complete[{cap}] failed: {e}")

    def Status(self, request, context):
        from zndx.engine.v1 import engine_pb2 as zpb
        eps = [zpb.Endpoint(capability=e.capability, model=e.spec.model,
                            healthy=e.healthy, gpu_ids=e.gpu_ids)
               for e in self._native.mgr.status()]
        return zpb.StatusResponse(project="atelier", endpoints=eps,
                                  total_gpus=_gpu_count())


def _gpu_count() -> int:
    """nvidia-smi-based count — the engine venv has no torch (vLLM is foreign)."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        return len(out.stdout.strip().splitlines()) if out.returncode == 0 else 0
    except (OSError, subprocess.TimeoutExpired):
        return 0


def serve(port: int | None = None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    servicer = AtelierEngineServicer()
    bind_port = port or servicer.cfg.grpc_port
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pbg.add_AtelierEngineServicer_to_server(servicer, server)
    # The shared federation face — one stub, any signals engine.
    from zndx.engine.v1 import engine_pb2_grpc as zpbg
    zpbg.add_EngineServicer_to_server(ZndxEngineServicer(servicer), server)
    server.add_insecure_port(f"[::]:{bind_port}")
    server.start()
    from atelier.engine.events import emit
    emit(servicer.cfg.log_dir, "engine_start", project="atelier",
         grpc_port=bind_port, capabilities=list(servicer.cfg.capabilities))
    print(f"atelier-engine gRPC listening on :{bind_port} "
          f"(capabilities: {list(servicer.cfg.capabilities)}; "
          f"services: atelier.engine.AtelierEngine + zndx.engine.v1.Engine)",
          flush=True)

    def _stop(*_):
        servicer.mgr.shutdown()
        server.stop(grace=2)
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
