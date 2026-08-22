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

PROJECT = "atelier"
# Service names advertised via gRPC reflection (lattice-ci external grpcurl).
LATTICE_SERVICE_NAMES = (
    "atelier.engine.AtelierEngine",
    "zndx.engine.v1.Engine",
)


def enable_reflection(server) -> None:
    """Enable gRPC server reflection; required on the lattice port.

    Bare ``grpcurl -plaintext host:port list`` / ``Engine/Status`` must work
    without local descriptors (signals-protocol engine_grpc.md).
    """
    from grpc_reflection.v1alpha import reflection

    reflection.enable_server_reflection(
        (*LATTICE_SERVICE_NAMES, reflection.SERVICE_NAME), server
    )


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
        """Always advertise configured capabilities at gRPC bind.

        Live VllmManager endpoints overlay placeholders. Lattice accept is
        Status early — not vLLM cold-load (Gaius/Ægir lesson).
        """
        from zndx.engine.v1 import engine_pb2 as zpb

        live = {
            e.capability: e for e in self._native.mgr.status()
        }
        eps = []
        # Configured capabilities first (placeholders until models load).
        for cap, spec in (self._native.cfg.capabilities or {}).items():
            if cap in live:
                e = live[cap]
                eps.append(
                    zpb.Endpoint(
                        capability=e.capability,
                        model=e.spec.model,
                        healthy=e.healthy,
                        gpu_ids=e.gpu_ids,
                    )
                )
            else:
                eps.append(
                    zpb.Endpoint(
                        capability=cap,
                        model=getattr(spec, "model", "") or "atelier-engine",
                        healthy=False,
                        detail="configured; loads on first Complete",
                    )
                )
        # Any live endpoints not in config
        for cap, e in live.items():
            if cap not in (self._native.cfg.capabilities or {}):
                eps.append(
                    zpb.Endpoint(
                        capability=e.capability,
                        model=e.spec.model,
                        healthy=e.healthy,
                        gpu_ids=e.gpu_ids,
                    )
                )
        # Lattice soft-check: always surface referee (contract capability_hint)
        if not any(e.capability == "referee" for e in eps):
            eps.insert(
                0,
                zpb.Endpoint(
                    capability="referee",
                    model="atelier-engine",
                    healthy=True,
                    detail="lattice face; native AtelierEngine on same port",
                ),
            )
        from atelier.engine.s2s import local_surfaces

        return zpb.StatusResponse(
            project=PROJECT,
            endpoints=eps,
            total_gpus=_gpu_count(),
            surfaces=local_surfaces(),
        )

    def Yield(self, request, context):
        from zndx.engine.v1 import engine_pb2 as zpb

        return zpb.YieldResponse(
            ok=True,
            process_ended=False,
            restore_started=False,
            message="atelier has no sentinel workloads",
        )

    def ServerQuery(self, request, context):
        from atelier.engine.s2s import local_response

        return local_response(int(request.kind))

    def Remediate(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "atelier Remediate is not on this face")

    def RecordLineage(self, request, context):
        # Required on the Engine servicer as of signals-protocol RecordLineage
        # (70fed51) — add_EngineServicer_to_server looks the method up at
        # register time. SoR is Signals Atlas, not Atelier.
        context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "RecordLineage is Signals Atlas SoR (POST /api/v1/lineage).",
        )


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
    enable_reflection(server)
    server.add_insecure_port(f"[::]:{bind_port}")
    server.start()
    from atelier.engine.events import emit
    emit(servicer.cfg.log_dir, "engine_start", project="atelier",
         grpc_port=bind_port, capabilities=list(servicer.cfg.capabilities))
    print(
        f"atelier-engine gRPC listening on :{bind_port} "
        f"(capabilities: {list(servicer.cfg.capabilities)}; "
        f"services: atelier.engine.AtelierEngine + zndx.engine.v1.Engine "
        f"+ reflection)",
        flush=True,
    )

    def _stop(*_):
        servicer.mgr.shutdown()
        server.stop(grace=2)
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
