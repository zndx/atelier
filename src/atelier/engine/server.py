"""Atelier capability/gRPC engine server.

Two faces on one port: the native ``atelier.engine.AtelierEngine`` and the shared federation face
``zndx.engine.v1.Engine`` (+ reflection). Atelier HOSTS only the capabilities configured under
``engine.capabilities`` in ``config/base.conf`` (``referee`` — the architecturally-independent judge
family) through the in-engine vLLM manager; every OTHER capability — ``instruct`` foremost, since
2026-09-08 an operating profile of the federation's Qwen3.8-27B (thinking on, effort low) — is
FORWARDED over ``zndx.engine.v1`` to the peer whose Status serves it (``forwarder.py``). No vLLM
endpoint is ever exposed to callers; no fallback from one capability to another.

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
from atelier.engine.forwarder import DEFAULT_CAPABILITY, CapabilityForwarder, NoPeerServes
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
    def __init__(self, cfg=None, forwarder=None) -> None:
        self.cfg = cfg or load_engine_config()
        self.mgr = VllmManager(self.cfg)          # HOSTED capabilities (config/base.conf engine.capabilities)
        self.fwd = forwarder or CapabilityForwarder()  # everything else → the peer whose Status serves it

    def hosts(self, capability: str) -> bool:
        return capability in (self.cfg.capabilities or {})

    def _complete(self, cap: str, prompt: str, system_prompt: str, max_tokens: int, temperature: float,
                  json_schema: str) -> dict:
        if self.hosts(cap):
            return self.mgr.complete(cap, prompt, system_prompt, max_tokens, temperature, json_schema=json_schema)
        return self.fwd.complete(cap, prompt, system_prompt, max_tokens, temperature, json_schema=json_schema)

    def Complete(self, request, context):
        cap = request.capability or DEFAULT_CAPABILITY
        try:
            out = self._complete(cap, request.prompt, request.system_prompt or "",
                                 request.max_tokens or 512, request.temperature or 0.7,
                                 request.json_schema or "")
        except NoPeerServes as e:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
        except grpc.RpcError as e:  # the serving peer's answer is the answer — no retry, no fallback
            context.abort(e.code(), f"complete[{cap}] refused by the serving peer: {e.details()}")
        except Exception as e:  # noqa: BLE001 — surface as gRPC error, keep engine up
            context.abort(grpc.StatusCode.INTERNAL, f"complete[{cap}] failed: {e}")
        return pb.CompleteResponse(
            text=out["text"], model=out["model"],
            prompt_tokens=out["prompt_tokens"],
            completion_tokens=out["completion_tokens"],
            latency_ms=out["latency_ms"],
            reasoning_content=out["reasoning_content"],
            finish_reason=out["finish_reason"])

    def EnsureEndpoint(self, request, context):
        cap = request.capability or DEFAULT_CAPABILITY
        if not self.hosts(cap):
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"atelier does not host {cap!r} — it is forwarded to the federation peer whose Status "
                f"serves it (see EngineStatus for the resolved route); hosted: {list(self.cfg.capabilities)}")
        try:
            ep = self.mgr.ensure(cap)
            return pb.EndpointStatus(
                capability=ep.capability, model=ep.spec.model, healthy=ep.healthy,
                port=ep.port, gpu_ids=ep.gpu_ids, detail=str(ep.log_path))
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.INTERNAL, f"ensure failed: {e}")

    def EngineStatus(self, request, context):
        """Native face: hosted live endpoints + the resolved federation ROUTES for forwarded ones."""
        eps = [pb.EndpointStatus(capability=e.capability, model=e.spec.model,
                                 healthy=e.healthy, port=e.port, gpu_ids=e.gpu_ids)
               for e in self.mgr.status()]
        eps += [pb.EndpointStatus(capability=r.capability, model=r.model, healthy=True, port=0,
                                  gpu_ids=list(r.gpu_ids), detail=f"forwarded → {r.label}")
                for r in self.fwd.routes() if not self.hosts(r.capability)]
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
        """Hosted capability → the in-engine manager; anything else is forwarded VERBATIM
        (tools_json / messages_json / capabilities[] ride along) to the peer whose Status serves
        it, which aligns to the capability's operating profile and reports it (``profile``)."""
        from zndx.engine.v1 import engine_pb2 as zpb
        cap = request.capability or DEFAULT_CAPABILITY
        try:
            if not self._native.hosts(cap):
                return self._native.fwd.forward(request)
            out = self._native.mgr.complete(
                cap, request.prompt, request.system_prompt or "",
                request.max_tokens or 512, request.temperature or 0.7,
                json_schema=request.json_schema or "",
            )
        except NoPeerServes as e:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))
        except grpc.RpcError as e:
            context.abort(e.code(), f"complete[{cap}] refused by the serving peer: {e.details()}")
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.INTERNAL, f"complete[{cap}] failed: {e}")
        return zpb.CompleteResponse(
            text=out["text"], model=out["model"],
            prompt_tokens=out["prompt_tokens"],
            completion_tokens=out["completion_tokens"],
            latency_ms=out["latency_ms"],
            reasoning_content=out["reasoning_content"],
            finish_reason=out["finish_reason"])

    def Status(self, request, context):
        """Advertise the capabilities this engine HOSTS (config engine.capabilities) at gRPC bind.

        Live VllmManager endpoints overlay placeholders. Forwarded capabilities (`instruct`) are
        NOT listed — an engine that forwards must not claim (capabilities.md §Operating profiles).
        Lattice accept is Status early — not vLLM cold-load (Gaius/Ægir lesson).
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

    def WatchWorkload(self, request, context):
        """Held-open intended serving set: hosted (configured) capabilities as intents, with the live
        manager state as `actual`; forwarded capabilities are not intents of THIS engine."""
        import time as _time

        from zndx.engine.v1 import engine_pb2 as zpb

        generation = 0
        while context.is_active():
            live = {e.capability: e for e in self._native.mgr.status()}
            intents = []
            for cap, spec in (self._native.cfg.capabilities or {}).items():
                e = live.get(cap)
                intents.append(zpb.WorkloadIntent(
                    capability=cap, alias=cap, model=getattr(spec, "model", ""),
                    port=int(getattr(e, "port", 0) or 0) if e else 0,
                    gpu_ids=list(getattr(e, "gpu_ids", []) or []) if e else [],
                    backend=zpb.SERVING_BACKEND_VLLM_LOCAL,
                    actual=(zpb.WORKLOAD_STATUS_SERVING if (e and e.healthy) else
                            zpb.WORKLOAD_STATUS_STARTING if e else zpb.WORKLOAD_STATUS_ABSENT)))
            yield zpb.WorkloadProfile(
                phase=zpb.WORKLOAD_PHASE_SETTLED, generation=generation, intents=intents,
                settled_at_unix_ms=int(_time.time() * 1000),
                detail="atelier: hosted capabilities load on first Complete (ABSENT until then); "
                       "instruct is forwarded, not an intent here")
            _time.sleep(15)
            generation += 1

    def Announce(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED,
                      "atelier is not a peer directory (Announce is served by Aegir)")

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
        f"(hosted: {list(servicer.cfg.capabilities)}; instruct/thinking forwarded to the federation; "
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
