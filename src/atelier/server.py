"""Atelier gRPC server startup."""

import logging
import os
from concurrent import futures

import grpc

from atelier.config import load_config
from atelier.proto import atelier_pb2_grpc
from atelier.service import AtelierServicer

log = logging.getLogger(__name__)


def _bootstrap_db(cfg):
    """Run database migrations on CAI. PGlite is started externally by start-app.sh."""
    if cfg.is_cml:
        from atelier.db.bootstrap import run_migrations
        run_migrations(cfg.db_url)
        log.info("Migrations applied: %s", cfg.db_url)


def serve(blocking: bool = True) -> grpc.Server:
    """Start the gRPC server.

    Args:
        blocking: If True, block until termination.

    Returns:
        The gRPC server instance.
    """
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    _bootstrap_db(cfg)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=cfg.grpc_max_workers),
    )
    atelier_pb2_grpc.add_AtelierServicer_to_server(
        AtelierServicer(), server,
    )
    bound = server.add_insecure_port(f"[::]:{cfg.grpc_port}")
    if bound == 0:
        # grpc returns 0 on bind failure and start() would proceed portless —
        # a co-tenant (e.g. the Gaius engine on 50051) squatting our port
        # must be a loud crash, not a silent no-op the readiness probe
        # false-positives against.
        raise RuntimeError(
            f"gRPC bind failed on port {cfg.grpc_port} — already in use? "
            f"Set ATELIER_GRPC_PORT to a free port (local devenv pins 50071)."
        )
    server.start()
    print(f"Atelier gRPC server listening on port {cfg.grpc_port}")

    if blocking:
        server.wait_for_termination()

    return server


if __name__ == "__main__":
    serve()
