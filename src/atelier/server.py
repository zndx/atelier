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
    """Bootstrap embedded PostgreSQL + migrations on CAI when no DB URL is configured."""
    if os.environ.get("ATELIER_DB_URL"):
        return  # Explicit URL set — nothing to bootstrap

    if cfg.is_cml:
        from atelier.db.bootstrap import ensure_database, run_migrations
        db_url = ensure_database()
        cfg.db_url = db_url
        run_migrations(db_url)
        log.info("Embedded PostgreSQL bootstrapped for CML: %s", db_url)


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
    server.add_insecure_port(f"[::]:{cfg.grpc_port}")
    server.start()
    print(f"Atelier gRPC server listening on port {cfg.grpc_port}")

    if blocking:
        server.wait_for_termination()

    return server


if __name__ == "__main__":
    serve()
