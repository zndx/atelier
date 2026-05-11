# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

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
    server.add_insecure_port(f"[::]:{cfg.grpc_port}")
    server.start()
    print(f"Atelier gRPC server listening on port {cfg.grpc_port}")

    if blocking:
        server.wait_for_termination()

    return server


if __name__ == "__main__":
    serve()
