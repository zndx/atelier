"""Atelier gRPC server startup."""

from concurrent import futures

import grpc

from atelier.config import load_config
from atelier.proto import atelier_pb2_grpc
from atelier.service import AtelierServicer


def serve(blocking: bool = True) -> grpc.Server:
    """Start the gRPC server.

    Args:
        blocking: If True, block until termination.

    Returns:
        The gRPC server instance.
    """
    cfg = load_config()
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
