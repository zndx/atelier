# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Atelier gRPC client wrapper with error handling.

Mirrors the Fine Tuning Studio pattern: wraps the generated stub
with automatic error handling and reads endpoint from config.
"""

import grpc

from atelier.config import load_config
from atelier.proto.atelier_pb2_grpc import AtelierStub


class AtelierClient:
    """gRPC client for communicating with the Atelier server."""

    def __init__(self, host: str | None = None, port: int | None = None):
        cfg = load_config()
        host = host or "localhost"
        port = port or cfg.grpc_port
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = AtelierStub(self.channel)

    def _handle_grpc_error(self, error: grpc.RpcError) -> str:
        error_message = error.details()
        if error_message and error_message.startswith(
            "Exception calling application:"
        ):
            error_message = error_message.replace(
                "Exception calling application:", ""
            ).strip()
        return error_message or str(error)
