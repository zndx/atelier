#!/bin/bash
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

# Generate Python protobuf definitions and gRPC code from atelier.proto.
set -euo pipefail

uv run python -m grpc_tools.protoc \
    -I=. \
    --python_out=. \
    --grpc_python_out=. \
    --pyi_out=. \
    ./src/atelier/proto/atelier.proto

# Fix generated imports: grpc_tools generates `from src.atelier.proto import ...`
# because the proto file lives under src/. We need `from atelier.proto import ...`
# since src/ is the package root (via pyproject.toml src layout).
GRPC_FILE="src/atelier/proto/atelier_pb2_grpc.py"
if [ -f "$GRPC_FILE" ]; then
    sed -i 's/from src\.atelier\.proto/from atelier.proto/g' "$GRPC_FILE"
    sed -i 's/src_dot_atelier_dot_proto_dot_/atelier_dot_proto_dot_/g' "$GRPC_FILE"
fi

echo "Proto stubs generated."
