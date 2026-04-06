#!/bin/bash
# Generate Python protobuf definitions and gRPC code from atelier.proto.
set -euo pipefail

uv run python -m grpc_tools.protoc \
    -I=. \
    --python_out=. \
    --grpc_python_out=. \
    --pyi_out=. \
    ./src/atelier/proto/atelier.proto

echo "Proto stubs generated."
