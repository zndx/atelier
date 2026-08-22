#!/bin/bash
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

# Shared federation face (signals-protocol). Stubs live under src/zndx/.
uv run python -m grpc_tools.protoc \
    -I=external/signals-protocol/proto \
    --python_out=src \
    --grpc_python_out=src \
    --pyi_out=src \
    zndx/engine/v1/engine.proto

if [[ -f external/signals-protocol/proto/zndx/scheduler/v1/scheduler.proto ]]; then
    uv run python -m grpc_tools.protoc \
        -I=external/signals-protocol/proto \
        --python_out=src \
        --grpc_python_out=src \
        --pyi_out=src \
        zndx/scheduler/v1/scheduler.proto
    mkdir -p src/zndx/scheduler/v1
    printf '%s\n' \
        '"""Generated bindings for zndx.scheduler.v1 (signals-protocol)."""' \
        'from . import scheduler_pb2, scheduler_pb2_grpc' \
        '__all__ = ["scheduler_pb2", "scheduler_pb2_grpc"]' \
        > src/zndx/scheduler/v1/__init__.py
    touch src/zndx/scheduler/__init__.py
fi

echo "Proto stubs generated."
