# gRPC Service

The core service follows the Fine Tuning Studio proto-first pattern.

## Proto Definition

The service contract lives in `src/atelier/proto/atelier.proto`. All API types are generated from this single source of truth.

## Architecture Layers

1. **Proto** (`atelier.proto`) — Service contract and message definitions
2. **Servicer** (`service.py`) — Thin router dispatching to business logic
3. **Client** (`client.py`) — Wrapper around generated stub with error handling
4. **Gateway** (`gateway.py`) — FastAPI bridge from REST to gRPC

## Generating Stubs

```bash
just proto
```

This runs `bin/generate-proto.sh` which invokes `grpc_tools.protoc` to produce `_pb2.py`, `_pb2_grpc.py`, and `.pyi` files.
