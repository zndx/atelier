# System Overview

Atelier is a multi-service application with a gRPC core, FastAPI HTTP gateway, and React frontend.

```d2
direction: down

ui: React Frontend {
  canvas: XYFlow Agent Canvas
  embeddings: Embeddings
}

gateway: FastAPI HTTP Gateway {
  tooltip: "Serves React build\nBridges REST → gRPC"
}

grpc: gRPC Core Service {
  tooltip: "Proto-first API\nAgent orchestration\nData management"
}

agents: Claude Agent SDK {
  tooltip: "Keystone agents\nAdaptive workflows"
}

db: PostgreSQL {
  tooltip: "devenv: PG 16 + pgvector\nCAI: PGlite (Node.js)"
}

ui -> gateway: REST /api/*
gateway -> grpc: gRPC :50051
grpc -> agents
grpc -> db
```

On the shared lab host the product servicer is **:50071** (Gaius holds
**:50051**). The Signals lattice engine is a separate process on **:50251**
(`python -m atelier.engine.server`). See [gRPC & Gateway](./grpc.md) and
[Signals Peer Unit](../operations/peer-unit.md).
