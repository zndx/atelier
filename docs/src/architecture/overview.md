<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

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
