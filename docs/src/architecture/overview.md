# System Overview

Atelier is a multi-service application with a gRPC core, FastAPI HTTP gateway, and React frontend.

```d2
direction: down

ui: React Frontend {
  canvas: XYFlow Agent Canvas
  atlas: embedding-atlas Viewer
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

db: SQLite State DB

ui -> gateway: REST /api/*
gateway -> grpc: gRPC :50051
grpc -> agents
grpc -> db
```
