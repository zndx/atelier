# Introduction

Atelier is an agentic classification workbench for Cloudera AI. It combines the Claude Agent SDK for adaptive keystone-agent orchestration with an Embeddings for interactive visualization of classification results produced by the [signals](https://github.com/rch/signals) pipeline.

## Key Components

- **gRPC Core Service** — Proto-first API following the Fine Tuning Studio pattern
- **React Frontend** — Ant Design UI with XYFlow canvas for agent workflow visualization
- **Claude Agent SDK** — Keystone agents that evolve alongside classification workflows
- **Embeddings** — Interactive parquet visualization of classification output (powered by [embedding-atlas](https://github.com/apple/embedding-atlas))
- **HOCON Configuration** — Single source of truth with environment variable substitution

## Deployment

Atelier deploys as a Cloudera AI Application from a Git URL. It can also run locally via `devenv up` for development.
