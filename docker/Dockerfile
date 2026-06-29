# syntax=docker/dockerfile:1

# ─────────────────────────────────────────────────────────────────────
# Stage 1 — build the React (ui-next) frontend into ui/dist.
# Vite's build.outDir is ../ui/dist, so we recreate the repo layout
# (ui-next + external tarball + ui/) and let the build emit there.
# ─────────────────────────────────────────────────────────────────────
FROM node:20-bookworm-slim AS ui-builder
WORKDIR /build

# The embedding-atlas dependency is a committed local tarball referenced
# as file:../external/embedding-atlas-0.18.1.tgz from ui-next.
COPY external/embedding-atlas-0.18.1.tgz ./external/embedding-atlas-0.18.1.tgz
COPY ui-next/package.json ui-next/package-lock.json* ./ui-next/
RUN mkdir -p ui && cd ui-next && npm install

COPY ui-next ./ui-next
# npm install above ran ui-next's postinstall, populating ui/public/ghostty.
RUN cd ui-next && npm run build
# Result: /build/ui/dist  (+ /build/ui/public/ghostty for the gateway mount)


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — python runtime: gRPC core + FastAPI gateway (serves ui/dist).
# ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

# System libs needed by the ML stack (torch/catboost/sentence-transformers,
# docling) and runtime tooling (git for any VCS deps, curl for healthcheck).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

# Install Python deps first (cached unless pyproject/src change).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e .

# Pre-cache the embedding model so first-run startup is fast and offline-safe.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# pyproject pins bare `psycopg` (pure-python), which needs a system libpq
# that the slim base image doesn't ship. Install the binary build so a
# working libpq is bundled — otherwise every DB connection raises
# "no pq wrapper available" and the gateway never starts. (Appended after
# the model cache to preserve the expensive dependency layers above.)
RUN pip install "psycopg[binary]>=3.2.0"

# Remaining backend sources needed at runtime.
COPY config ./config
COPY db ./db
COPY bin ./bin
COPY scripts ./scripts
COPY data ./data
COPY docker/entrypoint.sh ./docker/entrypoint.sh

# Built frontend + ghostty assets from stage 1.
COPY --from=ui-builder /build/ui ./ui

# Writable runtime dirs (config materialization, parquet, scratch).
RUN mkdir -p build/data build/config .app && chmod +x docker/entrypoint.sh

EXPOSE 8090 50051
CMD ["bash", "docker/entrypoint.sh"]
